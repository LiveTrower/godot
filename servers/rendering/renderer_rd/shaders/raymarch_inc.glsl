// Copyright (c) 2023 Tomasz Stachowiak
//
// This contribution is dual licensed under EITHER OF
//
//     Apache License, Version 2.0, (http://www.apache.org/licenses/LICENSE-2.0)
//     MIT license (http://opensource.org/licenses/MIT)
//
// at your option.
//
// This is a port of the Bevy's [`raymarch.wgsl`] to GLSL.
//
// [`raymarch.hlsl`]:
// https://gist.github.com/h3r2tic/9c8356bdaefbe80b1a22ae0aaee192db

#ifdef USE_MULTIVIEW
layout(set = 1, binding = 38) uniform texture2DArray depth_texture;
#else
layout(set = 1, binding = 38) uniform texture2D depth_texture;
#endif // USE_MULTIVIEW

struct HybridRootFinder {
	uint linear_steps;
	uint bisection_steps;
	bool use_secant;
	float linear_march_exponent;

	float jitter;
	float min_t;
	float max_t;
};

HybridRootFinder hybrid_root_finder_new_with_linear_steps(uint v) {
	HybridRootFinder res;
	res.linear_steps = v;
	res.bisection_steps = 0u;
	res.use_secant = false;
	res.linear_march_exponent = 1.0;
	res.jitter = 1.0;
	res.min_t = 0.0;
	res.max_t = 1.0;
	return res;
}

struct DistanceWithPenetration {
	/// Distance to the surface of which a root we're trying to find
	float distance;

	/// Whether to consider this sample valid for intersection.
	/// Mostly relevant for allowing the ray marcher to travel behind surfaces,
	/// as it will mark surfaces it travels under as invalid.
	bool valid;

	/// Conservative estimate of depth to which the ray penetrates the marched surface.
	float penetration;
};

struct DepthRaymarchDistanceFn {
	vec2 depth_tex_size;

	bool march_behind_surfaces;
	float depth_thickness;

	bool use_sloppy_march;
};

vec2 ndc_to_uv(vec2 ndc) {
	return ndc * 0.5 + 0.5;
}

DistanceWithPenetration depth_raymarch_distance_fn_evaluate(
		inout DepthRaymarchDistanceFn distance_fn,
		vec3 ray_point_cs) {
	vec2 interp_uv = ndc_to_uv(ray_point_cs.xy);

	float ray_depth = 1.0 / ray_point_cs.z;

	// We're using both point-sampled and bilinear-filtered values from the depth buffer.
	//
	// That's really stupid but works like magic. For samples taken near the ray origin,
	// the discrete nature of the depth buffer becomes a problem. It's not a land of continuous surfaces,
	// but a bunch of stacked duplo bricks.
	//
	// Technically we should be taking discrete steps in distance_fn duplo land, but then we're at the mercy
	// of arbitrary quantization of our directions -- and sometimes we'll take a step which would
	// claim that the ray is occluded -- even though the underlying smooth surface wouldn't occlude it.
	//
	// If we instead take linear taps from the depth buffer, we reconstruct the linear surface.
	// That fixes acne, but introduces false shadowing near object boundaries, as we now pretend
	// that everything is shrink-wrapped by distance_fn continuous 2.5D surface, and our depth thickness
	// heuristic ends up falling apart.
	//
	// The fix is to consider both the smooth and the discrete surfaces, and only claim occlusion
	// when the ray descends below both.
	//
	// The two approaches end up fixing each other's artifacts:
	// * The false occlusions due to duplo land are rejected because the ray stays above the smooth surface.
	// * The shrink-wrap surface is no longer continuous, so it's possible for rays to miss it.

#ifdef USE_MULTIVIEW
	float linear_depth = 1.0 / textureLod(sampler2DArray(depth_texture, SAMPLER_LINEAR_CLAMP), vec3(interp_uv, ViewIndex), 0).r;
	float unfiltered_depth = 1.0 / textureLod(sampler2DArray(depth_texture, SAMPLER_NEAREST_CLAMP), vec3(interp_uv, ViewIndex), 0).r;
#else
	float linear_depth = 1.0 / textureLod(sampler2D(depth_texture, SAMPLER_LINEAR_CLAMP), interp_uv, 0).r;
	float unfiltered_depth = 1.0 / textureLod(sampler2D(depth_texture, SAMPLER_NEAREST_CLAMP), interp_uv, 0).r;
#endif // USE_MULTIVIEW

	float max_depth;
	float min_depth;

	if (distance_fn.use_sloppy_march) {
		max_depth = unfiltered_depth;
		min_depth = unfiltered_depth;
	} else {
		max_depth = max(linear_depth, unfiltered_depth);
		min_depth = min(linear_depth, unfiltered_depth);
	}

	float bias = 0.000002;

	DistanceWithPenetration res;
	res.distance = max_depth * (1.0 + bias) - ray_depth;

	// distance_fn will be used at the end of the ray march to potentially discard the hit.
	res.penetration = ray_depth - min_depth;

	if (distance_fn.march_behind_surfaces) {
		res.valid = res.penetration < distance_fn.depth_thickness;
	} else {
		res.valid = true;
	}

	return res;
}

bool hybrid_root_finder_find_root(
		inout HybridRootFinder root_finder,
		vec3 start,
		vec3 end,
		inout DepthRaymarchDistanceFn distance_fn,
		out float hit_t,
		out float miss_t,
		out DistanceWithPenetration hit_d) {
	vec3 dir = end - start;

	float min_t = root_finder.min_t;
	float max_t = root_finder.max_t;

	DistanceWithPenetration min_d = DistanceWithPenetration(0.0, false, 0.0);
	DistanceWithPenetration max_d = DistanceWithPenetration(0.0, false, 0.0);

	float step_size = (max_t - min_t) / float(root_finder.linear_steps);

	bool intersected = false;

	//
	// Ray march using linear steps

	if (root_finder.linear_steps > 0u) {
		float candidate_t = mix(
				min_t,
				max_t,
				pow(
						root_finder.jitter / float(root_finder.linear_steps),
						root_finder.linear_march_exponent));

		vec3 candidate = start + dir * candidate_t;
		DistanceWithPenetration candidate_d = depth_raymarch_distance_fn_evaluate(distance_fn, candidate);
		intersected = candidate_d.distance < 0.0 && candidate_d.valid;

		if (intersected) {
			max_t = candidate_t;
			max_d = candidate_d;
			// The `[min_t .. max_t]` interval contains an intersection. End the linear search.
		} else {
			// No intersection yet. Carry on.
			min_t = candidate_t;
			min_d = candidate_d;

			for (uint step = 1u; step < root_finder.linear_steps; step += 1u) {
				float candidate_t = mix(
						root_finder.min_t,
						root_finder.max_t,
						pow(
								(float(step) + root_finder.jitter) / float(root_finder.linear_steps),
								root_finder.linear_march_exponent));

				vec3 candidate = start + dir * candidate_t;
				DistanceWithPenetration candidate_d = depth_raymarch_distance_fn_evaluate(distance_fn, candidate);
				intersected = candidate_d.distance < 0.0 && candidate_d.valid;

				if (intersected) {
					max_t = candidate_t;
					max_d = candidate_d;
					// The `[min_t .. max_t]` interval contains an intersection.
					// End the linear search.
					break;
				} else {
					// No intersection yet. Carry on.
					min_t = candidate_t;
					min_d = candidate_d;
				}
			}
		}
	}

	miss_t = min_t;
	hit_t = min_t;

	//
	// Refine the hit using bisection

	if (intersected) {
		for (uint step = 0u; step < root_finder.bisection_steps; step += 1u) {
			float mid_t = (min_t + max_t) * 0.5;
			vec3 candidate = start + dir * mid_t;
			DistanceWithPenetration candidate_d = depth_raymarch_distance_fn_evaluate(distance_fn, candidate);

			if (candidate_d.distance < 0.0 && candidate_d.valid) {
				// Intersection at the mid point. Refine the first half.
				max_t = mid_t;
				max_d = candidate_d;
			} else {
				// No intersection yet at the mid point. Refine the second half.
				min_t = mid_t;
				min_d = candidate_d;
			}
		}

		if (root_finder.use_secant) {
			// Finish with one application of the secant method
			float total_d = min_d.distance + -max_d.distance;

			float mid_t = mix(min_t, max_t, min_d.distance / total_d);
			vec3 candidate = start + dir * mid_t;
			DistanceWithPenetration candidate_d = depth_raymarch_distance_fn_evaluate(distance_fn, candidate);

			// Only accept the result of the secant method if it improves upon
			// the previous result.
			//
			// Technically root_finder should be `abs(candidate_d.distance) <
			// min(min_d.distance, -max_d.distance) * frac`, but root_finder seems
			// sufficient.
			if (abs(candidate_d.distance) < min_d.distance * 0.9 && candidate_d.valid) {
				hit_t = mid_t;
				hit_d = candidate_d;
			} else {
				hit_t = max_t;
				hit_d = max_d;
			}

			return true;
		} else {
			hit_t = max_t;
			hit_d = max_d;
			return true;
		}
	} else {
		// Mark the conservative miss distance.
		hit_t = min_t;
		return false;
	}
}

struct DepthRayMarchResult {
	bool hit;
	float hit_t;
	vec2 hit_uv;
	float hit_penetration;
	float hit_penetration_frac;
};

struct DepthRayMarch {
	uint linear_steps;
	float linear_march_exponent;
	uint bisection_steps;
	bool use_secant;
	float jitter;
	vec3 ray_start_cs;
	vec3 ray_end_cs;
	bool march_behind_surfaces;
	bool use_sloppy_march;
	float depth_thickness_linear_z;
	vec2 depth_tex_size;
};

DepthRayMarch depth_ray_march_new_from_depth(vec2 depth_tex_size) {
	DepthRayMarch res;
	res.jitter = 1.0;
	res.linear_steps = 4u;
	res.bisection_steps = 0u;
	res.linear_march_exponent = 1.0;
	res.depth_tex_size = depth_tex_size;
	res.depth_thickness_linear_z = 1.0;
	res.march_behind_surfaces = false;
	res.use_sloppy_march = false;
	return res;
}

void depth_ray_march_to_cs_dir_impl(
		inout DepthRayMarch raymarch,
		vec4 dir_cs,
		bool infinite) {
	vec4 end_cs = vec4(raymarch.ray_start_cs, 1.0) + dir_cs;

	// Perform perspective division, but avoid dividing by zero for rays
	// heading directly towards the eye.
	end_cs /= (end_cs.w >= 0.0 ? 1.0 : -1.0) * max(1e-10, abs(end_cs.w));

	// Clip ray start to the view frustum
	vec3 delta_cs = end_cs.xyz - raymarch.ray_start_cs;
	vec3 near_edge = mix(vec3(-1.0, -1.0, 0.0), vec3(1.0, 1.0, 1.0), lessThan(delta_cs, vec3(0.0)));
	vec3 dist_to_near_edge = (near_edge - raymarch.ray_start_cs) / delta_cs;
	float max_dist_to_near_edge = max(dist_to_near_edge.x, dist_to_near_edge.y);
	raymarch.ray_start_cs += delta_cs * max(0.0, max_dist_to_near_edge);

	// Clip ray end to the view frustum

	delta_cs = end_cs.xyz - raymarch.ray_start_cs;
	vec3 far_edge = mix(vec3(-1.0, -1.0, 0.0), vec3(1.0, 1.0, 1.0), greaterThanEqual(delta_cs, vec3(0.0)));
	vec3 dist_to_far_edge = (far_edge - raymarch.ray_start_cs) / delta_cs;
	float min_dist_to_far_edge = min(
			min(dist_to_far_edge.x, dist_to_far_edge.y),
			dist_to_far_edge.z);

	if (infinite) {
		delta_cs *= min_dist_to_far_edge;
	} else {
		// If unbounded, would make the ray reach the end of the frustum
		delta_cs *= min(1.0, min_dist_to_far_edge);
	}

	raymarch.ray_end_cs = raymarch.ray_start_cs + delta_cs;
}

/// March from a clip-space position (w = 1)
void depth_ray_march_from_cs(inout DepthRayMarch raymarch, vec3 v) {
	raymarch.ray_start_cs = v;
}

/// March to a clip-space position (w = 1)
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
void depth_ray_march_to_cs(inout DepthRayMarch raymarch, vec3 end_cs) {
	vec4 dir = vec4(end_cs - raymarch.ray_start_cs, 0.0) * sign(end_cs.z);
	depth_ray_march_to_cs_dir_impl(raymarch, dir, false);
}

/// March towards a clip-space direction. Infinite (ray is extended to cover the whole view frustum).
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
void depth_ray_march_to_cs_dir(inout DepthRayMarch raymarch, vec4 dir) {
	depth_ray_march_to_cs_dir_impl(raymarch, dir, true);
}

/// March to a world-space position.
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
void depth_ray_march_to_ws(inout DepthRayMarch raymarch, vec3 end) {
	depth_ray_march_to_cs(raymarch, end);
}

/// March towards a world-space direction. Infinite (ray is extended to cover the whole view frustum).
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
void depth_ray_march_to_ws_dir(inout DepthRayMarch raymarch, vec4 dir_clip) {
	depth_ray_march_to_cs_dir_impl(raymarch, dir_clip, true);
}

/// Perform the ray march.
DepthRayMarchResult depth_ray_march_march(inout DepthRayMarch raymarch) {
	DepthRayMarchResult res = DepthRayMarchResult(false, 0.0, vec2(0.0), 0.0, 0.0);

	vec2 ray_start_uv = ndc_to_uv(raymarch.ray_start_cs.xy);
	vec2 ray_end_uv = ndc_to_uv(raymarch.ray_end_cs.xy);

	vec2 ray_uv_delta = ray_end_uv - ray_start_uv;
	vec2 ray_len_px = ray_uv_delta * raymarch.depth_tex_size;

	uint min_px_per_step = 1u;
	int step_count = max(
			2,
			min(int(raymarch.linear_steps), int(floor(length(ray_len_px) / float(min_px_per_step)))));

	float linear_z_to_scaled_linear_z = 1.0 / scene_data_block.data.z_near;
	float depth_thickness = raymarch.depth_thickness_linear_z * linear_z_to_scaled_linear_z;

	DepthRaymarchDistanceFn distance_fn;
	distance_fn.depth_tex_size = raymarch.depth_tex_size;
	distance_fn.march_behind_surfaces = raymarch.march_behind_surfaces;
	distance_fn.depth_thickness = depth_thickness;
	distance_fn.use_sloppy_march = raymarch.use_sloppy_march;

	DistanceWithPenetration hit;

	float hit_t = 0.0;
	float miss_t = 0.0;
	HybridRootFinder root_finder = hybrid_root_finder_new_with_linear_steps(uint(step_count));
	root_finder.bisection_steps = raymarch.bisection_steps;
	root_finder.use_secant = raymarch.use_secant;
	root_finder.linear_march_exponent = raymarch.linear_march_exponent;
	root_finder.jitter = raymarch.jitter;

	bool intersected = hybrid_root_finder_find_root(
			root_finder,
			raymarch.ray_start_cs,
			raymarch.ray_end_cs,
			distance_fn,
			hit_t,
			miss_t,
			hit);

	res.hit_t = hit_t;

	if (intersected && hit.penetration < depth_thickness && hit.distance < depth_thickness) {
		res.hit = true;
		res.hit_uv = mix(ray_start_uv, ray_end_uv, res.hit_t);
		res.hit_penetration = hit.penetration / linear_z_to_scaled_linear_z;
		res.hit_penetration_frac = hit.penetration / depth_thickness;
		return res;
	}

	res.hit_t = miss_t;
	res.hit_uv = mix(ray_start_uv, ray_end_uv, res.hit_t);

	return res;
}

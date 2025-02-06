#include "brdf_inc.glsl"

#extension GL_EXT_control_flow_attributes : require

// This annotation macro must be placed before any loops that rely on specialization constants as their upper bound.
// Drivers may choose to unroll these loops based on the possible range of the value that can be deduced from the
// spec constant, which can lead to their code generation taking a much longer time than desired.
#ifdef UBERSHADER
// Prefer to not unroll loops on the ubershader to reduce code size as much as possible.
#define SPEC_CONSTANT_LOOP_ANNOTATION [[dont_unroll]]
#else
// Don't make an explicit hint on specialized shaders.
#define SPEC_CONSTANT_LOOP_ANNOTATION
#endif

vec3 specular_lobe(float metallic, vec3 f0, float anisotropy, vec3 T, vec3 B, float roughness, vec3 H, vec3 V, vec3 L, float cNdotH, float cNdotL, float cNdotV, float cLdotH, vec3 energy_compensation){
	float alpha_ggx = roughness * roughness;
#if defined(LIGHT_ANISOTROPY_USED)
	float aspect = sqrt_IEEE_int_approximation(1.0 - abs(anisotropy) * 0.9);
	float ax = alpha_ggx / aspect;
	float ay = alpha_ggx * aspect;
	if (anisotropy < 0.0) {
		float atemp = ax;
		ax = ay;
		ay = atemp;
	}
	float TdotV = dot(T, V);
	float BdotV = dot(B, V);
	float TdotL = dot(T, L);
	float BdotL = dot(B, L);
	float TdotH = dot(T, H);
	float BdotH = dot(B, H);

	float D = D_GGX_anisotropic(cNdotH, ax, ay, TdotH, BdotH);
	float G = V_GGX_anisotropic(ax, ay, TdotV, TdotL, BdotV, BdotL, cNdotV, cNdotL);
#else // isotropic
	float D = D_GGX(cNdotH, alpha_ggx);
	float G = V_GGX(cNdotL, cNdotV, alpha_ggx);
#endif
	// Calculate Fresnel using specular occlusion term from Filament:
	// https://google.github.io/filament/Filament.html#lighting/occlusion/specularocclusion
	float f90 = clamp(dot(f0, vec3(50.0 * 0.33)), metallic, 1.0);
	vec3 F = SchlickFresnel(f0, f90, cLdotH);
	return energy_compensation * (D * G * F * cNdotL);
}

vec3 sheen_lobe(float sheen_roughness, float sheen, vec3 sheen_color, float cNdotH, float cNdotV, float cNdotL, out float attenuation){
	float D = D_Charlie(sheen_roughness, cNdotH);
	float V = V_Neubelt(cNdotV, cNdotL);
	float dfg_sheen = prefiltered_dfg(sheen_roughness, cNdotV).z;
	// Albedo scaling of the base layer before we layer sheen on top
	attenuation = 1.0 - max3(sheen_color) * dfg_sheen * sheen;
	return D * V * sheen_color * sheen * cNdotL;
}

float clearcoat_lobe(float clearcoat_roughness, float clearcoat, float ccNdotH, float cLdotH, float ccNdotL, out float attenuation){
	float D = D_GGX(ccNdotH, mix(0.001, 0.1, clearcoat_roughness));
	float V = V_Kelemen(cLdotH);
	float F = clearcoat * SchlickFresnel(0.04, 0.96, cLdotH);
	// The clear coat layer assumes an IOR of 1.5 (4% reflectance)
	attenuation = 1.0 - F * clearcoat;
	return clearcoat * D * V * F * ccNdotL;
}

void light_compute(vec3 N, vec3 L, vec3 V, float A, vec3 light_color, bool is_directional, float attenuation, vec3 f0, uint orms, float specular_amount, vec3 albedo, inout float alpha, vec2 screen_uv,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
		float transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_SHEEN_USED
		float sheen, float sheen_roughness, vec3 sheen_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 T, vec3 B, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {
	vec4 orms_unpacked = unpackUnorm4x8(orms);
	float roughness = orms_unpacked.y;
	float metallic = orms_unpacked.z;

#if defined(LIGHT_CODE_USED)
	// Light is written by the user shader.
	mat4 inv_view_matrix = scene_data_block.data.inv_view_matrix;
	mat4 read_view_matrix = scene_data_block.data.view_matrix;

#ifdef USING_MOBILE_RENDERER
	mat4 read_model_matrix = instances.data[draw_call.instance_index].transform;
#else
	mat4 read_model_matrix = instances.data[instance_index_interp].transform;
#endif

#undef projection_matrix
#define projection_matrix scene_data_block.data.projection_matrix
#undef inv_projection_matrix
#define inv_projection_matrix scene_data_block.data.inv_projection_matrix

	vec2 read_viewport_size = scene_data_block.data.viewport_size;
	vec3 normal = N;
	vec3 light = L;
	vec3 view = V;

#CODE : LIGHT
#else // !LIGHT_CODE_USED
	float NdotL = min(A + dot(N, L), 1.0);
	float cNdotV = max(dot(N, V), 1e-4);

#ifdef LIGHT_TRANSMITTANCE_USED
	{
#ifdef SSS_MODE_SKIN
		float scale = 8.25 / transmittance_depth;
		float d = scale * abs(transmittance_z);
		float dd = -d * d;
		vec3 profile = vec3(0.233, 0.455, 0.649) * exp(dd / 0.0064) +
				vec3(0.1, 0.336, 0.344) * exp(dd / 0.0484) +
				vec3(0.118, 0.198, 0.0) * exp(dd / 0.187) +
				vec3(0.113, 0.007, 0.007) * exp(dd / 0.567) +
				vec3(0.358, 0.004, 0.0) * exp(dd / 1.99) +
				vec3(0.078, 0.0, 0.0) * exp(dd / 7.41);

		diffuse_light += profile * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI);

#else
		float scale = 8.25 / transmittance_depth;
		float d = scale * abs(transmittance_z);
		float dd = -d * d;
		diffuse_light += exp(dd) * transmittance_color.rgb * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI);
#endif
	}
#endif //LIGHT_TRANSMITTANCE_USED

	// We skip checking on attenuation on directional lights to avoid a branch that is not as beneficial for directional lights as the other ones.
	const float EPSILON = 1e-6f;
	if (is_directional || attenuation > EPSILON) {
		float cNdotL = max(NdotL, 0.0);

		//Multiscattering
		vec2 dfg = prefiltered_dfg(roughness, cNdotV).xy;
		vec3 energy_compensation = get_energy_compensation(f0, dfg.y);

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED) || defined(LIGHT_SHEEN_USED)
		vec3 H = normalize(V + L);
		float cLdotH = clamp(A + dot(L, H), 0.0, 1.0);
#endif

#if defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_SHEEN_USED)
		float cNdotH = clamp(A + dot(N, H), 0.0, 1.0);
#endif

		//Specular Light
		float cc_attenuation = 1.0;
		float sh_attenuation = 1.0;
#if defined(LIGHT_SHEEN_USED)
		vec3 sheen_specular_brdf_NL = sheen_lobe(sheen_roughness, sheen, sheen_color, cNdotH, cNdotV, cNdotL, sh_attenuation);
		specular_light += sheen_specular_brdf_NL * light_color * attenuation * specular_amount;
#endif

#if defined(LIGHT_CLEARCOAT_USED)
		// Clearcoat ignores normal_map, use vertex normal instead
		float ccNdotL = clamp(A + dot(vertex_normal, L), 0.0, 1.0);
		float ccNdotH = clamp(A + dot(vertex_normal, H), 0.0, 1.0);
		float clearcoat_specular_brdf_NL = clearcoat_lobe(clearcoat_roughness, clearcoat, ccNdotH, cLdotH, ccNdotL, cc_attenuation);
		specular_light += clearcoat_specular_brdf_NL * light_color * attenuation * specular_amount;
#endif

		if (roughness > 0.0) {
#if defined(SPECULAR_TOON)

			vec3 R = normalize(-reflect(L, N));
			float RdotV = dot(R, V);
			float mid = 1.0 - roughness;
			mid *= mid;
			float intensity = smoothstep(mid - roughness * 0.5, mid + roughness * 0.5, RdotV) * mid;
			diffuse_light += light_color * intensity * attenuation * specular_amount; // write to diffuse_light, as in toon shading you generally want no reflection

#elif defined(SPECULAR_DISABLED)
			// none..

#elif defined(SPECULAR_SCHLICK_GGX)
			// shlick+ggx as default
			vec3 specular_brdf_NL = specular_lobe(metallic, f0, 
#if defined(LIGHT_ANISOTROPY_USED)
			anisotropy, T, B,
#else
			0.0, vec3(0.0), vec3(0.0),
#endif
			roughness, H, V, L, cNdotH, cNdotL, cNdotV, cLdotH, energy_compensation);

			specular_light += specular_brdf_NL * light_color * attenuation * cc_attenuation * sh_attenuation * specular_amount;
#endif
		}

		//Diffuse Light
		if (metallic < 1.0) {
			float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance

#if defined(DIFFUSE_LAMBERT_WRAP)
			diffuse_brdf_NL = Diffuse_Lambert_Wrap(NdotL, roughness);
#elif defined(DIFFUSE_TOON)
			diffuse_brdf_NL = Diffuse_Toon(NdotL, roughness);
#elif defined(DIFFUSE_BURLEY)
			diffuse_brdf_NL = Normalized_Diffuse_Burley(cNdotV, cNdotL, cLdotH, roughness);
#else
			// lambert
			diffuse_brdf_NL = Diffuse_Lambert(cNdotL);
#endif
			//apply diffuse
			diffuse_light += light_color * diffuse_brdf_NL * attenuation;
#if defined(LIGHT_BACKLIGHT_USED)
			diffuse_light += light_color * ((1.0 - diffuse_brdf_NL) * backlight * attenuation * cc_attenuation * sh_attenuation * (1.0 / M_PI));
#endif
#if defined(LIGHT_RIM_USED)
			// Epsilon min to prevent pow(0, 0) singularity which results in undefined behavior.
			float rim_light = pow(max(1e-4, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
			diffuse_light += (rim_light * rim * attenuation * cc_attenuation * sh_attenuation * (1.0 / M_PI)) * mix(vec3(1.0) / (albedo + 0.000001), vec3(1.0), rim_tint) * light_color;
#endif
		}
#ifdef USE_SHADOW_TO_OPACITY
			alpha = min(alpha, clamp(1.0 - attenuation, 0.0, 1.0));
#endif
	}
#endif // LIGHT_CODE_USED
}

#ifndef SHADOWS_DISABLED

// Interleaved Gradient Noise
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float quick_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

float sample_directional_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord, float taa_frame_count) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_directional_soft_shadow_samples() == 0) {
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_directional_soft_shadow_samples(); i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.directional_soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(sc_directional_soft_shadow_samples()));
}

float sample_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec3 coord, float taa_frame_count) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples() == 0) {
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_soft_shadow_samples(); i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(sc_soft_shadow_samples()));
}

float sample_omni_pcf_shadow(texture2D shadow, float blur_scale, vec2 coord, vec4 uv_rect, vec2 flip_offset, float depth, float taa_frame_count) {
	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples() == 0) {
		vec2 pos = coord * 0.5 + 0.5;
		pos = uv_rect.xy + pos * uv_rect.zw;
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;
	vec2 offset_scale = blur_scale * 2.0 * scene_data_block.data.shadow_atlas_pixel_size / uv_rect.zw;

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_soft_shadow_samples(); i++) {
		vec2 offset = offset_scale * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy);
		vec2 sample_coord = coord + offset;

		float sample_coord_length_squared = dot(sample_coord, sample_coord);
		bool do_flip = sample_coord_length_squared > 1.0;

		if (do_flip) {
			float len = sqrt_IEEE_int_approximation(sample_coord_length_squared);
			sample_coord = sample_coord * (2.0 / len - 1.0);
		}

		sample_coord = sample_coord * 0.5 + 0.5;
		sample_coord = uv_rect.xy + sample_coord * uv_rect.zw;

		if (do_flip) {
			sample_coord += flip_offset;
		}
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(sample_coord, depth, 1.0));
	}

	return avg * (1.0 / float(sc_soft_shadow_samples()));
}

float sample_directional_soft_shadow(texture2D shadow, vec3 pssm_coord, vec2 tex_scale, float taa_frame_count) {
	//find blocker
	float blocker_count = 0.0;
	float blocker_average = 0.0;

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_directional_penumbra_shadow_samples(); i++) {
		vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
		float d = textureLod(sampler2D(shadow, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
		if (d > pssm_coord.z) {
			blocker_average += d;
			blocker_count += 1.0;
		}
	}

	if (blocker_count > 0.0) {
		//blockers found, do soft shadow
		blocker_average /= blocker_count;
		float penumbra = (-pssm_coord.z + blocker_average) / (1.0 - blocker_average);
		tex_scale *= penumbra;

		float s = 0.0;

		SPEC_CONSTANT_LOOP_ANNOTATION
		for (uint i = 0; i < sc_directional_penumbra_shadow_samples(); i++) {
			vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
			s += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(suv, pssm_coord.z, 1.0));
		}

		return s / float(sc_directional_penumbra_shadow_samples());

	} else {
		//no blockers found, so no shadow
		return 1.0;
	}
}

#endif // SHADOWS_DISABLED

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float taa_frame_count, vec3 albedo, inout float alpha, vec2 screen_uv,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_SHEEN_USED
		float sheen, float sheen_roughness, vec3 sheen_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 tangent, vec3 binormal, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {
	const float EPSILON = 1e-6f;

	// Omni light attenuation.
	vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float omni_attenuation = get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation);

	// Compute size.
	float size = 0.0;
	if (sc_use_light_soft_shadows() && omni_lights.data[idx].size > 0.0) {
		float t = omni_lights.data[idx].size / max(0.001, light_length);
		size = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

	float shadow = 1.0;
#ifndef SHADOWS_DISABLED
	// Omni light shadow.
	if (omni_attenuation > EPSILON && omni_lights.data[idx].shadow_opacity > 0.001) {
		// there is a shadowmap
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 base_uv_rect = omni_lights.data[idx].atlas_rect;
		base_uv_rect.xy += texel_size;
		base_uv_rect.zw -= texel_size * 2.0;

		// Omni lights use direction.xy to store to store the offset between the two paraboloid regions
		vec2 flip_offset = omni_lights.data[idx].direction.xy;

		vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_dir = normalize(local_vert);

		vec3 local_normal = normalize(mat3(omni_lights.data[idx].shadow_matrix) * normal);
		vec3 normal_bias = local_normal * omni_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

		if (sc_use_light_soft_shadows() && omni_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			vec3 basis_normal = shadow_dir;
			vec3 v0 = abs(basis_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, basis_normal));
			vec3 bitangent = normalize(cross(tangent, basis_normal));
			float z_norm = 1.0 - shadow_len * omni_lights.data[idx].inv_radius;

			tangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;
			bitangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;

			SPEC_CONSTANT_LOOP_ANNOTATION
			for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
				vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;

				vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);

				vec4 uv_rect = base_uv_rect;

				if (pos.z >= 0.0) {
					uv_rect.xy += flip_offset;
				}

				pos.z = 1.0 + abs(pos.z);
				pos.xy /= pos.z;

				pos.xy = pos.xy * 0.5 + 0.5;
				pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;

				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos.xy, 0.0).r;
				if (d > z_norm) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				tangent *= penumbra;
				bitangent *= penumbra;

				z_norm += omni_lights.data[idx].inv_radius * omni_lights.data[idx].shadow_bias;

				shadow = 0.0;

				SPEC_CONSTANT_LOOP_ANNOTATION
				for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
					vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;
					vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					pos = normalize(pos + normal_bias);

					vec4 uv_rect = base_uv_rect;

					if (pos.z >= 0.0) {
						uv_rect.xy += flip_offset;
					}

					pos.z = 1.0 + abs(pos.z);
					pos.xy /= pos.z;

					pos.xy = pos.xy * 0.5 + 0.5;
					pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(pos.xy, z_norm, 1.0));
				}

				shadow /= float(sc_penumbra_shadow_samples());
				shadow = mix(1.0, shadow, omni_lights.data[idx].shadow_opacity);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
			vec4 uv_rect = base_uv_rect;

			vec3 shadow_sample = normalize(shadow_dir + normal_bias);
			if (shadow_sample.z >= 0.0) {
				uv_rect.xy += flip_offset;
				flip_offset *= -1.0;
			}

			shadow_sample.z = 1.0 + abs(shadow_sample.z);
			vec2 pos = shadow_sample.xy / shadow_sample.z;
			float depth = shadow_len - omni_lights.data[idx].shadow_bias;
			depth *= omni_lights.data[idx].inv_radius;
			depth = 1.0 - depth;
			shadow = mix(1.0, sample_omni_pcf_shadow(shadow_atlas, omni_lights.data[idx].soft_shadow_scale / shadow_sample.z, pos, uv_rect, flip_offset, depth, taa_frame_count), omni_lights.data[idx].shadow_opacity);
		}
	}
#endif

	vec3 color = omni_lights.data[idx].color;

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth; //no transmittance by default
	transmittance_color.a *= omni_attenuation;
#ifndef SHADOWS_DISABLED
	if (omni_lights.data[idx].shadow_opacity > 0.001) {
		// Redo shadowmapping, but shrink the model a bit to avoid artifacts.
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 uv_rect = omni_lights.data[idx].atlas_rect;
		uv_rect.xy += texel_size;
		uv_rect.zw -= texel_size * 2.0;

		// Omni lights use direction.xy to store to store the offset between the two paraboloid regions
		vec2 flip_offset = omni_lights.data[idx].direction.xy;

		vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal) * omni_lights.data[idx].transmittance_bias, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_sample = normalize(local_vert);

		if (shadow_sample.z >= 0.0) {
			uv_rect.xy += flip_offset;
			flip_offset *= -1.0;
		}

		shadow_sample.z = 1.0 + abs(shadow_sample.z);
		vec2 pos = shadow_sample.xy / shadow_sample.z;
		float depth = shadow_len * omni_lights.data[idx].inv_radius;
		depth = 1.0 - depth;

		pos = pos * 0.5 + 0.5;
		pos = uv_rect.xy + pos * uv_rect.zw;
		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos, 0.0).r;
		transmittance_z = (depth - shadow_z) / omni_lights.data[idx].inv_radius;
	}
#endif // !SHADOWS_DISABLED
#endif // LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector() && omni_lights.data[idx].projector_rect != vec4(0.0)) {
		vec3 local_v = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
		local_v = normalize(local_v);

		vec4 atlas_rect = omni_lights.data[idx].projector_rect;

		if (local_v.z >= 0.0) {
			atlas_rect.y += atlas_rect.w;
		}

		local_v.z = 1.0 + abs(local_v.z);

		local_v.xy /= local_v.z;
		local_v.xy = local_v.xy * 0.5 + 0.5;
		vec2 proj_uv = local_v.xy * atlas_rect.zw;

		if (sc_projector_use_mipmaps()) {
			vec2 proj_uv_ddx;
			vec2 proj_uv_ddy;
			{
				vec3 local_v_ddx = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0)).xyz;
				local_v_ddx = normalize(local_v_ddx);

				if (local_v_ddx.z >= 0.0) {
					local_v_ddx.z += 1.0;
				} else {
					local_v_ddx.z = 1.0 - local_v_ddx.z;
				}

				local_v_ddx.xy /= local_v_ddx.z;
				local_v_ddx.xy = local_v_ddx.xy * 0.5 + 0.5;

				proj_uv_ddx = local_v_ddx.xy * atlas_rect.zw - proj_uv;

				vec3 local_v_ddy = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0)).xyz;
				local_v_ddy = normalize(local_v_ddy);

				if (local_v_ddy.z >= 0.0) {
					local_v_ddy.z += 1.0;
				} else {
					local_v_ddy.z = 1.0 - local_v_ddy.z;
				}

				local_v_ddy.xy /= local_v_ddy.z;
				local_v_ddy.xy = local_v_ddy.xy * 0.5 + 0.5;

				proj_uv_ddy = local_v_ddy.xy * atlas_rect.zw - proj_uv;
			}

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	vec3 light_rel_vec_norm = light_rel_vec / light_length;
	light_compute(normal, light_rel_vec_norm, eye_vec, size, color, false, omni_attenuation * shadow, f0, orms, omni_lights.data[idx].specular_amount, albedo, alpha, screen_uv,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * omni_attenuation, rim_tint,
#endif
#ifdef LIGHT_SHEEN_USED
			sheen, sheen_roughness, sheen_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			tangent, binormal, anisotropy,
#endif
			diffuse_light,
			specular_light);
}

vec2 normal_to_panorama(vec3 n) {
	n = normalize(n);
	vec2 panorama_coords = vec2(atan(n.x, n.z), acos(-n.y));

	if (panorama_coords.x < 0.0) {
		panorama_coords.x += M_PI * 2.0;
	}

	panorama_coords /= vec2(M_PI * 2.0, M_PI);
	return panorama_coords;
}

void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float taa_frame_count, vec3 albedo, inout float alpha, vec2 screen_uv,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_SHEEN_USED
		float sheen, float sheen_roughness, vec3 sheen_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 tangent, vec3 binormal, float anisotropy,
#endif
		inout vec3 diffuse_light,
		inout vec3 specular_light) {
	const float EPSILON = 1e-6f;

	// Spot light attenuation.
	vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	vec3 light_rel_vec_norm = light_rel_vec / light_length;
	float spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	vec3 spot_dir = spot_lights.data[idx].direction;

	// This conversion to a highp float is crucial to prevent light leaking
	// due to precision errors in the following calculations (cone angle is mediump).
	highp float cone_angle = spot_lights.data[idx].cone_angle;
	float scos = max(dot(-light_rel_vec_norm, spot_dir), cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cone_angle));
	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation);

	// Compute size.
	float size = 0.0;
	if (sc_use_light_soft_shadows() && spot_lights.data[idx].size > 0.0) {
		float t = spot_lights.data[idx].size / max(0.001, light_length);
		size = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

	float shadow = 1.0;
#ifndef SHADOWS_DISABLED
	// Spot light shadow.
	if (spot_attenuation > EPSILON && spot_lights.data[idx].shadow_opacity > 0.001) {
		vec3 normal_bias = normal * light_length * spot_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(normal, light_rel_vec_norm)));

		//there is a shadowmap
		vec4 v = vec4(vertex + normal_bias, 1.0);

		vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
		splane.z += spot_lights.data[idx].shadow_bias / (light_length * spot_lights.data[idx].inv_radius);
		splane /= splane.w;

		if (sc_use_light_soft_shadows() && spot_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker
			float z_norm = dot(spot_dir, -light_rel_vec) * spot_lights.data[idx].inv_radius;

			vec2 shadow_uv = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			float uv_size = spot_lights.data[idx].soft_shadow_size * z_norm * spot_lights.data[idx].soft_shadow_scale;
			vec2 clamp_max = spot_lights.data[idx].atlas_rect.xy + spot_lights.data[idx].atlas_rect.zw;

			SPEC_CONSTANT_LOOP_ANNOTATION
			for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
				vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
				suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
				if (d > splane.z) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				uv_size *= penumbra;

				shadow = 0.0;

				SPEC_CONSTANT_LOOP_ANNOTATION
				for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
					vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
					suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(suv, splane.z, 1.0));
				}

				shadow /= float(sc_penumbra_shadow_samples());
				shadow = mix(1.0, shadow, spot_lights.data[idx].shadow_opacity);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
			//hard shadow
			vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
			shadow = mix(1.0, sample_pcf_shadow(shadow_atlas, spot_lights.data[idx].soft_shadow_scale * scene_data_block.data.shadow_atlas_pixel_size, shadow_uv, taa_frame_count), spot_lights.data[idx].shadow_opacity);
		}
	}
#endif // SHADOWS_DISABLED

	vec3 color = spot_lights.data[idx].color;
	float specular_amount = spot_lights.data[idx].specular_amount;

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth;
	transmittance_color.a *= spot_attenuation;
#ifndef SHADOWS_DISABLED
	if (spot_lights.data[idx].shadow_opacity > 0.001) {
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal) * spot_lights.data[idx].transmittance_bias, 1.0));
		splane /= splane.w;

		vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), shadow_uv.xy, 0.0).r;

		shadow_z = shadow_z * 2.0 - 1.0;
		float z_far = 1.0 / spot_lights.data[idx].inv_radius;
		float z_near = 0.01;
		shadow_z = 2.0 * z_near * z_far / (z_far + z_near - shadow_z * (z_far - z_near));

		//distance to light plane
		float z = dot(spot_dir, -light_rel_vec);
		transmittance_z = z - shadow_z;
	}
#endif // !SHADOWS_DISABLED
#endif // LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector() && spot_lights.data[idx].projector_rect != vec4(0.0)) {
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex, 1.0));
		splane /= splane.w;

		vec2 proj_uv = splane.xy * spot_lights.data[idx].projector_rect.zw;

		if (sc_projector_use_mipmaps()) {
			//ensure we have proper mipmaps
			vec4 splane_ddx = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0));
			splane_ddx /= splane_ddx.w;
			vec2 proj_uv_ddx = splane_ddx.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 splane_ddy = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0));
			splane_ddy /= splane_ddy.w;
			vec2 proj_uv_ddy = splane_ddy.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	light_compute(normal, light_rel_vec_norm, eye_vec, size, color, false, spot_attenuation * shadow, f0, orms, spot_lights.data[idx].specular_amount, albedo, alpha, screen_uv,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * spot_attenuation, rim_tint,
#endif
#ifdef LIGHT_SHEEN_USED
			sheen, sheen_roughness, sheen_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			tangent, binormal, anisotropy,
#endif
			diffuse_light, specular_light);
}

void reflection_process(uint ref_index, vec3 vertex, vec3 ref_vec, vec3 normal, float roughness, vec3 ambient_light, vec3 specular_light,
#ifdef LIGHT_CLEARCOAT_USED
		vec3 cc_specular_light, vec3 cc_ref_vec, float cc_perceptual_roughness, inout vec3 cc_reflection_accum,
#endif
#ifdef LIGHT_SHEEN_USED
		vec3 sh_specular_light, vec3 sh_ref_vec, float sheen_perceptual_roughness, inout vec3 sh_reflection_accum,
#endif
		inout vec4 ambient_accum, inout vec4 reflection_accum) {
	vec3 box_extents = reflections.data[ref_index].box_extents;
	vec3 local_pos = (reflections.data[ref_index].local_matrix * vec4(vertex, 1.0)).xyz;

	if (any(greaterThan(abs(local_pos), box_extents))) { //out of the reflection box
		return;
	}

	float blend = 1.0;
	if (reflections.data[ref_index].blend_distance != 0.0) {
		vec3 axis_blend_distance = min(vec3(reflections.data[ref_index].blend_distance), box_extents);
		vec3 blend_axes = abs(local_pos) - box_extents + axis_blend_distance;
		blend_axes /= axis_blend_distance;
		blend_axes = clamp(1.0 - blend_axes, vec3(0.0), vec3(1.0));

		blend = pow(blend_axes.x * blend_axes.y * blend_axes.z, 2.0);
	}

	if (reflections.data[ref_index].intensity > 0.0 && reflection_accum.a < 1.0) { // compute reflection

		vec3 local_ref_vec = (reflections.data[ref_index].local_matrix * vec4(ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { //box project

			vec3 nrdir = normalize(local_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		vec4 reflection;
		float reflection_blend = max(0.0, blend - reflection_accum.a);

		reflection.rgb = textureLod(samplerCubeArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_ref_vec, reflections.data[ref_index].index), sqrt(roughness) * MAX_ROUGHNESS_LOD).rgb * sc_luminance_multiplier();
		reflection.rgb *= reflections.data[ref_index].exposure_normalization;
		reflection.a = reflection_blend;

		reflection.rgb *= reflections.data[ref_index].intensity;
		reflection.rgb *= reflection.a;

		reflection_accum += reflection;

#ifdef LIGHT_CLEARCOAT_USED
		vec3 local_cc_ref_vec = (reflections.data[ref_index].local_matrix * vec4(cc_ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { // Box project.

			vec3 nrdir = normalize(local_cc_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_cc_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		vec3 cc_reflection = textureLod(samplerCubeArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_cc_ref_vec, reflections.data[ref_index].index), cc_perceptual_roughness * MAX_ROUGHNESS_LOD).rgb * sc_luminance_multiplier();
		cc_reflection *= reflections.data[ref_index].exposure_normalization;
		if (reflections.data[ref_index].exterior) {
			cc_reflection = mix(cc_specular_light, cc_reflection, blend);
		}
		cc_reflection *= reflections.data[ref_index].intensity * blend;
		cc_reflection_accum += cc_reflection;
#endif // LIGHT_CLEARCOAT_USED

#ifdef LIGHT_SHEEN_USED
		vec3 local_sh_ref_vec = (reflections.data[ref_index].local_matrix * vec4(sh_ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { // Box project.

			vec3 nrdir = normalize(local_sh_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_sh_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		vec3 sh_reflection = textureLod(samplerCubeArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_sh_ref_vec, reflections.data[ref_index].index), sheen_perceptual_roughness * MAX_ROUGHNESS_LOD).rgb * sc_luminance_multiplier();
		sh_reflection *= reflections.data[ref_index].exposure_normalization;
		if (reflections.data[ref_index].exterior) {
			sh_reflection = mix(sh_specular_light, sh_reflection, blend);
		}
		sh_reflection *= reflections.data[ref_index].intensity * blend;
		sh_reflection_accum += sh_reflection;
#endif // LIGHT_SHEEN_USED
	}

	if (ambient_accum.a >= 1.0) {
		return;
	}

	switch (reflections.data[ref_index].ambient_mode) {
		case REFLECTION_AMBIENT_DISABLED: {
			//do nothing
		} break;
		case REFLECTION_AMBIENT_ENVIRONMENT: {
			vec3 local_amb_vec = (reflections.data[ref_index].local_matrix * vec4(normal, 0.0)).xyz;
			vec4 ambient_out;
			float ambient_blend = max(0.0, blend - ambient_accum.a);

			ambient_out.rgb = textureLod(samplerCubeArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_amb_vec, reflections.data[ref_index].index), MAX_ROUGHNESS_LOD).rgb;
			ambient_out.rgb *= reflections.data[ref_index].exposure_normalization;
			ambient_out.a = ambient_blend;
			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
		case REFLECTION_AMBIENT_COLOR: {
			vec4 ambient_out;
			float ambient_blend = max(0.0, blend - ambient_accum.a);

			ambient_out.rgb = reflections.data[ref_index].ambient;
			ambient_out.a = ambient_blend;
			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
	}
}

float blur_shadow(float shadow) {
	return shadow;
#if 0
	//disabling for now, will investigate later
	float interp_shadow = shadow;
	if (gl_HelperInvocation) {
		interp_shadow = -4.0; // technically anything below -4 will do but just to make sure
	}

	uvec2 fc2 = uvec2(gl_FragCoord.xy);
	interp_shadow -= dFdx(interp_shadow) * (float(fc2.x & 1) - 0.5);
	interp_shadow -= dFdy(interp_shadow) * (float(fc2.y & 1) - 0.5);

	if (interp_shadow >= 0.0) {
		shadow = interp_shadow;
	}
	return shadow;
#endif
}

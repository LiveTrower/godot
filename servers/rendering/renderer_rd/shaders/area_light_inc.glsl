float integrate_edge_hill(vec3 p0, vec3 p1) {
	// Approximation suggested by Hill and Heitz, calculating the integral of the spherical cosine distribution over the line between p0 and p1.
	// Runs faster than the exact formula of Baum et al. (1989).
	float cosTheta = dot(p0, p1);

	float x = cosTheta;
	float y = abs(x);
	float a = 5.42031 + (3.12829 + 0.0902326 * y) * y;
	float b = 3.45068 + (4.18814 + y) * y;
	float theta_sintheta = a / b;

	if (x < 0.0) {
		theta_sintheta = M_PI * inversesqrt(1.0 - x * x) - theta_sintheta;
	}
	return theta_sintheta * cross(p0, p1).y;
}

float integrate_edge(vec3 p_proj0, vec3 p_proj1, vec3 p0, vec3 p1) {
	float epsilon = 0.00001;
	bool opposite_sides = dot(p_proj0, p_proj1) < -1.0 + epsilon;
	if (opposite_sides) {
		// calculate the point on the line p0 to p1 that is closest to the vertex (origin)
		vec3 half_point_t = p0 + normalize(p1 - p0) * dot(p0, normalize(p0 - p1));
		vec3 half_point = normalize(half_point_t);
		return integrate_edge_hill(p_proj0, half_point) + integrate_edge_hill(half_point, p_proj1);
	}
	return integrate_edge_hill(p_proj0, p_proj1);
}

void clip_quad_to_horizon(inout vec3 L[5], out int vertex_count) {
	// detect clipping config
	int config = 0;
	if (L[0].y > 0.0) {
		config += 1;
	}
	if (L[1].y > 0.0) {
		config += 2;
	}
	if (L[2].y > 0.0) {
		config += 4;
	}
	if (L[3].y > 0.0) {
		config += 8;
	}

	// clip
	vertex_count = 0;

	if (config == 0) {
		// clip all
	} else if (config == 1) // V1 clip V2 V3 V4
	{
		vertex_count = 3;
		L[1] = -L[1].y * L[0] + L[0].y * L[1];
		L[2] = -L[3].y * L[0] + L[0].y * L[3];
	} else if (config == 2) // V2 clip V1 V3 V4
	{
		vertex_count = 3;
		L[0] = -L[0].y * L[1] + L[1].y * L[0];
		L[2] = -L[2].y * L[1] + L[1].y * L[2];
	} else if (config == 3) // V1 V2 clip V3 V4
	{
		vertex_count = 4;
		L[2] = -L[2].y * L[1] + L[1].y * L[2];
		L[3] = -L[3].y * L[0] + L[0].y * L[3];
	} else if (config == 4) // V3 clip V1 V2 V4
	{
		vertex_count = 3;
		L[0] = -L[3].y * L[2] + L[2].y * L[3];
		L[1] = -L[1].y * L[2] + L[2].y * L[1];
	} else if (config == 5) // V1 V3 clip V2 V4) impossible
	{
		vertex_count = 0;
	} else if (config == 6) // V2 V3 clip V1 V4
	{
		vertex_count = 4;
		L[0] = -L[0].y * L[1] + L[1].y * L[0];
		L[3] = -L[3].y * L[2] + L[2].y * L[3];
	} else if (config == 7) // V1 V2 V3 clip V4
	{
		vertex_count = 5;
		L[4] = -L[3].y * L[0] + L[0].y * L[3];
		L[3] = -L[3].y * L[2] + L[2].y * L[3];
	} else if (config == 8) // V4 clip V1 V2 V3
	{
		vertex_count = 3;
		L[0] = -L[0].y * L[3] + L[3].y * L[0];
		L[1] = -L[2].y * L[3] + L[3].y * L[2];
		L[2] = L[3];
	} else if (config == 9) // V1 V4 clip V2 V3
	{
		vertex_count = 4;
		L[1] = -L[1].y * L[0] + L[0].y * L[1];
		L[2] = -L[2].y * L[3] + L[3].y * L[2];
	} else if (config == 10) // V2 V4 clip V1 V3) impossible
	{
		vertex_count = 0;
	} else if (config == 11) // V1 V2 V4 clip V3
	{
		vertex_count = 5;
		L[4] = L[3];
		L[3] = -L[2].y * L[3] + L[3].y * L[2];
		L[2] = -L[2].y * L[1] + L[1].y * L[2];
	} else if (config == 12) // V3 V4 clip V1 V2
	{
		vertex_count = 4;
		L[1] = -L[1].y * L[2] + L[2].y * L[1];
		L[0] = -L[0].y * L[3] + L[3].y * L[0];
	} else if (config == 13) // V1 V3 V4 clip V2
	{
		vertex_count = 5;
		L[4] = L[3];
		L[3] = L[2];
		L[2] = -L[1].y * L[2] + L[2].y * L[1];
		L[1] = -L[1].y * L[0] + L[0].y * L[1];
	} else if (config == 14) // V2 V3 V4 clip V1
	{
		vertex_count = 5;
		L[4] = -L[0].y * L[3] + L[3].y * L[0];
		L[0] = -L[0].y * L[1] + L[1].y * L[0];
	} else if (config == 15) // V1 V2 V3 V4
	{
		vertex_count = 4;
	}

	if (vertex_count == 3) {
		L[3] = L[0];
	}
	if (vertex_count == 4) {
		L[4] = L[0];
	}
}

vec3 wrapped_normal(vec3 N, vec3 L, float w) {
	float cosTheta = dot(N, L);
	float wrappedCosTheta = clamp((cosTheta + w) / (1.0 + w), 0.0, 1.0);
	float sinMaximumAngleChange = w;
	float sinMinimumAngleChange = 0.0;
	float sinPhi = mix(sinMaximumAngleChange, sinMinimumAngleChange, wrappedCosTheta);
	float cosPhi = sqrt(1.0 - sinPhi * sinPhi);
	return normalize(cosPhi * N + sinPhi * cross(cross(N, L), N));
}

vec3 ltc_evaluate(vec3 normal, vec3 eye_vec, mat3 M_inv, vec3 points[4]) {
	// construct the orthonormal basis around the normal vector
	vec3 x, z;
	z = -normalize(eye_vec - normal * dot(eye_vec, normal)); // expanding the angle between view and normal vector to 90 degrees, this gives a normal vector
	x = cross(normal, z);

	// rotate area light in (T1, normal, T2) basis
	M_inv = M_inv * transpose(mat3(x, normal, z));

	vec3 L[5];
	L[0] = M_inv * points[0];
	L[1] = M_inv * points[1];
	L[2] = M_inv * points[2];
	L[3] = M_inv * points[3];

	int n = 0;
	clip_quad_to_horizon(L, n);
	if (n == 0) {
		return vec3(0, 0, 0);
	}

	vec3 L_proj[5];
	// project onto unit sphere
	L_proj[0] = normalize(L[0]);
	L_proj[1] = normalize(L[1]);
	L_proj[2] = normalize(L[2]);
	L_proj[3] = normalize(L[3]);
	L_proj[4] = normalize(L[4]);

	// Prevent abnormal values when the light goes through (or close to) the fragment
	vec3 pnorm = normalize(cross(L_proj[0] - L_proj[1], L_proj[2] - L_proj[1]));
	if (abs(dot(pnorm, L_proj[0])) < 1e-10) {
		// we could just return black, but that would lead to some black pixels in front of the light.
		// Better, we check if the fragment is on the light, and return white if so.
		vec3 r10 = points[0] - points[1];
		vec3 r12 = points[2] - points[1];
		float alpha = -dot(points[1], r10) / dot(r10, r10);
		float beta = -dot(points[1], r12) / dot(r12, r12);
		if (0.0 < alpha && alpha < 1.0 && 0.0 < beta && beta < 1.0) { // fragment is on light {
			return vec3(2 * M_PI);
		} else {
			return vec3(0.0);
		}
	}

	float I;
	I = integrate_edge(L_proj[0], L_proj[1], L[0], L[1]);
	I += integrate_edge(L_proj[1], L_proj[2], L[1], L[2]);
	I += integrate_edge(L_proj[2], L_proj[3], L[2], L[3]);
	if (n >= 4) {
		I += integrate_edge(L_proj[3], L_proj[4], L[3], L[4]);
	}
	if (n == 5) {
		I += integrate_edge(L_proj[4], L_proj[0], L[4], L[0]);
	}

	return vec3(abs(I));
}

/**
 * Calculates the azimuthal angle.
 */
float phi(vec3 v) {
	float p = atan(v.y, v.x);

	if (p < 0) {
		p += 2 * M_PI;
	}

	return p;
}

/**
 * Rotates the vector v around the axis given a certain angle.
 */
vec3 rotateVector(vec3 v, vec3 axis, float angle) {
	float s = sin(angle);
	float c = cos(angle);

	return v * c + axis * dot(v, axis) * (1.f - c) + s * cross(axis, v);
}

/**
 * Fetch the LTC coefficients in a 32x32 lookup table.
 */
vec3 fetchCoeffs(float cosThetaO, float sheen_roughness) {
	// Compute table indices and interpolation factors.
	//return texture(ltc_sheen_lut, vec2(sqrt(sheen_roughness), cosThetaO)).xyz;
	return vec3(0.0);
}

/**
 * Evaluate the LTC distribution in its default coordinate system.
 */
float ltc_eval_sheen(vec3 wi, vec3 ltcCoeffs, vec3 N) {
	float aInv = ltcCoeffs[0];
	float bInv = ltcCoeffs[1];

	vec3 wiOrg = vec3(aInv * wi.x + bInv * wi.z, aInv * wi.y, wi.z);

	float len = length(wiOrg);

	float det = aInv * aInv;
	float jacobian = det / (len * len * len);

	float cosThetaIOrg = clamp(dot(N, wiOrg), 0.0f, 1.0f);

	return cosThetaIOrg / M_PI * jacobian;
}

/**
 * The sheen layer we are going to use.
 */
vec3 sheenModel(vec3 vertex, vec3 view, vec3 normal, vec3 points[4], float sheen, float sheen_roughness) {
	// Get the light position by getting its center.
	vec3 lightPosition = points[0] +
			points[1] +
			points[2] +
			points[3];
	lightPosition /= 4.0;

	// Calculate the view direction and the light direction.
	vec3 wo = normalize(view);
	vec3 wi = normalize(lightPosition);

	vec3 N = normalize(normal);

	// Calculate its cosTheta values.
	float cosThetaO = clamp(abs(dot(N, wo)) + 1e-5, 0.0, 1.0);

	// Rotate coordinate frame to align with incident direction wo.
	float phiStd = phi(wo);
	vec3 wiStd = rotateVector(wi, vec3(0.0, 0.0, 1.0), -phiStd);

	// Evaluate LTC distribution in aligned coordinates.
	vec3 ltcCoeffs = fetchCoeffs(cosThetaO, sheen_roughness);
	float value = ltc_eval_sheen(wiStd, ltcCoeffs, N);

	// Consider the overall reflectance `R` and the artist-specified sheen scale.
	float R = ltcCoeffs[2];
	value *= R * sheen;

	float res = value; // cosThetaI;
	res = clamp(res, 0.0, 1.0);

	return vec3(res);
}

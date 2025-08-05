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

vec4 sample_area_light_cookie(vec3 L[5], vec3 F, float perceptualRoughness) {
    // L[0..3] : LL UL UR LR

    vec3  origin = L[0];
    vec3  right = L[3] - origin;
    vec3  up = L[1] - origin;

    vec3  normal = cross(right, up);
    float   sqArea = dot(normal, normal);
    normal *= inversesqrt(sqArea);

    // Compute intersection of irradiance vector with the area light plane
    float   hitDistance = dot(origin, normal) / dot(F, normal);
    vec3  hitPosition = hitDistance * normal;
    hitPosition -= origin;  // Relative to bottom-left corner

    // Here, right and up vectors are not necessarily orthonormal
    // We create the orthogonal vector "ortho" by projecting "up" onto the vector orthogonal to "right"
    //  ortho = up - (up.right') * right'
    // Where right' = right / sqrt( dot( right, right ) ), the normalized right vector
    float   recSqLengthRight = 1.0 / dot(right, right);
    float   upRightMixing = dot(up, right);
    vec3  ortho = up - upRightMixing * right * recSqLengthRight;

    // The V coordinate along the "up" vector is simply the projection against the ortho vector
    float   v = dot(hitPosition, ortho) / dot(ortho, ortho);

    // The U coordinate is not only the projection against the right vector
    //  but also the subtraction of the influence of the up vector upon the right vector
    //  (indeed, if the up & right vectors are not orthogonal then a certain amount of
    //  the up coordinate also influences the right coordinate)
    //
    //       |    up
    // ortho ^....*--------*
    //       |   /:       /
    //       |  / :      /
    //       | /  :     /
    //       |/   :    /
    //       +----+-->*----->
    //            : right
    //          mix of up into right that needs to be subtracted from simple projection on right vector
    //
    float   u = (dot(hitPosition, right) - upRightMixing * v) * recSqLengthRight;
    // We create automatic quad emissive mesh for area light. For those to be displayed in the direction
    // of the light when they are single sided, we need to reverse the winding order.
    // Because of this reverse of winding order, to get a matching area light reflection,
    // we need to flip the x axis.
    vec2  hitUV = vec2(u, v);

    // Assuming the original cosine lobe distribution Do is enclosed in a cone of 90 deg  aperture,
    //  following the idea of orthogonal projection upon the area light's plane we find the intersection
    //  of the cone to be a disk of area PI*d^2 where d is the hit distance we computed above.
    // We also know the area of the transformed polygon A = sqrt( sqArea ) and we pose the ratio of covered area as PI.d^2 / A.
    //
    // Knowing the area in square texels of the cookie texture A_sqTexels = texture width * texture height (default is 128x128 square texels)
    //  we can deduce the actual area covered by the cone in square texels as:
    //  A_covered = Pi.d^2 / A * A_sqTexels
    //
    // From this, we find the mip level as: mip = log2( sqrt( A_covered ) ) = log2( A_covered ) / 2
    // Also, assuming that A_sqTexels is of the form 2^n * 2^n we get the simplified expression: mip = log2( Pi.d^2 / A ) / 2 + n
    //
    // Compute the cookie mip count using the cookie size in the atlas
	float   cookieWidth = textureSize(sampler2D(decal_atlas_srgb, light_projector_sampler), 0).x; // cookies and atlas are guaranteed to be POT
	float   cookieMipCount = round(log2(cookieWidth));
    float   mipLevel = 0.5 * log2(1e-8 + M_PI * hitDistance*hitDistance * inversesqrt(sqArea)) + cookieMipCount;

    // We want to prevent the texture from accessing to the lower mips when evaluating the specular lobe
    // when operating on low roughness points. We progressively give access from mip 3 the rest of the mips between the range 0.0 -> 0.3
    // in the perceptual roughness space
    float mipTrimming = saturate((0.3 - perceptualRoughness) / 0.3);
    mipLevel = clamp(mipLevel, 0, mix(cookieMipCount, 3.0, mipTrimming));

    return textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), saturate(hitUV), mipLevel);
}

vec3 ltc_evaluate(vec3 vertex, vec3 normal, vec3 eye_vec, mat3 M_inv, vec3 points[4], out vec3 L1[5]) {
	// construct the orthonormal basis around the normal vector
	vec3 x, z;
	z = -normalize(eye_vec - normal * dot(eye_vec, normal)); // expanding the angle between view and normal vector to 90 degrees, this gives a normal vector, unless view=normal. TODO: in that case, we have a problem.
	x = cross(normal, z);

	// rotate area light in (T1, normal, T2) basis
	M_inv = M_inv * transpose(mat3(x, normal, z));

	vec3 L[5];
	L[0] = M_inv * points[0];
	L[1] = M_inv * points[1];
	L[2] = M_inv * points[2];
	L[3] = M_inv * points[3];
	L1 = L;

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

////////////////////////////////// Sheen shading //////////////////////////////////
float phi(vec3 v) {
    float p = atan(v.y, v.x);
    
    if (p < 0) {
        p += 2.0 * M_PI;
    }
    
    return p; 
}

vec3 rotate_vector(vec3 v, vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    
    return v * c + axis * dot(v, axis) * (1.f - c) + s * cross(axis, v);
}

vec3 fetch_coeffs(float alpha, float cosThetaO) {
    float row = max(0.0, min(alpha, 1.0));
    float col = max(0.0, min(cosThetaO, 1.0));

    return texture(ltc_sheen, vec2(col, row)).rgb;
}

float ltc_evaluate_sheen(vec3 wi, vec3 ltcCoeffs, vec3 N) {
    float aInv = ltcCoeffs[0];
    float bInv = ltcCoeffs[1];
    
    vec3 wiOrg = vec3(aInv * wi.x + bInv * wi.z, aInv * wi.y, wi.z);
    
    float len = length(wiOrg);

    float det = aInv * aInv;
    float jacobian = det / (len * len * len);
    
    float cosThetaIOrg = clamp(dot(N, wiOrg), 0.0f, 1.0f);

    return cosThetaIOrg / M_PI * jacobian;
}
/////////////////////////////////////////////////////////////////////////////////

////////////////////////////////// Line Shape //////////////////////////////////
void build_orthonormal_basis(in vec3 n, out vec3 b1, out vec3 b2) {
    if (n.z < -0.9999999) {
        b1 = vec3( 0.0, -1.0, 0.0);
        b2 = vec3(-1.0,  0.0, 0.0);
        return;
    }
    float a = 1.0 / (1.0 + n.z);
    float b = -n.x*n.y*a;
    b1 = vec3(1.0 - n.x*n.x*a, b, -n.x);
    b2 = vec3(b, 1.0 - n.y*n.y*a, -n.y);
}

mat3 Minv;
float D(vec3 w) {
    vec3 wo = Minv * w;
    float lo = length(wo);
    float res = 1.0/M_PI * max(0.0, wo.z/lo) * abs(determinant(Minv)) / (lo*lo*lo);
    return res;
}

float Fpo(float d, float l) {
    return l/(d*(d*d + l*l)) + atan(l/d)/(d*d);
}

float Fwt(float d, float l) {
    return l*l/(d*(d*d + l*l));
}

float I_diffuse_line(vec3 p1, vec3 p2) {
    // tangent
    vec3 wt = normalize(p2 - p1);

    // clamping
    if (p1.z <= 0.0 && p2.z <= 0.0) return 0.0;
    if (p1.z < 0.0) p1 = (+p1*p2.z - p2*p1.z) / (+p2.z - p1.z);
    if (p2.z < 0.0) p2 = (-p1*p2.z + p2*p1.z) / (-p2.z + p1.z);

    // parameterization
    float l1 = dot(p1, wt);
    float l2 = dot(p2, wt);

    // shading point orthonormal projection on the line
    vec3 po = p1 - l1*wt;

    // distance to line
    float d = length(po);

    // integral
    float I = (Fpo(d, l2) - Fpo(d, l1)) * po.z +
              (Fwt(d, l2) - Fwt(d, l1)) * wt.z;
    return I / M_PI;
}

float I_ltc_line(vec3 p1, vec3 p2) {
    // transform to diffuse configuration
    vec3 p1o = Minv * p1;
    vec3 p2o = Minv * p2;
    float I_diffuse = I_diffuse_line(p1o, p2o);

    // width factor
    vec3 ortho = normalize(cross(p1, p2));
    float w =  1.0 / length(inverse(transpose(Minv)) * ortho);

    return w * I_diffuse;
}

vec3 ltc_evaluate(vec3 N, vec3 V, vec3 cylinderP1, vec3 cylinderP2, float R) {
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    mat3 B = transpose(mat3(T1, T2, N));

    vec3 p1 = B * cylinderP1;
    vec3 p2 = B * cylinderP2;

    // analytic integration
    float Iline = R * I_ltc_line(p1, p2);
    return vec3(min(1.0, Iline + 0.0));
}
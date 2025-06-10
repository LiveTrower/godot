// Functions related to lighting

#include "math_inc.glsl"

half D_GGX(half NoH, half roughness, hvec3 n, hvec3 h) {
	half a = NoH * roughness;
#ifdef EXPLICIT_FP16
	hvec3 NxH = cross(n, h);
	half k = roughness / (dot(NxH, NxH) + a * a);
#else
	float k = roughness / (1.0 - NoH * NoH + a * a);
#endif
	half d = k * k * half(1.0 / M_PI);
	return saturateHalf(d);
}

half V_GGX(half NdotL, half NdotV, half alpha) {
#ifdef EXPLICIT_FP16
	half v = half(0.5) / mix(half(2.0) * NdotL * NdotV, NdotL + NdotV, alpha);
	return saturateHalf(v);
#else
	// From "Course Notes: Moving Frostbite to PBR", page 12: https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
	float Lambda_GGXV = NdotL * sqrt_IEEE_int_approximation((-NdotV * alpha + NdotV) * NdotV + alpha);
	float Lambda_GGXL = NdotV * sqrt_IEEE_int_approximation((-NdotL * alpha + NdotL) * NdotL + alpha);

	return 0.5 / (Lambda_GGXV + Lambda_GGXL);
#endif
}

half D_GGX_anisotropic(half cos_theta_m, half alpha_x, half alpha_y, half cos_phi, half sin_phi) {
	half alpha2 = alpha_x * alpha_y;
	vec3 v = vec3(alpha_y * cos_phi, alpha_x * sin_phi, alpha2 * cos_theta_m);
	float v2 = dot(v, v);
	half w2 = half(float(alpha2) / v2);
	return alpha2 * w2 * w2 * half(1.0 / M_PI);
}

// From "Course Notes: Moving Frostbite to PBR", page 12: https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
// This is a modified version to take anisotropy into account
half V_GGX_anisotropic(half alpha_x, half alpha_y, half TdotV, half TdotL, half BdotV, half BdotL, half NdotV, half NdotL) {
#ifdef EXPLICIT_FP16
	half Lambda_V = NdotL * length(hvec3(alpha_x * TdotV, alpha_y * BdotV, NdotV));
	half Lambda_L = NdotV * length(hvec3(alpha_x * TdotL, alpha_y * BdotL, NdotL));
	half v = half(0.5) / (Lambda_V + Lambda_L);
	return saturateHalf(v);
#else
	float Lambda_GGXV = NdotL * sqrt_IEEE_int_approximation((-NdotV * (alpha_x * TdotV * TdotV + alpha_y * BdotV * BdotV) + NdotV) * NdotV + alpha_x * TdotV * TdotV + alpha_y * BdotV * BdotV);
	float Lambda_GGXL = NdotV * sqrt_IEEE_int_approximation((-NdotL * (alpha_x * TdotL * TdotL + alpha_y * BdotL * BdotL) + NdotL) * NdotL + alpha_x * TdotL * TdotL + alpha_y * BdotL * BdotL);

	return 0.5 / (Lambda_GGXV + Lambda_GGXL);
#endif
}

hvec3 SchlickFresnel(half f0, half f90, half u) {
	return f0 + (f90 - f0) * pow5(1.0 - u);
}

half SchlickFresnel(half f0, half f90, half u){
	return f0 + (f90 - f0) * pow5(1.0 - u);
}

hvec3 F0(half metallic, half specular, hvec3 albedo) {
	half dielectric = half(0.16) * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(hvec3(dielectric), albedo, hvec3(metallic));
}

half D_Charlie(half roughness, half NoH) {
    // Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
    half invAlpha  = 1.0 / roughness;
    half cos2h = NoH * NoH;
    half sin2h = max(half(1.0) - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
    return (half(2.0) + invAlpha) * pow(sin2h, invAlpha * half(0.5)) / half(2.0 * M_PI);
}

half V_Neubelt(half NoV, half NoL) {
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    return saturateHalf(half(1.0) / (half(4.0) * (NoL + NoV - NoL * NoV)));
}

half V_Kelemen(half LoH) {
    // Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
    return saturateHalf(0.25 / (LoH * LoH));
}

half Diffuse_Lambert(half NoL) {
	return NoL * half(1.0 / M_PI);
}

// Energy conserving lambert wrap shader.
// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
half Diffuse_Lambert_Wrap(half NoL, half roughness){
	half op_roughness = half(1.0) + roughness;
	return max(half(0.0), (NdotL + roughness) / (op_roughness * op_roughness)) * half(1.0 / M_PI);
}

half Diffuse_Burley(half LoH, half NoV, half NoL, half roughness){
	half FD90_minus_1 = half(2.0) * cLdotH * cLdotH * roughness - half(0.5);
	half FdV = half(1.0) + FD90_minus_1 * pow5(NoV);
	half FdL = half(1.0) + FD90_minus_1 * pow5(NoL);
	return half(1.0 / M_PI) * FdV * FdL * cNdotL;
}

// Normalized Disney diffuse function taken from Frostbite's PBR course notes (page 10):
// https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
half Normalized_Diffuse_Burley(half roughness, half NoV, half NoL, half LoH)
{
    half energyBias = mix(half(0.0), half(0.5), roughness);
    half energyFactor = mix(half(1.0), half(1.0/1.51), roughness);
    half fd90 = energyBias + half(2.0) * LoH * LoH * roughness;
	half f0 = half(1.0);
    half lightScatter = SchlickFresnel(f0, fd90, NoL);
    half viewScatter = SchlickFresnel(f0, fd90, NoV);

    return lightScatter * viewScatter * energyFactor * NoL * half(1.0 / M_PI);
}

half Diffuse_Toon(half NoL, half roughness){
	return smoothstep(-roughness, max(roughness, half(0.01)), NdotL) * half(1.0 / M_PI);
}

// [ Chan 2018, "Material Advances in Call of Duty: WWII" ]
half Diffuse_Chan(half roughness, half NoV, half NoL, half LoH, half NoH)
{
	half a2 = roughness * roughness;

	// a2 = 2 / ( 1 + exp2( 18 * g )
	half g = saturate( half(1.0 / 18.0) * log2( half(2) * rcp(a2) - half(1) ) );

	half F0 = LoH + pow5( half(1) - LoH );
	half FdV = half(1 - 0.75) * pow5( half(1) - NoV );
	half FdL = half(1 - 0.75) * pow5( half(1) - NoL );

	// Rough (F0) to smooth (FdV * FdL) response interpolation
	half Fd = mix( F0, FdV * FdL, saturateHalf( half(2.2) * g - half(0.5) ) );

	// Retro reflectivity contribution.
	half Fb = ( half(34.5 * g - 59) * g + half(24.5) ) * LoH * exp2( -max( half(73.2) * g - half(21.2), half(8.9) ) * sqrt( NoH ) );

	half Lobe = half(1 / M_PI) * (Fd + Fb) * NoL;

	// We clamp the BRDF lobe value to an arbitrary value of 1 to get some practical benefits at high roughness:
	// - This is to avoid too bright edges when using normal map on a mesh and the local bases, L, N and V ends up in an top emisphere setup.
	// - This maintains the full proper rough look of a sphere when not using normal maps.
	// - This also fixes the furnace test returning too much energy at the edge of a mesh.
	return min(half(1.0), Lobe);
}

// scales the specular reflections, needs to be be computed before lighting happens,
// but after environment, GI, and reflection probes are added
// Environment brdf approximation (Lazarov 2013)
// see https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
hvec2 BRDF_Aprox(half roughness, half NoV) {
	const hvec4 c0 = hvec4(-1.0, -0.0275, -0.572, 0.022);
	const hvec4 c1 = hvec4(1.0, 0.0425, 1.04, -0.04);
	hvec4 r = roughness * c0 + c1;

	half a004 = min(r.x * r.x, exp2(half(-9.28) * NoV)) * r.x + r.y;
	return hvec2(-1.04, 1.04) * a004 + r.zw;
}

// Dielectric (IOR=1.5) simplification of the full BRDF approximation above, from the same source.
half BRDF_Aprox_Nonmetal(half roughness, half NoV){
	const hvec4 c0 = hvec4(-1.0, -0.0275, -0.572, 0.022);
	const hvec4 c1 = hvec4(1.0, 0.0425, 1.04, -0.04);
	hvec2 r = roughness * c0.xy + c1.xy;
	return min(r.x * r.x, exp2(half(-9.28) * NoV)) * r.x + r.y;
}
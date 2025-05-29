// Functions related to lighting

#include "math_inc.glsl"

float D_GGX(float cos_theta_m, float alpha) {
	float a = cos_theta_m * alpha;
	float k = alpha / (1.0 - cos_theta_m * cos_theta_m + a * a);
	return k * k * (1.0 / M_PI);
}

// From "Course Notes: Moving Frostbite to PBR", page 12: https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
float V_GGX(float NdotL, float NdotV, float alpha) {
	float Lambda_GGXV = NdotL * sqrt_IEEE_int_approximation((-NdotV * alpha + NdotV) * NdotV + alpha);
	float Lambda_GGXL = NdotV * sqrt_IEEE_int_approximation((-NdotL * alpha + NdotL) * NdotL + alpha);

	return 0.5 / (Lambda_GGXV + Lambda_GGXL);
}

float D_GGX_anisotropic(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float alpha2 = alpha_x * alpha_y;
	highp vec3 v = vec3(alpha_y * cos_phi, alpha_x * sin_phi, alpha2 * cos_theta_m);
	highp float v2 = dot(v, v);
	float w2 = alpha2 / v2;
	float D = alpha2 * w2 * w2 * (1.0 / M_PI);
	return D;
}

// From "Course Notes: Moving Frostbite to PBR", page 12: https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
// This is a modified version to take anisotropy into account
float V_GGX_anisotropic(float alpha_x, float alpha_y, float TdotV, float TdotL, float BdotV, float BdotL, float NdotV, float NdotL) {
	float Lambda_GGXV = NdotL * sqrt_IEEE_int_approximation((-NdotV * (alpha_x * TdotV * TdotV + alpha_y * BdotV * BdotV) + NdotV) * NdotV + alpha_x * TdotV * TdotV + alpha_y * BdotV * BdotV);
	float Lambda_GGXL = NdotV * sqrt_IEEE_int_approximation((-NdotL * (alpha_x * TdotL * TdotL + alpha_y * BdotL * BdotL) + NdotL) * NdotL + alpha_x * TdotL * TdotL + alpha_y * BdotL * BdotL);

	return 0.5 / (Lambda_GGXV + Lambda_GGXL);
}

vec3 SchlickFresnel(vec3 f0, float f90, float u) {
	return f0 + (f90 - f0) * pow5(1.0 - u);
}

float SchlickFresnel(float f0, float f90, float u){
	return f0 + (f90 - f0) * pow5(1.0 - u);
}

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}

float D_Charlie(float roughness, float NoH) {
    // Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
    float invAlpha  = 1.0 / roughness;
    float cos2h = NoH * NoH;
    float sin2h = 1.0 - cos2h;
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * M_PI);
}

float V_Neubelt(float NoV, float NoL) {
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    return 1.0 / (4.0 * (NoL + NoV - NoL * NoV));
}

float V_Kelemen(float LoH) {
    // Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
    return 0.25 / (LoH * LoH);
}

float Diffuse_Lambert(float NoL) {
	return NoL * (1.0 / M_PI);
}

// Energy conserving lambert wrap shader.
// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
float Diffuse_Lambert_Wrap(float NoL, float roughness){
	return max(0.0, (NoL + roughness) / ((1.0 + roughness) * (1.0 + roughness))) * (1.0 / M_PI);
}

float Diffuse_Burley(float LoH, float NoV, float NoL, float roughness){
	float FD90_minus_1 = 2.0 * LoH * LoH * roughness - 0.5;
	float FdV = 1.0 + FD90_minus_1 * pow5(NoV);
	float FdL = 1.0 + FD90_minus_1 * pow5(NoL);
	return (1.0 / M_PI) * FdV * FdL * NoL;
}

// Normalized Disney diffuse function taken from Frostbite's PBR course notes (page 10):
// https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
float Normalized_Diffuse_Burley(float roughness, float NoV, float NoL, float LoH)
{
    float energyBias = mix(0.0, 0.5, roughness);
    float energyFactor = mix(1.0, 1.0/1.51, roughness);
    float fd90 = energyBias + 2.0 * LoH * LoH * roughness;
	float f0 = 1.0;
    float lightScatter = SchlickFresnel(f0, fd90, NoL);
    float viewScatter = SchlickFresnel(f0, fd90, NoV);

    return lightScatter * viewScatter * energyFactor * NoL * (1.0 / M_PI);
}

float Diffuse_Toon(float NoL, float roughness){
	return smoothstep(-roughness, max(roughness, 0.01), NoL) * (1.0 / M_PI);
}

// [ Chan 2018, "Material Advances in Call of Duty: WWII" ]
float Diffuse_Chan(float roughness, float NoV, float NoL, float LoH, float NoH)
{
	float a2 = roughness * roughness;

	// a2 = 2 / ( 1 + exp2( 18 * g )
	float g = saturate( (1.0 / 18.0) * log2( 2 * rcp(a2) - 1 ) );

	float F0 = LoH + pow5( 1 - LoH );
	float FdV = 1 - 0.75 * pow5( 1 - NoV );
	float FdL = 1 - 0.75 * pow5( 1 - NoL );

	// Rough (F0) to smooth (FdV * FdL) response interpolation
	float Fd = mix( F0, FdV * FdL, saturate( 2.2 * g - 0.5 ) );

	// Retro reflectivity contribution.
	float Fb = ( (34.5 * g - 59 ) * g + 24.5 ) * LoH * exp2( -max( 73.2 * g - 21.2, 8.9 ) * sqrt( NoH ) );

	float Lobe = (1 / M_PI) * (Fd + Fb) * NoL;

	// We clamp the BRDF lobe value to an arbitrary value of 1 to get some practical benefits at high roughness:
	// - This is to avoid too bright edges when using normal map on a mesh and the local bases, L, N and V ends up in an top emisphere setup.
	// - This maintains the full proper rough look of a sphere when not using normal maps.
	// - This also fixes the furnace test returning too much energy at the edge of a mesh.
	return min(1.0, Lobe);
}

// scales the specular reflections, needs to be be computed before lighting happens,
// but after environment, GI, and reflection probes are added
// Environment brdf approximation (Lazarov 2013)
// see https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
vec2 BRDF_Aprox(float roughness, float NoV) {
	const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
	const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
	vec4 r = roughness * c0 + c1;

	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Dielectric (IOR=1.5) simplification of the full BRDF approximation above, from the same source.
float BRDF_Aprox_Nonmetal(float roughness, float NoV){
	const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
	const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
	vec2 r = roughness * c0.xy + c1.xy;
	return min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
}
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
	half v = half(0.5) / mix(half(2.0) * NdotL * NdotV, NdotL + NdotV, alpha);
	return saturateHalf(v);
}

half D_GGX_anisotropic(half cos_theta_m, half alpha_x, half alpha_y, half cos_phi, half sin_phi) {
	half alpha2 = alpha_x * alpha_y;
	vec3 v = vec3(alpha_y * cos_phi, alpha_x * sin_phi, alpha2 * cos_theta_m);
	float v2 = dot(v, v);
	half w2 = half(float(alpha2) / v2);
	return alpha2 * w2 * w2 * half(1.0 / M_PI);
}

half V_GGX_anisotropic(half alpha_x, half alpha_y, half TdotV, half TdotL, half BdotV, half BdotL, half NdotV, half NdotL) {
	half Lambda_V = NdotL * length(hvec3(alpha_x * TdotV, alpha_y * BdotV, NdotV));
	half Lambda_L = NdotV * length(hvec3(alpha_x * TdotL, alpha_y * BdotL, NdotL));
	half v = half(0.5) / (Lambda_V + Lambda_L);
	return saturateHalf(v);
}

hvec3 SchlickFresnel(hvec3 f0, half f90, half u) {
	return f0 + (f90 - f0) * pow5(half(1.0) - u);
}

half SchlickFresnel(half f0, half f90, half u) {
	return f0 + (f90 - f0) * pow5(half(1.0) - u);
}

hvec3 F0(half metallic, half specular, hvec3 albedo) {
	half dielectric = half(0.16) * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(hvec3(dielectric), albedo, hvec3(metallic));
}

hvec3 f0_Clear_Coat_To_Surface(hvec3 f0) {
	// Approximation of iorTof0(f0ToIor(f0), 1.5)
	// This assumes that the clear coat layer has an IOR of 1.5
	// see https://github.com/google/filament/blob/837b2715a05f4656d4f524bce50d1b23ff8f84c9/shaders/src/surface_material.fs#L54-L62
	return clamp(f0 * (f0 * (0.941892 - 0.263008 * f0) + 0.346479) - 0.0285998, hvec3(0.0), hvec3(1.0));
}

float D_Charlie(float roughness, float NoH) {
	// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
	float invAlpha = 1.0 / roughness;
	float cos2h = NoH * NoH;
	float sin2h = 1.0 - cos2h;
	return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * M_PI);
}

float V_Neubelt(float NoV, float NoL) {
	// Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
	return 1.0 / (4.0 * (NoL + NoV - NoL * NoV));
}

half V_Kelemen(half LoH) {
	// Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
	return saturateHalf(half(0.25) / (LoH * LoH + 1e-4));
}

half Diffuse_Lambert(half NoL) {
	return NoL * half(1.0 / M_PI);
}

// Energy conserving lambert wrap shader.
// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
half Diffuse_Lambert_Wrap(half roughness, half NoL) {
	half op_roughness = half(1.0) + roughness;
	return max(half(0.0), (NoL + roughness) / (op_roughness * op_roughness)) * half(1.0 / M_PI);
}

half Diffuse_Burley(half roughness, half NoV, half NoL, half LoH) {
	half FD90_minus_1 = half(2.0) * LoH * LoH * roughness - half(0.5);
	half FdV = half(1.0) + FD90_minus_1 * pow5(half(1.0) - NoV);
	half FdL = half(1.0) + FD90_minus_1 * pow5(half(1.0) - NoL);
	return half(1.0 / M_PI) * FdV * FdL * NoL;
}

// Normalized Disney diffuse function taken from Frostbite's PBR course notes (page 10):
// https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v32.pdf
half Normalized_Diffuse_Burley(half roughness, half NoV, half NoL, half LoH) {
	half energyBias = mix(half(0.0), half(0.5), roughness);
	half energyFactor = mix(half(1.0), half(1.0 / 1.51), roughness);
	half fd90 = energyBias + half(2.0) * LoH * LoH * roughness;
	half f0 = half(1.0);
	half lightScatter = SchlickFresnel(f0, fd90, NoL);
	half viewScatter = SchlickFresnel(f0, fd90, NoV);

	return lightScatter * viewScatter * energyFactor * NoL * half(1.0 / M_PI);
}

half Diffuse_Toon(half NoL, half roughness) {
	return smoothstep(-roughness, max(roughness, half(0.01)), NoL) * half(1.0 / M_PI);
}

// [ Chan 2024, "Multiscattering Diffuse and Specular BRDFs", Unpublished manuscript ]
half Diffuse_Chan(half roughness, half NoL, half LoH, half NoH) {
	half Alpha = roughness * roughness;
	// The original writeup uses an FSmooth term inspired by Burley diffuse to balance energy between spec/diffuse.
	// However in our implementation the energy balance between diffuse and spec is handled externally, so we stick
	// to a plain lambertian for the Roughness=0 limit.
	half FSmooth = half(1.0);
	half Scale = max(half(0.55) - half(0.2) * roughness, half(1.25) - half(1.6) * roughness);
	const float Bias = saturate(half(4.0) * Alpha);
	const float FRough = Scale * (NoH + Bias) * rcp(NoH + half(0.025)) * LoH * LoH;
	const float DiffuseSS = mix(FSmooth, FRough, roughness);
	const float DiffuseMS = Alpha * half(0.38);
	return half(1.0 / M_PI) * (DiffuseSS + DiffuseMS) * NoL;
}

// https://dl.acm.org/doi/pdf/10.1145/192161.192213
float Diffuse_OrenNayar(float roughness, float NoV, float NoL, float LoV, vec3 light, vec3 view, vec3 normal) {
	const float INV_PI = 0.318309;

	float a = roughness * roughness; // Variance.

	vec3 r = 2.0 * NoL * normal - light; // Radiance.

	float NdotR = dot(normal, r);

	float theta_i = min(acos(NoL), 1e-4);
	float theta_r = min(acos(NdotR), 1e-4);

	vec3 l_proj = normalize(light - NoL * normal);
	vec3 v_proj = normalize(view - NoV * normal + 1.0);

	float cos_phi = dot(l_proj, v_proj);

	float alpha = max(theta_i, theta_r);
	float beta = min(theta_i, theta_r);

	float C1 = 1.0 - 0.5 * (a / (a + 0.33));

	float C2 = mix(
			0.45 * (a / (a + 0.09)) * sin(alpha),
			0.45 * (a / (a + 0.09)) * (sin(alpha) - pow((2.0 * beta) / M_PI, 3.0)),
			step(0.0, cos_phi));

	float C3 = 0.125 * (a / (a + 0.09)) * pow((4.0 * alpha * beta) / (M_PI * M_PI), 2.0);

	float L1 = cos(theta_i) * (C1 + C2 * cos_phi * tan(beta) + C3 * (1.0 - abs(cos_phi)) * tan((alpha + beta) / 2.0));
	float L2 = 0.17 * cos(theta_i) * ((a) / (a + 0.13)) * ((1.0 - cos_phi) * pow((2.0 * beta) / M_PI, 2.0));

	return max(min(L1 + L2, 1.0), 0.0) * NoL;
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
half BRDF_Aprox_Nonmetal(half roughness, half NoV) {
	const hvec4 c0 = hvec4(-1.0, -0.0275, -0.572, 0.022);
	const hvec4 c1 = hvec4(1.0, 0.0425, 1.04, -0.04);
	hvec2 r = roughness * c0.xy + c1.xy;
	return min(r.x * r.x, exp2(half(-9.28) * NoV)) * r.x + r.y;
}

#define M_PI 3.14159265359
#define M_TAU 6.28318530718
#define ROUGHNESS_MAX_LOD 5

#define MAX_VOXEL_GI_INSTANCES 8
#define MAX_VIEWS 2

#ifndef MOLTENVK_USED
#if defined(has_GL_KHR_shader_subgroup_ballot) && defined(has_GL_KHR_shader_subgroup_arithmetic)

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define USE_SUBGROUPS
#endif
#endif // MOLTENVK_USED

#if defined(USE_MULTIVIEW) && defined(has_VK_KHR_multiview)
#extension GL_EXT_multiview : enable
#endif

#include "../cluster_data_inc.glsl"
#include "../decal_data_inc.glsl"
#include "../scene_data_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) || defined(MODE_RENDER_MATERIAL) || defined(MODE_RENDER_SDF) || defined(MODE_RENDER_NORMAL_ROUGHNESS) || defined(MODE_RENDER_VOXEL_GI) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif

#if !defined(TANGENT_USED) && (defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED))
#define TANGENT_USED
#endif

layout(push_constant, std430) uniform DrawCall {
	uint instance_index;
	uint uv_offset;
	uint multimesh_motion_vectors_current_offset;
	uint multimesh_motion_vectors_previous_offset;
#ifdef UBERSHADER
	uint sc_packed_0;
	uint sc_packed_1;
	uint sc_packed_2;
	uint uc_packed_0;
#endif
}
draw_call;

/* Specialization Constants */

#ifdef UBERSHADER

#define POLYGON_CULL_DISABLED 0
#define POLYGON_CULL_FRONT 1
#define POLYGON_CULL_BACK 2

// Pull the constants from the draw call's push constants.
uint sc_packed_0() {
	return draw_call.sc_packed_0;
}

uint sc_packed_1() {
	return draw_call.sc_packed_1;
}

uint uc_cull_mode() {
	return (draw_call.uc_packed_0 >> 0) & 3U;
}

#else

// Pull the constants from the pipeline's specialization constants.
layout(constant_id = 0) const uint pso_sc_packed_0 = 0;
layout(constant_id = 1) const uint pso_sc_packed_1 = 0;

uint sc_packed_0() {
	return pso_sc_packed_0;
}

uint sc_packed_1() {
	return pso_sc_packed_1;
}

#endif

bool sc_use_forward_gi() {
	return ((sc_packed_0() >> 0) & 1U) != 0;
}

bool sc_use_light_projector() {
	return ((sc_packed_0() >> 1) & 1U) != 0;
}

bool sc_use_light_soft_shadows() {
	return ((sc_packed_0() >> 2) & 1U) != 0;
}

bool sc_use_directional_soft_shadows() {
	return ((sc_packed_0() >> 3) & 1U) != 0;
}

bool sc_decal_use_mipmaps() {
	return ((sc_packed_0() >> 4) & 1U) != 0;
}

bool sc_projector_use_mipmaps() {
	return ((sc_packed_0() >> 5) & 1U) != 0;
}

bool sc_use_depth_fog() {
	return ((sc_packed_0() >> 6) & 1U) != 0;
}

bool sc_use_lightmap_bicubic_filter() {
	return ((sc_packed_0() >> 7) & 1U) != 0;
}

uint sc_soft_shadow_samples() {
	return (sc_packed_0() >> 8) & 63U;
}

uint sc_penumbra_shadow_samples() {
	return (sc_packed_0() >> 14) & 63U;
}

uint sc_directional_soft_shadow_samples() {
	return (sc_packed_0() >> 20) & 63U;
}

uint sc_directional_penumbra_shadow_samples() {
	return (sc_packed_0() >> 26) & 63U;
}

bool sc_multimesh() {
	return ((sc_packed_1() >> 0) & 1U) != 0;
}

bool sc_multimesh_format_2d() {
	return ((sc_packed_1() >> 1) & 1U) != 0;
}

bool sc_multimesh_has_color() {
	return ((sc_packed_1() >> 2) & 1U) != 0;
}

bool sc_multimesh_has_custom_data() {
	return ((sc_packed_1() >> 3) & 1U) != 0;
}

float sc_luminance_multiplier() {
	// Not used in clustered renderer but we share some code with the mobile renderer that requires this.
	return 1.0;
}

#define SDFGI_MAX_CASCADES 8

/* Set 0: Base Pass (never changes) */

#include "../light_data_inc.glsl"

layout(set = 0, binding = 2) uniform sampler shadow_sampler;

#define INSTANCE_FLAGS_DYNAMIC (1 << 3)
#define INSTANCE_FLAGS_NON_UNIFORM_SCALE (1 << 4)
#define INSTANCE_FLAGS_USE_GI_BUFFERS (1 << 5)
#define INSTANCE_FLAGS_USE_SDFGI (1 << 6)
#define INSTANCE_FLAGS_USE_LIGHTMAP_CAPTURE (1 << 7)
#define INSTANCE_FLAGS_USE_LIGHTMAP (1 << 8)
#define INSTANCE_FLAGS_USE_SH_LIGHTMAP (1 << 9)
#define INSTANCE_FLAGS_USE_VOXEL_GI (1 << 10)
#define INSTANCE_FLAGS_PARTICLES (1 << 11)
#define INSTANCE_FLAGS_PARTICLE_TRAIL_SHIFT 16
#define INSTANCE_FLAGS_FADE_SHIFT 24
//3 bits of stride
#define INSTANCE_FLAGS_PARTICLE_TRAIL_MASK 0xFF

#define SCREEN_SPACE_EFFECTS_FLAGS_USE_SSAO 1
#define SCREEN_SPACE_EFFECTS_FLAGS_USE_SSIL 2

layout(set = 0, binding = 3, std430) restrict readonly buffer OmniLights {
	LightData data[];
}
omni_lights;

layout(set = 0, binding = 4, std430) restrict readonly buffer SpotLights {
	LightData data[];
}
spot_lights;

layout(set = 0, binding = 5, std430) restrict readonly buffer ReflectionProbeData {
	ReflectionData data[];
}
reflections;

layout(set = 0, binding = 6, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

#define LIGHTMAP_FLAG_USE_DIRECTION 1
#define LIGHTMAP_FLAG_USE_SPECULAR_DIRECTION 2

#define LIGHTMAP_SHADOWMASK_MODE_NONE 0
#define LIGHTMAP_SHADOWMASK_MODE_REPLACE 1
#define LIGHTMAP_SHADOWMASK_MODE_OVERLAY 2
#define LIGHTMAP_SHADOWMASK_MODE_ONLY 3

struct Lightmap {
	mat3 normal_xform;
	vec2 light_texture_size;
	float exposure_normalization;
	uint flags;
};

layout(set = 0, binding = 7, std140) restrict readonly buffer Lightmaps {
	Lightmap data[];
}
lightmaps;

struct LightmapCapture {
	vec4 sh[9];
};

layout(set = 0, binding = 8, std140) restrict readonly buffer LightmapCaptures {
	LightmapCapture data[];
}
lightmap_captures;

layout(set = 0, binding = 9) uniform texture2D decal_atlas;
layout(set = 0, binding = 10) uniform texture2D decal_atlas_srgb;

layout(set = 0, binding = 11, std430) restrict readonly buffer Decals {
	DecalData data[];
}
decals;

layout(set = 0, binding = 12, std430) restrict readonly buffer GlobalShaderUniformData {
	vec4 data[];
}
global_shader_uniforms;

struct SDFVoxelGICascadeData {
	vec3 position;
	float to_probe;
	ivec3 probe_world_offset;
	float to_cell; // 1/bounds * grid_size
	vec3 pad;
	float exposure_normalization;
};

layout(set = 0, binding = 13, std140) uniform SDFGI {
	vec3 grid_size;
	uint max_cascades;

	bool use_occlusion;
	int probe_axis_size;
	float probe_to_uvw;
	float normal_bias;

	vec3 lightprobe_tex_pixel_size;
	float energy;

	vec3 lightprobe_uv_offset;
	float y_mult;

	vec3 occlusion_clamp;
	uint pad3;

	vec3 occlusion_renormalize;
	uint pad4;

	vec3 cascade_probe_size;
	uint pad5;

	SDFVoxelGICascadeData cascades[SDFGI_MAX_CASCADES];
}
sdfgi;

layout(set = 0, binding = 14) uniform sampler DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP;

layout(set = 0, binding = 15) uniform texture2D best_fit_normal_texture;

layout(set = 0, binding = 16) uniform texture2D dfg;

/* Set 1: Render Pass (changes per render pass) */

layout(set = 1, binding = 0, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene_data_block;

struct ImplementationData {
	uint cluster_shift;
	uint cluster_width;
	uint cluster_type_size;
	uint max_cluster_element_count_div_32;

	uint ss_effects_flags;
	float ssao_light_affect;
	float ssao_ao_affect;
	uint pad1;

	mat4 sdf_to_bounds;

	ivec3 sdf_offset;
	uint pad2;

	ivec3 sdf_size;
	bool gi_upscale_for_msaa;

	bool volumetric_fog_enabled;
	float volumetric_fog_inv_length;
	float volumetric_fog_detail_spread;
	uint volumetric_fog_pad;
};

layout(set = 1, binding = 1, std140) uniform ImplementationDataBlock {
	ImplementationData data;
}
implementation_data_block;

#define implementation_data implementation_data_block.data

struct InstanceData {
	mat4 transform;
	mat4 prev_transform;
	uint flags;
	uint instance_uniforms_ofs; //base offset in global buffer for instance variables
	uint gi_offset; //GI information when using lightmapping (VCT or lightmap index)
	uint layer_mask;
	vec4 lightmap_uv_scale;
	vec4 compressed_aabb_position_pad; // Only .xyz is used. .w is padding.
	vec4 compressed_aabb_size_pad; // Only .xyz is used. .w is padding.
	vec4 uv_scale;
};

layout(set = 1, binding = 2, std430) buffer restrict readonly InstanceDataBuffer {
	InstanceData data[];
}
instances;

#ifdef USE_RADIANCE_CUBEMAP_ARRAY

layout(set = 1, binding = 3) uniform textureCubeArray radiance_cubemap;

#else

layout(set = 1, binding = 3) uniform textureCube radiance_cubemap;

#endif

layout(set = 1, binding = 4) uniform textureCubeArray reflection_atlas;

layout(set = 1, binding = 5) uniform texture2D shadow_atlas;

layout(set = 1, binding = 6) uniform texture2D directional_shadow_atlas;

layout(set = 1, binding = 7) uniform texture2DArray lightmap_textures[MAX_LIGHTMAP_TEXTURES * 2];

layout(set = 1, binding = 8) uniform texture3D voxel_gi_textures[MAX_VOXEL_GI_INSTANCES];

layout(set = 1, binding = 9, std430) buffer restrict readonly ClusterBuffer {
	uint data[];
}
cluster_buffer;

layout(set = 1, binding = 10) uniform sampler decal_sampler;

layout(set = 1, binding = 11) uniform sampler light_projector_sampler;

layout(set = 1, binding = 12 + 0) uniform sampler SAMPLER_NEAREST_CLAMP;
layout(set = 1, binding = 12 + 1) uniform sampler SAMPLER_LINEAR_CLAMP;
layout(set = 1, binding = 12 + 2) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_CLAMP;
layout(set = 1, binding = 12 + 3) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP;
layout(set = 1, binding = 12 + 4) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_CLAMP;
layout(set = 1, binding = 12 + 5) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP;
layout(set = 1, binding = 12 + 6) uniform sampler SAMPLER_NEAREST_REPEAT;
layout(set = 1, binding = 12 + 7) uniform sampler SAMPLER_LINEAR_REPEAT;
layout(set = 1, binding = 12 + 8) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT;
layout(set = 1, binding = 12 + 9) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT;
layout(set = 1, binding = 12 + 10) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_REPEAT;
layout(set = 1, binding = 12 + 11) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT;

#ifdef MODE_RENDER_SDF

layout(r16ui, set = 1, binding = 24) uniform restrict writeonly uimage3D albedo_volume_grid;
layout(r32ui, set = 1, binding = 25) uniform restrict writeonly uimage3D emission_grid;
layout(r32ui, set = 1, binding = 26) uniform restrict writeonly uimage3D emission_aniso_grid;
layout(r32ui, set = 1, binding = 27) uniform restrict uimage3D geom_facing_grid;

//still need to be present for shaders that use it, so remap them to something
#define depth_buffer shadow_atlas
#define color_buffer shadow_atlas
#define normal_roughness_buffer shadow_atlas

#define multiviewSampler sampler2D
#else

#ifdef USE_MULTIVIEW
layout(set = 1, binding = 24) uniform texture2DArray depth_buffer;
layout(set = 1, binding = 25) uniform texture2DArray color_buffer;
layout(set = 1, binding = 26) uniform texture2DArray normal_roughness_buffer;
layout(set = 1, binding = 27) uniform texture2DArray ao_buffer;
layout(set = 1, binding = 28) uniform texture2DArray ambient_buffer;
layout(set = 1, binding = 29) uniform texture2DArray reflection_buffer;
#define multiviewSampler sampler2DArray
#else // USE_MULTIVIEW
layout(set = 1, binding = 24) uniform texture2D depth_buffer;
layout(set = 1, binding = 25) uniform texture2D color_buffer;
layout(set = 1, binding = 26) uniform texture2D normal_roughness_buffer;
layout(set = 1, binding = 27) uniform texture2D ao_buffer;
layout(set = 1, binding = 28) uniform texture2D ambient_buffer;
layout(set = 1, binding = 29) uniform texture2D reflection_buffer;
#define multiviewSampler sampler2D
#endif
layout(set = 1, binding = 30) uniform texture2DArray sdfgi_lightprobe_texture;
layout(set = 1, binding = 31) uniform texture3D sdfgi_occlusion_cascades;

struct VoxelGIData {
	mat4 xform; // 64 - 64

	vec3 bounds; // 12 - 76
	float dynamic_range; // 4 - 80

	float bias; // 4 - 84
	float normal_bias; // 4 - 88
	bool blend_ambient; // 4 - 92
	uint mipmaps; // 4 - 96

	vec3 pad; // 12 - 108
	float exposure_normalization; // 4 - 112
};

layout(set = 1, binding = 32, std140) uniform VoxelGIs {
	VoxelGIData data[MAX_VOXEL_GI_INSTANCES];
}
voxel_gi_instances;

layout(set = 1, binding = 33) uniform texture3D volumetric_fog_texture;

#ifdef USE_MULTIVIEW
layout(set = 1, binding = 34) uniform texture2DArray ssil_buffer;
#else
layout(set = 1, binding = 34) uniform texture2D ssil_buffer;
#endif // USE_MULTIVIEW

#endif

vec4 normal_roughness_compatibility(vec4 p_normal_roughness) {
	float roughness = p_normal_roughness.w;
	if (roughness > 0.5) {
		roughness = 1.0 - roughness;
	}
	roughness /= (127.0 / 255.0);
	return vec4(normalize(p_normal_roughness.xyz * 2.0 - 1.0) * 0.5 + 0.5, roughness);
}

vec3 prefiltered_dfg(float lod, float NoV) {
    return textureLod(sampler2D(dfg, SAMPLER_LINEAR_CLAMP), vec2(NoV, 1.0 - lod), 0.0).rgb;
}

// Multiscatter from https://github.com/o3de/o3de/blob/development/Gems/Atom/Feature/Common/Assets/ShaderLib/Atom/Features/PBR/LightingUtils.azsli
vec3 get_energy_compensation(vec3 f0, vec2 env){
	// returned values of BRDF are formed by split sum approximation as shown below
    // brdf.x = integral{(BRDF / F) * (1 - (1 - VdotH)^5) * NdotL dL} 
    // brdf.y = integral{(BRDF / F) * (1 - VdotH)^5 * NdotL dL}
    // brdf.x + brdf.y = integral{ (BRDF / F) * (1 - alpha + alpha) * NdotL dL }
    //                 = integral{ (BRDF / F) * NdotL dL }
    //                 = integral{ ((GD / 4 * NdotL * NdotV)) * NdotL dL }
    // which is the integral of microfacet BRDF by assuming fresnel term F == 1 that represents total single scattering reflectance
    // for more information about compensation term please see:
    // https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
	vec3 compensation = f0 * ((1.0 / (env.x + env.y)) - 1.0);
	return compensation + 1.0;
}

/*float compute_micro_shadowing(float NoL, float visibility) {
    // Chan 2018, "Material Advances in Call of Duty: WWII"
    float aperture = inversesqrt(1.0 - min(visibility, 0.9999));
    float microShadow = clamp(NoL * aperture, 0, 1);
    return microShadow * microShadow;
}*/

void extremum_points(float radius_x, float radius_y, vec3 wo, float anisotropyAngle, out vec2 closeExtremum, out vec2 farExtremum, out vec2 roots)
{
    //==============================================================================//
    //================== Closed-form solution - Appendix A =========================//
    //==============================================================================//
    mat2 R = mat2(cos(anisotropyAngle), -sin(anisotropyAngle), sin(anisotropyAngle), cos(anisotropyAngle));
    vec2 o = wo.xy*R;
    
    vec2 o2 = o.xy*o.xy;
    float r2x = radius_x*radius_x;
    float r2y = radius_y*radius_y;
    float p0 = 2.*o.x*o.y*radius_x*radius_y;
    float p1 = o2.x*(r2x*r2y + r2y) - o2.y*(r2x*r2y + r2x) + r2x -r2y;
    float p2 = -p0*(r2y + 1.);
    float p3 = p0*p2*(r2x + 1.);
    float t0 = p2 == 0. ? step(0., p1)*M_PI/2. : atan((p1+sqrt(p1*p1-p3))/p2);
    float t1 = p2 == 0. ? t0+M_PI/2. : atan((p1-sqrt(p1*p1-p3))/p2);
    
    //==============================================================================//
    //==================== Return the slope space extrema ==========================//
    //==============================================================================//
    
    closeExtremum = R*vec2(radius_x*cos(t0), radius_y*sin(t0));
    farExtremum = R*vec2(radius_x*cos(t1), radius_y*sin(t1));
    roots = vec2(t0, t1);
}

vec3 slope_to_normal(vec2 p) {
    return normalize(vec3(-p, 1));
}

void clamp_extrema(inout vec3 farNormal0, inout vec3 farNormal1, vec3 wo, out vec2 tau)
{
    //==============================================================================//
    //================== Closed-form solution - Appendix B =========================//
    //==============================================================================//
    float p0 = farNormal1.x*wo.x + farNormal1.y*wo.y;
    float p1 = wo.z*(1.-farNormal1.z*farNormal1.z);
    float p3 = sqrt(p0*p0 + wo.z*p1);
    float t0 = max(0.5+farNormal1.z*(p0-p3)/(2.*p1), 0.);
    float t1 = min(0.5+farNormal1.z*(p0+p3)/(2.*p1), 1.);
    
    vec3 tmp = farNormal0;
    farNormal0 = normalize(mix(farNormal0, farNormal1, t0));
    farNormal1 = normalize(mix(farNormal0, farNormal1, t1));
    tau = vec2(t0, t1);
}

void shift_extrema(float radius_x, float radius_y, vec3 wo, float anisotropyAngle, vec2 roots, vec2 tau, inout vec3 farNormal0, inout vec3 farNormal1)
{
    //==============================================================================//
    //============== Update the closest extrema - Appendix C =======================//
    //==============================================================================//
    mat2 R = mat2(cos(anisotropyAngle), -sin(anisotropyAngle), sin(anisotropyAngle), cos(anisotropyAngle));
    float angleDelta = (tau.x > 0. ? tau.x : tau.y-1.)*(roots.x-roots.y);
    float min0t = roots.x+angleDelta;
    float min1t = roots.x-angleDelta+M_PI;
    vec2 close0Slope = R*vec2(radius_x*cos(min0t), radius_y*sin(min0t));
    vec2 close1Slope = R*vec2(radius_x*cos(min1t), radius_y*sin(min1t));
    vec3 close0Normal = slope_to_normal(close0Slope);
    vec3 close1Normal = slope_to_normal(close1Slope);
    
    //==============================================================================//
    //======================== Shift the furthest extrema ==========================//
    //==============================================================================//
    float minLength = length(reflect(-wo, close0Normal)-reflect(-wo, close1Normal));
    float maxLength = length(reflect(-wo, farNormal0)-reflect(-wo, farNormal1));
    float rateo = (maxLength-minLength)/maxLength;
    vec3 axisCenter = normalize(farNormal0+farNormal1);
    farNormal0 = normalize(axisCenter + rateo*(farNormal0-axisCenter));
    farNormal1 = normalize(axisCenter + rateo*(farNormal1-axisCenter));
}

/* Set 2 Skeleton & Instancing (can change per item) */

layout(set = 2, binding = 0, std430) restrict readonly buffer Transforms {
	vec4 data[];
}
transforms;

/* Set 3 User Material */

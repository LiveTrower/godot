#[compute]

#version 450

#VERSION_DEFINES

#include "../math_inc.glsl"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_image;
layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;
layout(set = 2, binding = 0) uniform sampler2D source_depth;

#define EPSILON 10e-5

#define GROUP_DIM_X    16
#define GROUP_DIM_Y    16

#define PAD 2
#define PAD2 (PAD * 2)

#define SHARED_MEMORY_SIZE 400

#define SAMPLE_RATIO_1  0.3
#define SAMPLE_RATIO_2  0.6

#define NUMBER_OF_SAMPLE 200

#define FOOTPRINT_SMALL     1
#define FOOTPRINT_MEDIUM 4

#define MILLIMETER_TO_METER 1e-3
#define METER_TO_MILLIMETER 1e3

#define DEPTH_CLIP_THRESHOLD_SCALE (0.5 * MILLIMETER_TO_METER)

#define M_PI 3.14159265359
#define ONE_OVER_PI 0.31830988618

#ifdef USE_HIGH_QUALITY
const int quality = 1.0;
#endif
#ifdef USE_MEDIUM_QUALITY
const int quality = 0.6;
#endif
#ifdef USE_LOW_QUALITY
const int quality = 0.3;
#endif

shared float s_color_r[SHARED_MEMORY_SIZE];
shared float s_color_g[SHARED_MEMORY_SIZE];
shared float s_color_b[SHARED_MEMORY_SIZE];
shared float s_depth[SHARED_MEMORY_SIZE];

layout(push_constant) uniform Params {
    vec2 screen_size;
	float unit_size;

	float scale;
    float scatter_distance;
	mat4 projection;
} params;

void set_color_depth(uint index, vec3 color, float depth) {
	s_color_r[index] = color.r;
	s_color_g[index] = color.g;
	s_color_b[index] = color.b;
	s_depth[index] = depth;
}

vec3 get_radiance(uint index) {
	return vec3(s_color_r[index], s_color_g[index], s_color_b[index]);
}

float get_depth(uint index) {
	return s_depth[index];
}

vec3 kernel(float l, float r, vec3 s, float minS)
{
    return l > 0 ? (exp(-s * l) + exp(-s * l / 3.0)) / (exp(-minS * r) + exp(-minS * r / 3.0)) : vec3(0.0);
}

float g(float u)
{
    return 4.0 * u * (2.0 * u + sqrt(1.0 + 4.0 * u * u)) + 1.0;
}

float get_sample_inverse_cdf_approx(float u, float s)
{
    float uc = 1.0 - u;
    float oneDivThree = 1.0 / 3.0;
    return 3.0 * log((1.0 + 1.0 / pow(g(uc), oneDivThree) + pow(g(uc), oneDivThree)) / (4.0 * uc)) / s;
}

float SF(uint index)
{
    return 2 * M_PI * index *  2 / (1 + sqrt(5));
}

const float cosTheta[4] = { 1.0, -1.0,  6.123233995736766e-17, -1.8369701987210297e-16 };
const float sinTheta[4] = { 0.0,  1.2246467991473532e-16, 1.0, -1.0 };

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation
float radical_inverse_vdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

void get_sample(in uint index, in uint gridIndex, in float s, out vec3 sp)
{
    float r = get_sample_inverse_cdf_approx(radical_inverse_vdc(index), s);
    float x = r * cos(SF(index));
    float y = r * sin(SF(index));

    sp = vec3(x * cosTheta[gridIndex] - y * sinTheta[gridIndex], 
                x * sinTheta[gridIndex] + y * cosTheta[gridIndex], r);
}

float linear_depth(uvec2 uv) {
	float depth = texture(source_depth, uv).r;
	depth = (params.projection[3][2] - params.projection[3][3] * depth) / (params.projection[2][3] * depth - params.projection[2][2]);
	return params.unit_size / (params.projection[2][3] * depth + params.projection[3][3]);
}

void main() {
	vec4 color_x = texture(source_image, uvec2(gl_GlobalInvocationID.xy));

	uint dim_x_with_pad = GROUP_DIM_X + PAD2;

	uint texel_offset_x = gl_WorkGroupID.x * GROUP_DIM_X - PAD;
    uint texel_offset_y = gl_WorkGroupID.y * GROUP_DIM_Y - PAD;

	uvec2 texel_index = uvec2( 
        texel_offset_x + (gl_LocalInvocationIndex % dim_x_with_pad), 
        texel_offset_y + (gl_LocalInvocationIndex / dim_x_with_pad)
    );

	set_color_depth(gl_LocalInvocationIndex, texture(source_image, texel_index).rgb, linear_depth(texel_index));

	uint total_loading_thread = SHARED_MEMORY_SIZE - GROUP_DIM_X * GROUP_DIM_Y;
	if(gl_LocalInvocationIndex < total_loading_thread) {
		uint index = GROUP_DIM_X * GROUP_DIM_Y + gl_LocalInvocationIndex;
		texel_index = uvec2( 
        	texel_offset_x + (index % dim_x_with_pad), 
        	texel_offset_y + (index / dim_x_with_pad)
    	);
		set_color_depth(index, texture(source_image, texel_index).rgb, linear_depth(texel_index));
	}

	/*if(abs(color_x.a) < EPSILON)
    {
        imageStore(dest_image, ivec2(gl_GlobalInvocationID.xy), vec4(color_x.rgb, 0.0f));
        return;
    }*/

	memoryBarrierShared();
	barrier();

	//+--------------------------------------+
    //|  subsurface scattering preprocessing |
    //+--------------------------------------+

	uvec2 s_mem_id = gl_LocalInvocationID.xy + uvec2(PAD, PAD);
    uint s_mem_index = s_mem_id.y * dim_x_with_pad + s_mem_id.x;

	vec3 s =  rcp(vec3(params.scatter_distance) + EPSILON);
    float  min_s = min(s.x, min(s.y, s.z));

	vec2 texel_size = vec2(1.0 / params.screen_size.x, 1.0 / params.screen_size.y);
    
    float depth_x = get_depth(s_mem_index);

	float profile_to_screen_x = (params.projection[0][0] / depth_x) * MILLIMETER_TO_METER / texel_size.x;
    float profile_to_screen_y = (params.projection[1][1] / depth_x) * MILLIMETER_TO_METER / texel_size.y;

	float max_radius = get_sample_inverse_cdf_approx(0.9, min_s);

	uint max_foot_print = uint(max(max_radius * profile_to_screen_x, max_radius * profile_to_screen_y));

	uint num_sample = uint(NUMBER_OF_SAMPLE * quality * (max_foot_print <= FOOTPRINT_MEDIUM ?  max_foot_print <=  FOOTPRINT_SMALL ? SAMPLE_RATIO_1 : SAMPLE_RATIO_2 : 1.0));

	uint sample_grid_index = (gl_GlobalInvocationID.y % 2) * 2 + gl_GlobalInvocationID.x % 2;

	//+------------------------------------+
    //|  subsurface scattering convolution |
    //+------------------------------------+

	vec3 total_sum = vec3(0.0);
    vec3 weight_sum = vec3(0.0);
    for(uint i = 0; i < num_sample; ++i)
    {
        vec3 sp;
        get_sample(i, sample_grid_index, min_s, sp);

		int u = int(round(sp.x * profile_to_screen_x));
        int v = int(round(sp.y * profile_to_screen_y));

		uvec2 nbr_s_mem_id = s_mem_id + uvec2(u, v);

		float depth_nbr = 0.0;

		vec3 color_nbr = vec3(0.0);

		if(nbr_s_mem_id.x >= dim_x_with_pad || nbr_s_mem_id.x < 0 || nbr_s_mem_id.y >= (GROUP_DIM_Y + PAD2) || nbr_s_mem_id.y < 0)
        {
            uvec2 nbr_texel_index = uvec2(gl_GlobalInvocationID.x + u, gl_GlobalInvocationID.y + v);
            depth_nbr = texture(source_depth, nbr_texel_index).r;
            color_nbr = texture(source_image, nbr_texel_index).rgb;
        }
        else
        {
            nbr_s_mem_id = s_mem_id + uvec2(u, v);
            uint nbr_s_mem_index = nbr_s_mem_id.y * dim_x_with_pad + nbr_s_mem_id.x;
            depth_nbr = get_depth(nbr_s_mem_index);
            color_nbr = get_radiance(nbr_s_mem_index);
        }

		if(depth_nbr > 0.0 && (color_nbr.x + color_nbr.y + color_nbr.z) > 0.0)
        {
            float r = sp.z;
            float d = abs(depth_x - depth_nbr) * METER_TO_MILLIMETER;

            float surface_distance = sqrt(r * r + d * d);
 
            vec3 weight = kernel(surface_distance, r, s, min_s);

            total_sum  += weight * color_nbr;    
            weight_sum += weight;
        }
	}

	vec3 subsurface_scattering_color = total_sum / (weight_sum + EPSILON);

    vec3 scatter_color = vec3(1.0, 0.0, 0.0);
    vec3 out_color = subsurface_scattering_color * scatter_color * abs(color_x.a);

	imageStore(dest_image, ivec2(gl_GlobalInvocationID.xy), vec4(out_color, 1.0));
}
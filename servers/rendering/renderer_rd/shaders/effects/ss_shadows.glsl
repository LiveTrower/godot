#[compute]
#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_depth;
layout(r16f, set = 0, binding = 1) uniform restrict writeonly image2D shadow_image;

#include "ss_shadows_inc.glsl"

layout(set = 0, binding = 2, std140) uniform SceneData {
    mat4 projection;
    mat4 inv_projection;
    float z_far;
    float z_near;
} scene_data;

layout(set = 0, binding = 3, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

layout(set = 0, binding = 4, std430) restrict readonly buffer OmniLights {
	LightData data[];
}
omni_lights;

layout(set = 0, binding = 5, std430) restrict readonly buffer SpotLights {
	LightData data[];
}
spot_lights;

layout(set = 0, binding = 6, std430) buffer restrict readonly ClusterBuffer {
	uint data[];
}
cluster_buffer;

layout(push_constant, std430) uniform Params {
    ivec2 screen_size;
    uint cluster_shift;
	uint cluster_width;
    uint max_cluster_element_count_div_32;
    float thickness;
    uint directional_light_count;
    float taa_frame_count;
    uint pad[4];
} params;

const float sss_max_depth = 5;


float ScreenSpaceShadows(vec2 ssC, vec3 ray_pos, vec3 ray_dir, float step_length)
{
    // Compute ray step
    vec3 ray_step = ray_dir * step_length;

    // Offset starting position with temporal interleaved gradient noise
	float offset  = quick_hash(ssC, params.taa_frame_count);
    ray_pos      += ray_step * offset;

    // Ray march towards the light
    float occlusion = 0.0;
    vec2 ray_uv   = vec2(0.0);
    for (uint i = 0u; i < uint(SAMPLES); i++)
    {
        // Step the ray
        ray_pos += ray_step;
        ray_uv  = view_to_uv(ray_pos, scene_data.projection);

        // Ensure the UV coordinates are inside the screen and if depth is too large
        //if (!is_valid_uv(ray_uv) || depth_z > sss_max_depth)
        if (!is_valid_uv(ray_uv))
            return 1.0f;

        // Compute the difference between the ray's and the camera's depth
        float depth_z = get_linear_depth(ray_uv, scene_data.z_near, scene_data.z_far);
        float depth_delta   = -ray_pos.z - depth_z;

        if ((depth_delta > 0.0f) && (depth_delta < params.thickness))
        {
            // Mark as occluded
            occlusion = 1.0f;

            // Fade out as we approach the edges of the screen
            occlusion *= screen_fade(ray_uv);

            break;
        }
    }

    // Convert to visibility
    return 1.0 - occlusion;
}

void cluster_get_item_range(uint p_offset, out uint item_min, out uint item_max, out uint item_from, out uint item_to) {
	uint item_min_max = cluster_buffer.data[p_offset];
	item_min = item_min_max & 0xFFFF;
	item_max = item_min_max >> 16;

	item_from = item_min >> 5;
	item_to = (item_max == 0) ? 0 : ((item_max - 1) >> 5) + 1; //side effect of how it is stored, as item_max 0 means no elements
}

uint cluster_get_range_clip_mask(uint i, uint z_min, uint z_max) {
	int local_min = clamp(int(z_min) - int(i) * 32, 0, 31);
	int mask_width = min(int(z_max) - int(z_min), 32 - local_min);
	return bitfieldInsert(uint(0), uint(0xFFFFFFFF), local_min, mask_width);
}

void main() {
    ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(ssC, params.screen_size))) {
		return;
	}

    vec2 uv = (ssC + 0.5) / vec2(params.screen_size);
    uvec2 screen_pos = uvec2(ssC * params.screen_size);
    vec3 position = get_position(uv, scene_data.inv_projection);

    uvec2 cluster_pos = screen_pos >> params.cluster_shift;
    uint cluster_offset = (params.cluster_width * cluster_pos.y + cluster_pos.x) * (params.max_cluster_element_count_div_32 + 32);
    uint cluster_z = uint(clamp((-position.z / scene_data.z_far) * 32.0, 0.0, 31.0));
    float result = 0.0;

    { // Directional lights
        for (uint i = 0; i < MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS; i++) {
            if (i >= params.directional_light_count) {
                break;
            }

            DirectionalLightData directional_light = directional_lights.data[i];
            float step_length = directional_light.contact_shadow_length / SAMPLES;
            result += ScreenSpaceShadows(ssC, position, directional_light.direction, step_length);
        }
    }

    { // Omni lights
        uint cluster_omni_offset = cluster_offset;

        uint item_min;
        uint item_max;
        uint item_from;
        uint item_to;

        cluster_get_item_range(cluster_omni_offset + params.max_cluster_element_count_div_32 + cluster_z, item_min, item_max, item_from, item_to);

        for (uint i = item_from; i < item_to; i++) {
            uint mask = cluster_buffer.data[cluster_omni_offset + i];
			mask &= cluster_get_range_clip_mask(i, item_min, item_max);
			uint merged_mask = mask;

            while (merged_mask != 0) {
                uint bit = findMSB(merged_mask);
				merged_mask &= ~(1 << bit);

				uint light_index = 32 * i + bit;

                LightData omni_light = omni_lights.data[light_index];
                float step_length = omni_light.contact_shadow_length / SAMPLES;
                result += ScreenSpaceShadows(ssC, position, omni_light.position, step_length);
            }
        }
    }

    imageStore(shadow_image, ssC, vec4(result));
}
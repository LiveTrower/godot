// Copyright 2023 Sony Interactive Entertainment.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// If you have feedback, or found this code useful, we'd love to hear from you.
// https://www.bendstudio.com
// https://www.twitter.com/bendstudio
// 
// We are *always* looking for talented graphics and technical programmers!
// https://www.bendstudio.com/careers


// Common screen space shadow projection code (GPU):
//--------------------------------------------------------------

// The main shadow generation function is WriteScreenSpaceShadow(), it will read a depth texture, and write to a shadow texture
// This code is setup to target DX12 DXC shader compiler, but has also been tested on PS5 with appropriate API remapping. 
// It can compile to DX11, but requires some modifications (e.g., early-out's use of wave intrinsics is not supported in DX11).
// Note; you can customize the 'EarlyOutPixel' function to perform custom early-out logic to optimize this shader.

// The following Macros must be defined in the compute shader file before including this header:
//		
//		
//		#define WAVE_SIZE 64						// Wavefront size of the compute shader running this code. 
//													// numthreads[WAVE_SIZE, 1, 1]
//													// Only tested with 64.
//		
//		#define SAMPLE_COUNT 60						// Number of shadow samples per-pixel.
//													// Determines overall cost, as this value controls the length of the shadow (in pixels).
//													// The number of texture-reads performed per-thread will be (SAMPLE_COUNT / WAVE_SIZE + 2) * 2.
//													// Recommended starting value is 60 (This would be 4 reads per thread if WAVE_SIZE is 64). A value of 64 would require 6 reads.
//		
//		// Not all shadow samples are treated the same:
//		//	The bulk of samples will average together in to groups of 4, to produce a slightly smoothed result (so one sample cannot fully show the pixel)
//		//	However, the samples very close to the start pixel can optionally be forced to disable this averaging, so a single sample can fully shadow the pixel (HardShadowSamples)
//		//	Plus, a number of the last (most distant) samples can (for a small cost) apply a fade-out effect to soften a hash shadow cutoff (FadeOutSamples)
//		
//		#define HARD_SHADOW_SAMPLES 4				// Number of initial shadow samples that will produce a hard shadow, and not perform sample-averaging.
//													// This trades aliasing for grounding pixels very close to the shadow caster.
//													// Recommended starting value: 4
//		
//		#define FADE_OUT_SAMPLES 8					// Number of samples that will fade out at the end of the shadow (for a minor cost).
//													// Recommended starting value: 8

#version 450

#include "../light_data_inc.glsl"
#include "bend_sss_inc.glsl"

// Required extensions for subgroup operations (wavefront equivalent)
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

#define WAVE_SIZE           64
#define SAMPLE_COUNT        256 // shadow samples per-pixel, determines overall cost, default: 60
#define HARD_SHADOW_SAMPLES 0   // initial shadow samples that will produce a hard shadow, and not perform sample-averaging, defauult: 4
#define FADE_OUT_SAMPLES    64  // samples that will fade out at the end of the shadow (for a minor cost), default : 8

// Compute shader layout
layout(local_size_x = WAVE_SIZE, local_size_y = 1, local_size_z = 1) in;

// GLSL equivalent configurations
#define USE_HALF_PIXEL_OFFSET 1
#define USE_UV_PIXEL_BIAS 1

// Textures and samplers
layout(set = 0, binding = 0) uniform sampler2D source_depth;
layout(r16f, set = 0, binding = 1) uniform restrict writeonly image2DArray shadow_image;

layout(set = 0, binding = 2, std140) uniform SceneData {
    mat4 projection;
} scene_data;

layout(set = 0, binding = 3, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

layout(push_constant, std430) uniform Params {
    ivec2 screen_size;
    float far_depth_value;
    float near_depth_value;
    vec2 invsource_depth_size;
} params;

// Visual configuration
const float SurfaceThickness = 0.005f;
const float BilinearThreshold = 0.02f;
const float ShadowContrast = 4;
const bool IgnoreEdgePixels = false;
const bool UsePrecisionOffset = false;
const bool BilinearSamplingOffsetMode = false;

// Debug views
const bool DebugOutputEdgeMask = false;
const bool DebugOutputThreadIndex = false;
const bool DebugOutputWaveIndex = false;

// Culling / Early out
const vec2 DepthBounds = vec2(0, 1);
const bool UseEarlyOut = false;

bool EarlyOutPixel(ivec2 pixel_xy, float depth) {
    //OPTIONAL TODO; customize this function to return true if the pixel should early-out for custom reasons. E.g., A shadow map pass already found the pixel was in shadow / backfaced, etc.
	// Recommended to keep this code very simple!

	// Example:
	// return inParameters.CustomShadowMapTerm[pixel_xy] == 0;

	//(void)pixel_xy;	//unused by this implementation, avoid potential compiler warning.

	// The compiled code will be more optimal if the 'depth' value is not referenced.
    return depth >= DepthBounds.y || depth <= DepthBounds.x;
}


// Gets the start pixel coordinates for the pixels in the wavefront
// Also returns the delta to get to the next pixel after WAVE_COUNT pixels along the ray
void ComputeWavefrontExtents(ivec3 inGroupID, uint inGroupThreadID,
                            vec4 light_coordinate, ivec2 wave_offset,
                            out vec2 outDeltaXY, out vec2 outPixelXY, 
                            out float outPixelDistance, out bool outMajorAxisX) {
    
    ivec2 xy = inGroupID.yz * WAVE_SIZE + wave_offset.xy;
    
    // Integer light position / fractional component
    vec2 light_xy = floor(light_coordinate.xy) + 0.5;
    vec2 light_xy_fraction = light_coordinate.xy - light_xy;
    bool reverse_direction = light_coordinate.w > 0.0;
    
    ivec2 sign_xy = ivec2(sign(vec2(xy)));
    bool horizontal = abs(xy.x + sign_xy.y) < abs(xy.y - sign_xy.x);
    
    ivec2 axis;
    axis.x = horizontal ? (+sign_xy.y) : (0);
    axis.y = horizontal ? (0) : (-sign_xy.x);
    
    // Apply wave offset
    xy = axis * int(inGroupID.x) + xy;
    vec2 xy_f = vec2(xy);
    
    // For interpolation to the light center, we only really care about the larger of the two axis
    bool x_axis_major = abs(xy_f.x) > abs(xy_f.y);
    float major_axis = x_axis_major ? xy_f.x : xy_f.y;
    
    float major_axis_start = abs(major_axis);
    float major_axis_end = abs(major_axis) - float(WAVE_SIZE);
    
    float ma_light_frac = x_axis_major ? light_xy_fraction.x : light_xy_fraction.y;
    ma_light_frac = major_axis > 0 ? -ma_light_frac : ma_light_frac;
    
    // back in to screen direction
    vec2 start_xy = xy_f + light_xy;
    
    // For the very inner most ring, we need to interpolate to a pixel centered UV, so the UV->pixel rounding doesn't skip output pixels
    vec2 end_xy = mix(light_coordinate.xy, start_xy, 
                     (major_axis_end + ma_light_frac) / (major_axis_start + ma_light_frac));
    
    // The major axis should be a round number
    vec2 xy_delta = (start_xy - end_xy);
    
    // Inverse the read order when reverse direction is true
    float thread_step = float(inGroupThreadID ^ (reverse_direction ? 0u : (WAVE_SIZE - 1u)));
    
    vec2 pixel_xy = mix(start_xy, end_xy, thread_step / float(WAVE_SIZE));
    float pixel_distance = major_axis_start - thread_step + ma_light_frac;
    
    outPixelXY = pixel_xy;
    outPixelDistance = pixel_distance;
    outDeltaXY = xy_delta;
    outMajorAxisX = x_axis_major;
}

// Number of reads per thread
#define READ_COUNT (SAMPLE_COUNT / WAVE_SIZE + 2)

// Shared memory (equivalent to groupshared)
shared float DepthData[READ_COUNT * WAVE_SIZE];
shared bool LdsEarlyOut;

void WriteScreenSpaceShadow(ivec3 inGroupID, int inGroupThreadID, vec4 light_coordinate, out float result, out vec2 write_xy) {
    vec2 xy_delta;
    vec2 pixel_xy;
    float pixel_distance;
    bool x_axis_major; // major axis is x axis? abs(xy_delta.x) > abs(xy_delta.y).
    
    ComputeWavefrontExtents(inGroupID, inGroupThreadID, xy_delta, pixel_xy, pixel_distance, x_axis_major);
    
    // Read in the depth values
    float sampling_depth[READ_COUNT];
    float shadowing_depth[READ_COUNT];
    float depth_thickness_scale[READ_COUNT];
    float sample_distance[READ_COUNT];
    
    const float direction = -light_coordinate.w;
    const float z_sign = params.near_depth_value > params.far_depth_value ? -1.0 : +1.0;
    
    int i;
    bool is_edge = false;
    bool skip_pixel = false;
    write_xy = floor(pixel_xy);
    
    for (i = 0; i < READ_COUNT; i++) {
        // We sample depth twice per pixel per sample, and interpolate with an edge detect filter
		// Interpolation should only occur on the minor axis of the ray - major axis coordinates should be at pixel centers
        vec2 read_xy = floor(pixel_xy);
        float minor_axis = x_axis_major ? pixel_xy.y : pixel_xy.x;
        
        // If a pixel has been detected as an edge, then optionally (inParameters.IgnoreEdgePixels) don't include it in the shadow
        const float edge_skip = 1e20; // if edge skipping is enabled, apply an extreme value/blend on edge samples to push the value out of range
        
        vec2 depths;
        float bilinear = fract(minor_axis) - 0.5;

#if USE_HALF_PIXEL_OFFSET
        read_xy += 0.5;
#endif

#if USE_UV_PIXEL_BIAS
        float bias = bilinear > 0 ? 1.0 : -1.0;
        vec2 offset_xy = vec2(x_axis_major ? 0.0 : bias, x_axis_major ? bias : 0.0);
        
        // HLSL enforces that a pixel offset is a compile-time constant, which isn't strictly required (and can sometimes be a bit faster)
		// So this fallback will use a manual uv offset instead
        // Return first component on the texture compared to original, due to texture format change
        depths.x = textureLod(source_depth, read_xy * params.invsource_depth_size, 0.0).r;
        depths.y = textureLod(source_depth, (read_xy + offset_xy) * params.invsource_depth_size, 0.0).r;
#else
        int bias = bilinear > 0 ? 1 : -1;
        ivec2 offset_xy = ivec2(x_axis_major ? 0 : bias, x_axis_major ? bias : 0);
        // Return first component on the texture compared to original, due to texture format change
        depths.x = textureLod(source_depth, read_xy * params.invsource_depth_size, 0.0).r;
        depths.y = textureLodOffset(source_depth, read_xy * params.invsource_depth_size, offset_xy, 0.0).r;
#endif
        
        // Depth thresholds (bilinear/shadow thickness) are based on a fractional ratio of the difference between sampled depth and the far clip depth
        depth_thickness_scale[i] = abs(params.far_depth_value - depths.x);
        
        // If depth variance is more than a specific threshold, then just use point filtering
        bool use_point_filter = abs(depths.x - depths.y) > depth_thickness_scale[i] * BilinearThreshold;
        
        // Store for debug output when inParameters.DebugOutputEdgeMask is true
        if (i == 0) is_edge = use_point_filter;
        
        if (BilinearSamplingOffsetMode) {
            bilinear = use_point_filter ? 0.0 : bilinear;
            //both shadow depth and starting depth are the same in this mode, unless shadow skipping edges
            sampling_depth[i] = mix(depths.x, depths.y, abs(bilinear));
            shadowing_depth[i] = (IgnoreEdgePixels && use_point_filter) ? edge_skip : sampling_depth[i];
        } else {
            // The pixel starts sampling at this depth
            sampling_depth[i] = depths.x;
            
            float edge_depth = IgnoreEdgePixels ? edge_skip : depths.x;
            // Any sample in this wavefront is possibly interpolated towards the bilinear sample
			// So use should use a shadowing depth that is further away, based on the difference between the two samples
            float shadow_depth = depths.x + abs(depths.x - depths.y) * z_sign;
            
            // Shadows cast from this depth
            shadowing_depth[i] = use_point_filter ? edge_depth : shadow_depth;
        }
        
        // Store for later
        sample_distance[i] = pixel_distance + (WAVE_SIZE * i) * direction;

        // Iterate to the next pixel along the ray. This will be WAVE_SIZE pixels along the ray...
        pixel_xy += xy_delta * direction;
    }
    
    // Using early out, and no debug mode is enabled?
    if (UseEarlyOut && !DebugOutputWaveIndex && !DebugOutputThreadIndex && !DebugOutputEdgeMask) {
        // read the depth of the pixel we are shadowing, and early-out
		// The compiler will typically rearrange this code to put it directly after the first depth read
        skip_pixel = EarlyOutPixel(ivec2(write_xy), sampling_depth[0]);
        
        // are all threads in this wave out of bounds?
        bool early_out = !subgroupAny(!skip_pixel);
        
        // WaveGetLaneCount returns the hardware wave size
        if (gl_SubgroupSize == WAVE_SIZE) {
            // Optimal case:
			// If each wavefront is just a single wave, then we can trivially early-out.
            if (early_out) return;
        } else {
            // This wavefront is made up of multiple small waves, so we need to coordinate them for all to early-out together.
			// Doing this can make the worst case (all pixels drawn) a bit more expensive (~15%), but the best-case (all early-out) is typically 2-3x better.
            LdsEarlyOut = true;

            groupMemoryBarrier();
            barrier();
            
            if (!early_out)
                LdsEarlyOut = false;
            
            groupMemoryBarrier();
            barrier();
            
            if (LdsEarlyOut) return;
        }
    }
    
    // Write the shadow depths to LDS
    for (i = 0; i < READ_COUNT; i++) {
        // Perspective correct the shadowing depth, in this space, all light rays are parallel
        float stored_depth = (shadowing_depth[i] - light_coordinate.z) / sample_distance[i];
        
        if (i != 0) {
            // For pixels within sampling distance of the light, it is possible that sampling will
			// overshoot the light coordinate for extended reads. We want to ignore these samples
            stored_depth = sample_distance[i] > 0 ? stored_depth : 1e10;
        }
        
        // Store the depth values in groupshared
        int idx = (i * WAVE_SIZE) + int(inGroupThreadID);
        DepthData[idx] = stored_depth;
    }
    
    // Sync wavefronts now groupshared DepthData is written
    groupMemoryBarrier();
    barrier();
    
    // If the starting depth isn't in depth bounds, then we don't need a shadow
    if (skip_pixel) return;
    
    float start_depth = sampling_depth[0];
    
    // mix away from far depth by a tiny fraction?
    if (UsePrecisionOffset)
        start_depth = mix(start_depth, params.far_depth_value, -1.0 / 65535.0);
    
    // perspective correct the depth
    start_depth = (start_depth - light_coordinate.z) / sample_distance[0];
    
    // Start by reading the next value
    int sample_index = int(inGroupThreadID) + 1;
    
    vec4 shadow_value = vec4(1.0);
    float hard_shadow = 1.0;
    
    // This is the inverse of how large the shadowing window is for the projected sample data. 
	// All values in the LDS sample list are scaled by 1.0 / sample_distance, such that all light directions become parallel.
	// The multiply by sample_distance[0] here is to compensate for the projection divide in the data.
	// The 1.0 / inParameters.SurfaceThickness is to adjust user selected thickness. So a 0.5% thickness will scale depth values from [0,1] to [0,200]. The shadow window is always 1 wide.
	// 1.0 / depth_thickness_scale[0] is because SurfaceThickness is percentage of remaining depth between the sample and the far clip - not a percentage of the full depth range.
	// The min() function is to make sure the window is a minimum width when very close to the light. The +direction term will bias the result so the pixel at the very center of the light is either fully lit or shadowed
    float depth_scale = min(sample_distance[0] + direction, 1.0 / SurfaceThickness) * 
                       sample_distance[0] / depth_thickness_scale[0];
    
    start_depth = start_depth * depth_scale - z_sign;
    
    // Hard shadow samples
    for (i = 0; i < HARD_SHADOW_SAMPLES; i++) {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);

        // We want to find the distance of the sample that is closest to the reference depth
        hard_shadow = min(hard_shadow, depth_delta);
    }
    
    // Brute force go!
	// The main shadow samples, averaged in to a set of 4 shadow values
    for (i = HARD_SHADOW_SAMPLES; i < SAMPLE_COUNT - FADE_OUT_SAMPLES; i++) {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);

        // Do the same as the hard_shadow code above, but this will accumulate to 4 separate values.
		// By using 4 values, the average shadow can be taken, which can help soften single-pixel shadows.
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta);
    }
    
    // Final fade out samples
    for (i = SAMPLE_COUNT - FADE_OUT_SAMPLES; i < SAMPLE_COUNT; i++) {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        
        // Add the fade value to these samples
        const float fade_out = float(i + 1 - (SAMPLE_COUNT - FADE_OUT_SAMPLES)) / 
                              float(FADE_OUT_SAMPLES + 1) * 0.75;
        
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta + fade_out);
    }
    
    // Apply the contrast value.
	// A value of 0 indicates a sample was exactly matched to the reference depth (and the result is fully shadowed)
	// We want some boost to this range, so samples don't have to exactly match to produce a full shadow. 
    shadow_value = clamp(shadow_value * ShadowContrast + (1.0 - ShadowContrast), 0.0, 1.0);
    hard_shadow = clamp(hard_shadow * ShadowContrast + (1.0 - ShadowContrast), 0.0, 1.0);
    
    // Take the average of 4 samples, this is useful to reduces aliasing noise in the source depth, especially with long shadows.
    result = dot(shadow_value, vec4(0.25));

    // If the first samples are always producing a hard shadow, then compute this value separately.
    result += min(hard_shadow, result);
}

void main() {
    float result;
    vec2 write_xy;
    vec4 light_projection;
    ivec2 wave_offset;

    for (uint i = 0; i < MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS; i++) {
        if (i >= params.directional_light_count) {
            break;
        }

        light_projection = scene_data.projection * vec4(directional_lights.data[i].direction, 0.0);
        DataList data_list = BuildData(light_projection, params.screen_size, ivec2(0, 0), params.screen_size);

        for (int i = 0; i < data_list.DataCount; i++) {
            WaveData data = data_list.wave[i]; 
            WriteScreenSpaceShadow(gl_WorkGroupID, gl_LocalInvocationID.x, data.LightCoordinate_Shader, result, write_xy);
        }
    }

    //write the result
    if (DebugOutputEdgeMask)
        result = is_edge ? 1.0 : 0.0;
    if (DebugOutputThreadIndex)
        result = float(gl_LocalInvocationID.x) / float(WAVE_SIZE);
    if (DebugOutputWaveIndex)
        result = fract(float(inGroupID.x) / float(WAVE_SIZE));
    
    // Asking the GPU to write scattered single-byte pixels isn't great,
	// But thankfully the latency is hidden by all the work we're doing...
    imageStore(shadow_image, ivec2(write_xy), vec4(result));
}

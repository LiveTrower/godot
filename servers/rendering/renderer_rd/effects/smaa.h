/**************************************************************************/
/*  smaa.h                                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef SMAA_RD_H
#define SMAA_RD_H

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/smaa_blit.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/smaa_blend.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/smaa_edge_detection.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/smaa_weight.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_scene_render.h"

#include "servers/rendering_server.h"

namespace RendererRD {

class SMAA {
public:
	SMAA();
	~SMAA();

    struct SMAASettings{
        enum SMAAQuality{
            SMAA_LOW,
            SMAA_MEDIUM,
            SMAA_HIGH,
            SMAA_ULTRA,
            SMAA_MAX
        };

        enum SMAAEdgeDetectionMode{
            SMAA_EDGE_DETECTION_LUMA,
            SMAA_EDGE_DETECTION_COLOR,
            SMAA_EDGE_DETECTION_DEPTH,
            SMAA_EDGE_DETECTION_MAX
        };
    };

    void smaa(RID p_source_color, RID p_depth, RID p_dst_framebuffer, Size2 p_resolution, const SMAASettings &p_settings);

private:
    float threshold = 0.1;
    uint32_t max_search_steps = 8;
    bool disable_diag_detection = true;
    uint32_t max_search_steps_diag = 0;
    bool disable_corner_detection = true;
    uint32_t corner_rounding = 0;

	struct SMAAPushConstant {
        float texel_width;
        float texel_height;
        float resolution_width;
		float resolution_height;
	};

    SMAAPushConstant push_constant;

    struct SMAAEdge {
        SmaaEdgeDetectionShaderRD edge_detection_shader_rd;
        RID shader_version;
        RID edge_pipeline;
    } smaa_edge;

    struct SMAAWeight {
        SmaaWeightShaderRD weight_shader_rd;
        RID shader_version;
        RID weight_pipeline;
    } smaa_weight;

    struct SMAABlend {
        SmaaBlendShaderRD blend_shader_rd;
        RID shader_version;
        RID blend_pipeline;
    } smaa_blend;
    
    struct SMAABlit{
        SmaaBlitShaderRD blit_shader_rd;
        RID shader_version;
        RID blit_pipeline;
    } smaa_blit;
};

} // namespace RendererRD

#endif // SMAA_RD_H

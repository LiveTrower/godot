/**************************************************************************/
/*  smaa.cpp                                                              */
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

#include "smaa.h"
#include "servers/rendering/renderer_rd/texture_data/area_tex.h"
#include "servers/rendering/renderer_rd/texture_data/search_tex.h"

#include "core/config/project_settings.h"
#include "servers/rendering/renderer_rd/effects/copy_effects.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

SMAA::SMAA(){
    //RD::VertexAttribute va;
    //va.stride = 16;
    //va.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;

    RD::PipelineColorBlendState color_blend = RD::PipelineColorBlendState::create_blend();

    // Edge detection shader's specialization constant
    RD::PipelineSpecializationConstant threshold_constant;
    threshold_constant.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT;
    threshold_constant.constant_id = 0;
    threshold_constant.float_value = threshold;

    // Weight calculation shader's specialization constants
    RD::PipelineSpecializationConstant max_search_constant;
    max_search_constant.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
    max_search_constant.constant_id = 0;
    max_search_constant.int_value = max_search_steps;
    RD::PipelineSpecializationConstant disable_diag_constant;
    disable_diag_constant.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
    disable_diag_constant.constant_id = 1;
    disable_diag_constant.bool_value = disable_diag_detection;
    RD::PipelineSpecializationConstant max_search_diag_constant;
    max_search_diag_constant.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
    max_search_diag_constant.constant_id = 2;
    max_search_diag_constant.int_value = max_search_steps_diag;
    RD::PipelineSpecializationConstant disable_corner_constant;
    disable_corner_constant.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
    disable_corner_constant.constant_id = 3;
    disable_corner_constant.bool_value = disable_corner_detection;
    RD::PipelineSpecializationConstant corner_rounding_constant;
    corner_rounding_constant.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
    corner_rounding_constant.constant_id = 4;
    corner_rounding_constant.int_value = corner_rounding;

    /*RD::AttachmentFormat attachment_format;
    attachment_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
    attachment_format.usage_flags = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
    RD::FramebufferFormatID framebuffer_format = RD::get_singleton()->framebuffer_format_create({attachment_format});
    attachment_format.format = RD::DATA_FORMAT_R16G16_SFLOAT;
    RD::FramebufferFormatID rg_framebuffer_format = RD::get_singleton()->framebuffer_format_create({attachment_format});*/

    RD::TextureFormat tformat;
	tformat.format = RD::DATA_FORMAT_BC5_UNORM_BLOCK;
	tformat.width = 160;
	tformat.height = 560;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
	area_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), {area_tex});

    tformat.format = RD::DATA_FORMAT_R8_UNORM;
	tformat.width = 64;
	tformat.height = 16;
    search_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), {search_tex});

    //edge
    {
        Vector<String> smaa_modes;
        smaa_modes.push_back("\n#define SMAA_EDGE_LUMA");

        smaa_edge.edge_detection_shader_rd.initialize(smaa_modes);

        smaa_edge.shader_version = smaa_edge.edge_detection_shader_rd.version_create();
        //smaa_edge.edge_pipeline = RD::get_singleton()->render_pipeline_create(smaa_edge.edge_detection_shader_rd.version_get_shader(smaa_edge.shader_version, 0), rg_framebuffer_format, RD::get_singleton()->vertex_format_create({va}), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend, 0, 0, {threshold_constant});
        smaa_edge.edge_pipeline.setup(smaa_edge.edge_detection_shader_rd.version_get_shader(smaa_edge.shader_version, 0), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend, 0, {threshold_constant});
    }

    //weight
    {
        Vector<String> smaa_modes;
        smaa_modes.push_back("");

        smaa_weight.weight_shader_rd.initialize(smaa_modes);

        smaa_weight.shader_version = smaa_weight.weight_shader_rd.version_create();
        //smaa_weight.weight_pipeline = RD::get_singleton()->render_pipeline_create(smaa_weight.weight_shader_rd.version_get_shader(smaa_weight.shader_version, 0), framebuffer_format, RD::get_singleton()->vertex_format_create({va}), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend, 0, 0, {max_search_constant, disable_diag_constant, max_search_diag_constant, disable_corner_constant, corner_rounding_constant});
        smaa_weight.weight_pipeline.setup(smaa_weight.weight_shader_rd.version_get_shader(smaa_weight.shader_version, 0), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend, 0, {max_search_constant, disable_diag_constant, max_search_diag_constant, disable_corner_constant, corner_rounding_constant});
    }

    //blend
    {
        Vector<String> smaa_modes;
        smaa_modes.push_back("");

        smaa_blend.blend_shader_rd.initialize(smaa_modes);

        smaa_blend.shader_version = smaa_blend.blend_shader_rd.version_create();
        //smaa_blend.blend_pipeline = RD::get_singleton()->render_pipeline_create(smaa_blend.blend_shader_rd.version_get_shader(smaa_blend.shader_version, 0), framebuffer_format, RD::get_singleton()->vertex_format_create({va}), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend);
        smaa_blend.blend_pipeline.setup(smaa_blend.blend_shader_rd.version_get_shader(smaa_blend.shader_version, 0), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend);
    }

    //blit
    {
        Vector<String> smaa_modes;
        smaa_modes.push_back("");

        smaa_blit.blit_shader_rd.initialize(smaa_modes);

        smaa_blit.shader_version = smaa_blit.blit_shader_rd.version_create();
        //smaa_blit.blit_pipeline = RD::get_singleton()->render_pipeline_create(smaa_blit.blit_shader_rd.version_get_shader(smaa_blit.shader_version, 0), framebuffer_format, RD::get_singleton()->vertex_format_create({va}), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend);
        smaa_blit.blit_pipeline.setup(smaa_blit.blit_shader_rd.version_get_shader(smaa_blit.shader_version, 0), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), color_blend);
    }
}

SMAA::~SMAA(){
    smaa_blend.blend_shader_rd.version_free(smaa_blend.shader_version);
    smaa_weight.weight_shader_rd.version_free(smaa_weight.shader_version);
    smaa_edge.edge_detection_shader_rd.version_free(smaa_edge.shader_version);
    smaa_blit.blit_shader_rd.version_free(smaa_blit.shader_version);
    RD::get_singleton()->free(area_texture);
    RD::get_singleton()->free(search_texture);
}

void SMAA::smaa(RID p_source_color, RID p_depth, RID p_dst_framebuffer, Size2 p_resolution, const SMAASettings &p_settings){
    UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
    MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

    RID edge_shader = smaa_edge.edge_detection_shader_rd.version_get_shader(smaa_edge.shader_version, 0);
    ERR_FAIL_COND(edge_shader.is_null());

    RID weight_shader = smaa_weight.weight_shader_rd.version_get_shader(smaa_weight.shader_version, 0);
    ERR_FAIL_COND(weight_shader.is_null());

    RID blend_shader = smaa_blend.blend_shader_rd.version_get_shader(smaa_blend.shader_version, 0);
    ERR_FAIL_COND(blend_shader.is_null());

    RID blit_shader = smaa_blit.blit_shader_rd.version_get_shader(smaa_blit.shader_version, 0);
    ERR_FAIL_COND(blit_shader.is_null());

    RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

    memset(&push_constant, 0, sizeof(SMAAPushConstant));
    push_constant.texel_height = 1 / p_resolution.height;
    push_constant.texel_width = 1 / p_resolution.width;
    push_constant.resolution_height = p_resolution.height;
    push_constant.resolution_width = p_resolution.width;

    /*Vector<uint8_t> verts = {
        0, 0, 128, 191, 0, 0, 128, 191, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 128, 191, 0, 0, 64, 64, 0, 0, 0, 
        0, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 191, 
        0, 0, 0, 64, 0, 0, 0, 0
    };*/

    //RD::VertexAttribute va;
    //va.stride = 16;
    //va.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;

    RD::SamplerState sampler_state;
    sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
    sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
    RID linear_sampler = RD::get_singleton()->sampler_create(sampler_state);

    RD::TextureFormat tformat;
    tformat.format = RD::DATA_FORMAT_R16G16_SFLOAT;
	tformat.width = p_resolution.width;
	tformat.height = p_resolution.height;
	tformat.usage_bits = (RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT);
    RID edges_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView());
    tformat.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
    RID blend_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView());
    RID copy_texture = RD::get_singleton()->texture_create(tformat, RD::TextureView());

    //RID edges_framebuffer = RD::get_singleton()->framebuffer_create({edges_texture});
    //RID blend_framebuffer = RD::get_singleton()->framebuffer_create({blend_texture});
    //RID copy_framebuffer = RD::get_singleton()->framebuffer_create({copy_texture});

    //RID vertex_buffer = RD::get_singleton()->vertex_buffer_create(verts.size(), verts, true);
    //RID vertex_array = RD::get_singleton()->vertex_array_create(3, RD::get_singleton()->vertex_format_create({va}), {vertex_buffer}, {});

    RD::Uniform u_source_color(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_color }));
    RD::Uniform u_depth(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, { default_sampler, p_depth });
    RD::Uniform u_blend_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, {default_sampler, blend_texture });

    RD::get_singleton()->draw_command_begin_label("SMAA");

    // First Pass: Edge Detection
    RD::get_singleton()->draw_command_begin_label("SMAA Edge Detection");
    RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_STORE, RD::INITIAL_ACTION_DISCARD, RD::FINAL_ACTION_DISCARD, Vector({Color(0.0, 0.0, 0.0, 0.0)}));
    RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, smaa_edge.edge_pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass()));
    RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(edge_shader, 0, u_source_color, u_depth), 0);
    RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(SMAAPushConstant));
    //RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array);
    RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
    RD::get_singleton()->draw_command_end_label();

    // Second Pass: Blending weight calculation
    RD::get_singleton()->draw_command_begin_label("SMAA Blending Weight Calculation");
    RD::Uniform u_edges_texture;
    u_edges_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    u_edges_texture.binding = 0;
    u_edges_texture.append_id(linear_sampler);
    u_edges_texture.append_id(edges_texture);

    RD::Uniform u_area_texture;
    u_area_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    u_area_texture.binding = 1;
    u_area_texture.append_id(linear_sampler);
    u_area_texture.append_id(area_texture);

    RD::Uniform u_search_texture;
    u_search_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    u_search_texture.binding = 2;
    u_search_texture.append_id(linear_sampler);
    u_search_texture.append_id(search_texture);

    draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_STORE, RD::INITIAL_ACTION_DISCARD, RD::FINAL_ACTION_DISCARD, Vector({Color(0.0, 0.0, 0.0, 0.0)}));
    RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, smaa_weight.weight_pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass()));
    RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(weight_shader, 0, u_edges_texture, u_area_texture, u_search_texture), 0);
    RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(SMAAPushConstant));
    //RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array);
    RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
    RD::get_singleton()->draw_command_end_label();

    // Copy source image to copy buffer for input in 3rd pass
    RD::get_singleton()->draw_command_begin_label("SMAA Copy Source Image");
    draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_DISCARD, RD::FINAL_ACTION_STORE, RD::INITIAL_ACTION_DISCARD, RD::FINAL_ACTION_DISCARD);
    RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, smaa_blit.blit_pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass()));
    RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(blit_shader, 0, u_source_color), 0);
    //RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array);
    RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
    RD::get_singleton()->draw_command_end_label();

    // Third Pass: Neighborhood Blending
    RD::get_singleton()->draw_command_begin_label("SMAA Neighborhood Blending");
    RD::Uniform u_color_texture;
    u_color_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    u_color_texture.binding = 0;
    u_color_texture.append_id(linear_sampler);
    u_color_texture.append_id(copy_texture);

    draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_DISCARD, RD::FINAL_ACTION_STORE, RD::INITIAL_ACTION_DISCARD, RD::FINAL_ACTION_DISCARD);
    RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, smaa_blend.blend_pipeline.get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer), false, RD::get_singleton()->draw_list_get_current_pass()));
    RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(blend_shader, 0, u_color_texture, u_blend_texture), 0);
    RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(SMAAPushConstant));
    //RD::get_singleton()->draw_list_bind_vertex_array(draw_list, vertex_array);
    RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();
    RD::get_singleton()->draw_command_end_label();

    RD::get_singleton()->draw_command_end_label();
}
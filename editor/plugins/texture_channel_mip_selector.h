/**************************************************************************/
/*  color_channel_selector.h                                              */
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

#pragma once

#include "scene/gui/box_container.h"

class PanelContainer;
class Button;
class SpinBox;

class TextureChannelMipSelector : public HBoxContainer {
	GDCLASS(TextureChannelMipSelector, HBoxContainer);

	static const unsigned int CHANNEL_COUNT = 4;

public:
	TextureChannelMipSelector();

	void set_available_channels_mask(uint32_t p_mask);
	void set_available_mip_levels(int p_lod);
	void set_channel_buttons_container_visibility(bool p_visible);
	void set_mip_level_container_visibility(bool p_visible);
	double get_selected_mip_level() const;
	uint32_t get_selected_channels_mask() const;
	Vector4 get_selected_channel_factors() const;

private:
	void _notification(int p_what);

	void create_button(unsigned int p_channel_index, const String &p_text);
	void create_mip_level_selector();
	void on_toggled(bool p_pressed);
	void on_channel_button_toggled(bool p_unused_pressed);
	void on_mip_level_value_changed(double p_value);

	static void _bind_methods();

	Button *channel_buttons[CHANNEL_COUNT] = {};
	HBoxContainer *channel_buttons_container = nullptr;
	HBoxContainer *mip_level_container = nullptr;
	SpinBox *spin_box = nullptr;
	PanelContainer *panel = nullptr;
	Button *toggle_button = nullptr;
};

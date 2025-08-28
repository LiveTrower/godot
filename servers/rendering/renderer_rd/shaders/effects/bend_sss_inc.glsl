// Generating a screen-space-shadow requires a number of Compute Shader dispatches
// The compute shader reads from a depth buffer, and writes a single-channel texture of the same dimensions
// Each dispatch is of the same compute shader, (see bend_sss_gpu.h).
// The number of dispatches required varies based on the on-screen location of the light.
// Typically there will be just one or two dispatches when the light is off-screen, and 4 to 6 when the light is on-screen.
// Syncing the GPU between individual dispatches is not required

// These structures and function are used to generate the number of dispatches, the wave count of each dispatch (X/Y/Z) and shader parameters for each dispatch

struct WaveData {
	ivec3 WaveCount;
	ivec2 WaveOffset_Shader;
};

struct DataList {
	vec4 LightCoordinate_Shader;

	WaveData wave[8];
	int DataCount;
};

// Helper functions
int bend_min(const int a, const int b) {
	return a > b ? b : a;
}
int bend_max(const int a, const int b) {
	return a > b ? a : b;
}

DataList BuildData(vec4 inLightProjection, ivec2 inViewportSize, ivec2 inMinRenderBounds, ivec2 inMaxRenderBounds, bool inExpandedZRange = false, int inWaveSize = 64) {
	DataList result = {};

	// Floating point division in the shader has a practical limit for precision when the light is *very* far off screen (~1m pixels+)
	// So when computing the light XY coordinate, use an adjusted w value to handle these extreme values
	float xy_light_w = inLightProjection.w;
	float FP_limit = 0.000002f * float(inWaveSize);

	if (xy_light_w >= 0 && xy_light_w < FP_limit) {
		xy_light_w = FP_limit;
	} else if (xy_light_w < 0 && xy_light_w > -FP_limit) {
		xy_light_w = -FP_limit;
	}

	// Need precise XY pixel coordinates of the light
	result.lightCoordinateShader.x = ((inLightProjection.x / xy_light_w) * 0.5 + 0.5) * float(inViewportSize.x);
	result.lightCoordinateShader.y = ((inLightProjection.y / xy_light_w) * -0.5 + 0.5) * float(inViewportSize.y);
	result.lightCoordinateShader.z = inLightProjection.w == 0.0 ? 0.0 : (inLightProjection.z / inLightProjection.w);
	// Use proper single precision float to avoid warning that the original code has
	result.lightCoordinateShader.w = inLightProjection.w > 0.0 ? 1.0 : -1.0;

	if (inExpandedZRange) {
		result.lightCoordinateShader.z = result.lightCoordinateShader.z * 0.5 + 0.5;
	}

	ivec2 light_xy = ivec2(
			int(result.lightCoordinateShader.x + 0.5),
			int(result.lightCoordinateShader.y + 0.5));

	// Make the bounds inclusive, relative to the light
	ivec4 biased_bounds = ivec4(
			inMinRenderBounds.x - light_xy.x,
			-(inMaxRenderBounds.y - light_xy.y),
			inMaxRenderBounds.x - light_xy.x,
			-(inMinRenderBounds.y - light_xy.y));

	// Process 4 quadrants around the light center,
	// They each form a rectangle with one corner on the light XY coordinate
	// If the rectangle isn't square, it will need breaking in two on the larger axis
	// 0 = bottom left, 1 = bottom right, 2 = top left, 2 = top right
	for (int q = 0; q < 4; q++) {
		// Quads 0 and 3 needs to be +1 vertically, 1 and 2 need to be +1 horizontally
		bool vertical = q == 0 || q == 3;

		// Bounds relative to the quadrant
		ivec4 bounds = ivec4(
				bend_max(0, ((q & 1) != 0 ? biased_bounds.x : -biased_bounds.z) / inWaveSize),
				bend_max(0, ((q & 2) != 0 ? biased_bounds.y : -biased_bounds.w) / inWaveSize),
				bend_max(0, (((q & 1) != 0 ? biased_bounds.z : -biased_bounds.x) + inWaveSize * (vertical ? 1 : 2) - 1) / inWaveSize),
				bend_max(0, (((q & 2) != 0 ? biased_bounds.w : -biased_bounds.y) + inWaveSize * (vertical ? 2 : 1) - 1) / inWaveSize));

		if ((bounds.z - bounds.x) > 0 && (bounds.w - bounds.y) > 0) {
			int bias_x = (q == 2 || q == 3) ? 1 : 0;
			int bias_y = (q == 1 || q == 3) ? 1 : 0;

			DispatchData &disp = result.wave[result.DataCount++];

			disp.WaveCount.x = inWaveSize;
			disp.WaveCount.y = bounds.z - bounds.x;
			disp.WaveCount.z = bounds.w - bounds.y;
			disp.WaveOffset_Shader.x = ((q & 1) ? bounds.x : -bounds.z) + bias_x;
			disp.WaveOffset_Shader.y = ((q & 2) ? -bounds.w : bounds.y) + bias_y;

			// We want the far corner of this quadrant relative to the light,
			// as we need to know where the diagonal light ray intersects with the edge of the bounds
			int axis_delta = biased_bounds.x - biased_bounds.y;
			if (q == 1) {
				axis_delta = biased_bounds.z + biased_bounds.y;
			}
			if (q == 2) {
				axis_delta = -biased_bounds.x - biased_bounds.w;
			}
			if (q == 3) {
				axis_delta = -biased_bounds.z + biased_bounds.w;
			}

			axis_delta = (axis_delta + inWaveSize - 1) / inWaveSize;

			if (axis_delta > 0) {
				DispatchData &disp2 = result.wave[result.DataCount++];

				// Take copy of current volume
				disp2 = disp;

				if (q == 0) {
					// Split on Y, split becomes -1 larger on x
					disp2.waveCount.z = bend_min(disp.waveCount.z, axis_delta);
					disp.waveCount.z -= disp2.waveCount.z;
					disp2.waveOffsetShader.y = disp.waveOffsetShader.y + disp.waveCount.z;
					disp2.waveOffsetShader.x--;
					disp2.waveCount.y++;
				}
				if (q == 1) {
					// Split on X, split becomes +1 larger on y
					disp2.waveCount.y = bend_min(disp.waveCount.y, axis_delta);
					disp.waveCount.y -= disp2.waveCount.y;
					disp2.waveOffsetShader.x = disp.waveOffsetShader.x + disp.waveCount.y;
					disp2.waveCount.z++;
				}
				if (q == 2) {
					// Split on X, split becomes -1 larger on y
					disp2.waveCount.y = bend_min(disp.waveCount.y, axis_delta);
					disp.waveCount.y -= disp2.waveCount.y;
					disp.waveOffsetShader.x += disp2.waveCount.y;
					disp2.waveCount.z++;
					disp2.waveOffsetShader.y--;
				}
				if (q == 3) {
					// Split on Y, split becomes +1 larger on x
					disp2.waveCount.z = bend_min(disp.waveCount.z, axis_delta);
					disp.waveCount.z -= disp2.waveCount.z;
					disp.waveOffsetShader.y += disp2.waveCount.z;
					disp2.waveCount.y++;
				}

				// Remove if too small
				if (disp2.waveCount.y > 0 && disp2.waveCount.z > 0) {
					disp2 = result.wave[--result.DataCount];
				}
				if (disp.waveCount.y <= 0 || disp.waveCount.z <= 0) {
					disp = result.wave[--result.DataCount];
				}
			}
		}
	}

	// Scale the shader values by the wave count, the shader expects this
	for (int i = 0; i < result.DataCount; i++) {
		result.wave[i].waveOffsetShader.x *= inWaveSize;
		result.wave[i].waveOffsetShader.y *= inWaveSize;
	}

	return result;
}

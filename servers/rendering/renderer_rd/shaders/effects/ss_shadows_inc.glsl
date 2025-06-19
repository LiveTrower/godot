#include "../cluster_data_inc.glsl"
#include "../light_data_inc.glsl"

// Interleaved Gradient Noise
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float quick_hash(vec2 pos, float taa_frame_count) {
	pos += taa_frame_count * 4.7526;
    pos += taa_frame_count * 3.1914;
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

vec2 view_to_uv(vec3 x, mat4 projection_matrix)
{
    vec4 uv = projection_matrix * vec4(x, 1.0);
    return (uv.xy / uv.w) * vec2(0.5f, -0.5f) + 0.5f;
}

float get_linear_depth(vec2 uv, float near, float far) {
    float depth = textureLod(source_depth, uv, 0.0).r * 2.0 - 1.0;
    return 2.0 * far * near / (near + far - depth * (near - far));
}

float screen_fade(vec2 uv)
{
    vec2 fade = max(vec2(0.0f), 12.0f * abs(uv - 0.5f) - 5.0f);
    return clamp(1.0f - dot(fade, fade), 0.0, 1.0);
}

vec3 get_position(vec2 uv, mat4 inv_projection_matrix)
{
	float depth = textureLod(source_depth, uv, 0.0).r;
	uv.y = 1.0 - uv.y;
	vec4 pos_clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
	vec4 view_position = inv_projection_matrix * pos_clip;
	return view_position.xyz / view_position.w;
}

float smoothbumpstep(float edge0, float edge1, float x)
{
	x = 1.0 - abs(clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0) - .5) * 2.0;
	return x * x * (3.0 - x - x);
}

bool is_valid_uv(vec2 value) { return (value.x >= 0.0f && value.x <= 1.0f) && (value.y >= 0.0f && value.y <= 1.0f); }
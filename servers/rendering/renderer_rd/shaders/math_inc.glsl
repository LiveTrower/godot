#define MEDIUMP_FLT_MAX    65504.0
#define saturateMediump(x) min(x, MEDIUMP_FLT_MAX)
#define saturate(x) clamp(x, 0, 1)
#define SQRT_TWO_PI 2.50663
#define ETA 1.55

/* [Drobot2014a] Low Level Optimizations for GCN. */
float sqrt_IEEE_int_approximation(float v) {
	return intBitsToFloat(0x1fbd1df5 + (floatBitsToInt(v) >> 1));
}

vec2 sqrt_IEEE_int_approximation(vec2 v) {
	return intBitsToFloat(0x1fbd1df5 + (floatBitsToInt(v) >> 1));
}

float pow2(float x) {
	return x * x;
}

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float max3(const vec3 v) {
    return max(v.x, max(v.y, v.z));
}

vec3 rcp(vec3 x) {
    return 1.0 / x;
}

float rcp(float x) {
    return 1.0 / x;
}

float luminance( vec3 linear_color )
{
	return dot( linear_color, vec3( 0.3, 0.59, 0.11 ) );
}
/* [Drobot2014a] Low Level Optimizations for GCN. */
float sqrt_IEEE_int_approximation(float v) {
	return intBitsToFloat(0x1fbd1df5 + (floatBitsToInt(v) >> 1));
}

vec2 sqrt_IEEE_int_approximation(vec2 v) {
	return intBitsToFloat(0x1fbd1df5 + (floatBitsToInt(v) >> 1));
}

half pow5(half x) {
    half x2 = x * x;
    return x2 * x2 * x;
}

half max3(const hvec3 v) {
    return max(v.x, max(v.y, v.z));
}

hvec3 rcp(hvec3 x) {
    return hvec3(1.0 / x);
}

half rcp(half x) {
    return half(1.0 / x);
}
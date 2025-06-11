#define saturate(x) clamp(x, half(0.0), half(1.0))

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
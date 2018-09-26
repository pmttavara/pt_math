/* pt_math.h - public domain branchless scalar math routines */
#ifndef PT_MATH_H
#define PT_MATH_H
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long PT__ull;
typedef long long PT__ll;
#define PT__FPCOPY64(dst, src)                                                 \
    do {                                                                       \
        int PT__i;                                                             \
        for (PT__i = 0; PT__i < 8; PT__i++) {                                  \
            ((char *)&(dst))[PT__i] = ((char *)&(src))[PT__i];                 \
        }                                                                      \
    } while (0)
#define PT__FPCOPY32(dst, src)                                                 \
    do {                                                                       \
        int PT__i;                                                             \
        for (PT__i = 0; PT__i < 4; PT__i++) {                                  \
            ((char *)&(dst))[PT__i] = ((char *)&(src))[PT__i];                 \
        }                                                                      \
    } while (0)

static int PT_abs(int x) {
    return (x < 0 ? -x : x);
}
static long PT_labs(long x) {
    return (x < 0 ? -x : x);
}
static long long PT_llabs(long long x) {
    return (x < 0 ? -x : x);
}
static double PT_fabs(double x) {
    return (x < 0 ? -x : x);
}
static float PT_fabsf(float x) {
    return (x < 0 ? -x : x);
}
#if defined(_MSC_VER) || defined(__GNUC__)
#include <emmintrin.h>
static double PT_sqrt(double x) {
    __m128d m;
    m = _mm_set_sd(x);
    m = _mm_sqrt_pd(m);
    x = _mm_cvtsd_f64(m);
    return x;
}
static float PT_sqrtf(float x) {
    __m128 m;
    m = _mm_set_ss(x);
    m = _mm_sqrt_ps(m);
    x = _mm_cvtss_f32(m);
    return x;
}
static double PT_rsqrt(double x) {
    return 1 / PT_sqrt(x);
}
static float PT_rsqrtf(float x) {
    __m128 m;
    m = _mm_set_ss(x);
    m = _mm_rsqrt_ps(m);
    x = _mm_cvtss_f32(m);
    return x;
}
#else
static double PT_sqrt(double x) {
    PT__ll y;
    double z;
    PT__FPCOPY64(y, x);
    y = ((y - 0x0010000000000000ll) >> 1) + 0x2000000000000000ll;
    PT__FPCOPY64(z, y);
    z = (x / z + z) * 0.5;
    return (x / z + z) * 0.5;
}
static float PT_sqrtf(float x) {
    int y;
    float z;
    PT__FPCOPY32(y, x);
    y = ((y - 0x00800000) >> 1) + 0x20000000;
    PT__FPCOPY32(z, y);
    z = (x / z + z) * 0.5f;
    return (x / z + z) * 0.5f;
}
static double PT_rsqrt(double x) {
    PT__ll y;
    double z;
    z = x * 0.5;
    PT__FPCOPY64(y, x);
    y = 0x5fe6eb50c7b537a9 - (y >> 1);
    PT__FPCOPY64(x, y);
    x *= 1.5 - z * x * x;
    return x * (1.5 - z * x * x);
}
static float PT_rsqrtf(float x) {
    int y;
    double z;
    z = x * 0.5f;
    PT__FPCOPY32(y, x);
    y = 0x5f375a86 - (y >> 1);
    PT__FPCOPY32(x, y);
    x *= 1.5f - z * x * x;
    return x * (1.5f - z * x * x);
}
#endif

static double PT_round(double x) {
    x += 6755399441055744.0;
    x -= 6755399441055744.0;
    return x;
}
static float PT_roundf(float x) {
    x += 12582912.0f;
    x -= 12582912.0f;
    return x;
}
static double PT_floor(double x) {
    return PT_round(x - 0.5);
}
static float PT_floorf(float x) {
    return PT_roundf(x - 0.5f);
}
static double PT_ceil(double x) {
    return PT_round(x + 0.5);
}
static float PT_ceilf(float x) {
    return PT_roundf(x + 0.5f);
}
static double PT_remainder(double x, double y) {
    return x - PT_floor(x / y) * y;
}
static float PT_remainderf(float x, float y) {
    return x - PT_floorf(x / y) * y;
}
static double PT_fmod(double x, double y) {
    return x - (PT__ll)(x / y) * y;
}
static float PT_fmodf(float x, float y) {
    return x - (PT__ll)(x / y) * y;
}
static double PT_trunc(double x) {
    return (double)(PT__ll)x;
}
static float PT_truncf(float x) {
    return (float)(PT__ll)x;
}
static double PT_sin(double x) {
    double y;
    x *= -0.31830988618379067;
    y = x + 13510798882111488.0;
    x -= y - 13510798882111488.0;
    x *= (x < 0 ? -x : x) - 1;
    return x * (3.5841304553896 * (x < 0 ? -x : x) + 3.1039673861526);
}
static float PT_sinf(float x) {
    float y;
    x *= -0.318309886f;
    y = x + 25165824.0f;
    x -= y - 25165824.0f;
    x *= (x < 0 ? -x : x) - 1;
    return x * (3.5841304553896f * (x < 0 ? -x : x) + 3.1039673861526f);
}
static double PT_cos(double x) {
    return PT_sin(x + 1.57079632679489662);
}
static float PT_cosf(float x) {
    return PT_sinf(x + 1.570796327f);
}
static double PT_tan(double x) {
    return PT_sin(x) / PT_cos(x);
}
static float PT_tanf(float x) {
    return PT_sinf(x) / PT_cosf(x);
}
static double PT_asin(double x) {
    x = PT_sqrt(1 + x) - PT_sqrt(1 - x);
    return x * (0.131754508171 * (x < 0 ? -x : x) + 0.924391722181);
}
static float PT_asinf(float x) {
    x = PT_sqrtf(1 + x) - PT_sqrtf(1 - x);
    return x * (0.131754508171f * (x < 0 ? -x : x) + 0.924391722181f);
}
static double PT_acos(double x) {
    return 1.57079632679489662 - PT_asin(x);
}
static float PT_acosf(float x) {
    return 1.570796327f - PT_asinf(x);
}
static double PT_atan(double x) {
    x /= (x < 0 ? -x : x) + 1;
    return x * ((x < 0 ? -x : x) *
                    (-1.45667498914 * (x < 0 ? -x : x) + 2.18501248371) +
                0.842458832225);
}
static float PT_atanf(float x) {
    x /= (x < 0 ? -x : x) + 1;
    return x * ((x < 0 ? -x : x) *
                    (-1.45667498914f * (x < 0 ? -x : x) + 2.18501248371f) +
                0.842458832225f);
}
static double PT_atan2(double y, double x) {
    double t;
    t = 3.1415926535897932 - (y < 0) * 6.28318530717958648;
    y = PT_atan(y / (x + !x)) + (x < 0) * t;
    return y + !x * (t * 0.5 - y);
}
static float PT_atan2f(float y, float x) {
    float t;
    t = 3.141592653f - (y < 0) * 6.283185307f;
    y = PT_atanf(y / (x + !x)) + (x < 0) * t;
    return y + !x * (t * 0.5f - y);
}
static double PT_exp2(double x) {
    double y;
    PT__ll exponent;
    exponent = (PT__ll)(x + 1023);
    x += 1023 - exponent;
    exponent <<= 52;
    PT__FPCOPY64(y, exponent);
    x *= x * 0.339766027260413688582 + 0.660233972739586311418;
    return (x + 1) * y;
}
static float PT_exp2f(float x) {
    float y;
    int exponent;
    exponent = (int)(x + 127);
    x += 127 - exponent;
    exponent <<= 23;
    PT__FPCOPY32(y, exponent);
    x *= x * 0.339766027f + 0.660233972f;
    return (x + 1) * y;
}
static double PT_log2(double x) {
    PT__ll y;
    double result;
    PT__FPCOPY64(y, x);
    y >>= 52;
    result = (double)y;
    PT__FPCOPY64(y, x);
    y = y & 0x000fffffffffffff | 0x3ff0000000000000;
    PT__FPCOPY64(x, y);
    result = result - 1024 + x * (x * -0.33333333333333333 + 2) -
             0.66666666666666666;
    return result;
}
static float PT_log2f(float x) {
    int y;
    float result;
    PT__FPCOPY32(y, x);
    y >>= 23;
    result = (float)y;
    PT__FPCOPY32(y, x);
    y = y & 0x007fffff | 0x3f800000;
    PT__FPCOPY32(x, y);
    result = result - 128 + x * (x * -0.333333333f + 2) - 0.666666666f;
    return result;
}
static double PT_exp(double x) {
    return PT_exp2(x * 1.44269504088896341);
}
static float PT_expf(float x) {
    return PT_exp2f(x * 1.442695041f);
}
static double PT_exp10(double x) {
    return PT_exp2(x * 3.321928094887362348);
}
static float PT_exp10f(float x) {
    return PT_exp2f(x * 3.32192809f);
}
static double PT_log(double x) {
    return PT_log2(x) * 0.6931471805599453094;
}
static float PT_logf(float x) {
    return PT_log2f(x) * 0.693147181f;
}
static double PT_log10(double x) {
    return PT_log2(x) * 0.3010299956639811952;
}
static float PT_log10f(float x) {
    return PT_log2f(x) * 0.301029996f;
}
static double PT_pow(double a, double b) {
    return PT_exp2(b * PT_log2(a));
}
static float PT_powf(float a, float b) {
    return PT_exp2f(b * PT_log2f(a));
}
static double PT_sinh(double x) {
    x = PT_exp(x);
    return (x - 1 / x) * 0.5;
}
static float PT_sinhf(float x) {
    x = PT_expf(x);
    return (x - 1 / x) * 0.5f;
}
static double PT_cosh(double x) {
    x = PT_exp(x);
    return (x + 1 / x) * 0.5;
}
static float PT_coshf(float x) {
    x = PT_expf(x);
    return (x + 1 / x) * 0.5f;
}
static double PT_tanh(double x) {
    x = PT_exp(x * -2);
    return (1 - x) / (1 + x);
}
static float PT_tanhf(float x) {
    x = PT_expf(x * -2);
    return (1 - x) / (1 + x);
}
static double PT_asinh(double x) {
    return PT_log(x + PT_sqrt(x * x + 1));
}
static float PT_asinhf(float x) {
    return PT_logf(x + PT_sqrtf(x * x + 1));
}
static double PT_acosh(double x) {
    return PT_log(x + PT_sqrt(x * x - 1));
}
static float PT_acoshf(float x) {
    return PT_logf(x + PT_sqrtf(x * x - 1));
}
static double PT_atanh(double x) {
    return PT_log((1 + x) / (1 - x)) * 0.5;
}
static float PT_atanhf(float x) {
    return PT_logf((1 + x) / (1 - x)) * 0.5f;
}
static double PT_erf(double x) {
    x = PT_exp2(x * 3.47203417614113462733);
    return x / ((x < 0 ? -x : x) + 1) * 2 - 1;
}
static float PT_erff(float x) {
    x = PT_exp2f(x * 3.472034176f);
    return x / ((x < 0 ? -x : x) + 1) * 2 - 1;
}
static double PT_erfc(double x) {
    return 1 - PT_erf(x);
}
static float PT_erfcf(float x) {
    return 1 - PT_erff(x);
}
#define PT_TAU 6.28318530717958648
#define PT_E 2.71828182845904524
#define PT_INFINITY ((float)(1e300 * 1e300))
#define PT_NAN ((float)(PT_INFINITY * 0))
#define PT_TAUf ((float)PT_TAU)
#define PT_Ef ((float)PT_E)
#define PT_ABS(x) ((x) < 0 ? -(x) : (x))
#define PT_FP_ZERO 0
#define PT_FP_SUBNORMAL 1
#define PT_FP_NORMAL 2
#define PT_FP_INFINITE 3
#define PT_FP_NAN 4
#define PT__CLASS(f) (sizeof(f) == 8 ? PT__dclass(f) : PT__fclass(f))
#define PT_fpclassify(f) ((int)(PT__CLASS(f) & 7))
#define PT_signbit(f) (!!(PT__CLASS(f) & 0x80000000u))
#define PT_isfinite(f) (PT_fpclassify(f) < PT_FP_INFINITE)
#define PT_isinf(f) (PT_fpclassify(f) == PT_FP_INFINITE)
#define PT_isnan(f) (PT_fpclassify(f) == PT_FP_NAN)
#define PT_isnormal(f) (PT_fpclassify(f) == PT_FP_NORMAL)
static unsigned int PT__dclass(double x) {
    PT__ull u;
    PT__FPCOPY64(u, x);
    unsigned int signbit = (u >> 32) & 0x80000000;
    u &= 0x7fffffffffffffff;
    if (u == 0) {
        return signbit | PT_FP_ZERO;
    }
    if (u < 0x0010000000000000) {
        return signbit | PT_FP_SUBNORMAL;
    }
    if (u < 0x7ff0000000000000) {
        return signbit | PT_FP_NORMAL;
    }
    if (u == 0x7ff0000000000000) {
        return signbit | PT_FP_INFINITE;
    }
    return signbit | PT_FP_NAN;
}
static unsigned int PT__fclass(float x) {
    unsigned int u;
    PT__FPCOPY32(u, x);
    unsigned int signbit = u & 0x80000000;
    u &= 0x7fffffff;
    if (u == 0) {
        return signbit | PT_FP_ZERO;
    }
    if (u < 0x00800000) {
        return signbit | PT_FP_SUBNORMAL;
    }
    if (u < 0x7f800000) {
        return signbit | PT_FP_NORMAL;
    }
    if (u == 0x7f800000) {
        return signbit | PT_FP_INFINITE;
    }
    return signbit | PT_FP_NAN;
}
#define PT_FNS(S, D, N)                                                        \
    S(abs, int)                                                                \
    S(labs, long)                                                              \
    S(llabs, long long)                                                        \
    S(fabs, double)                                                            \
    S(fabsf, float)                                                            \
    S(sqrt, double)                                                            \
    S(sqrtf, float)                                                            \
    N(rsqrt, double)                                                           \
    N(rsqrtf, float)                                                           \
    S(round, double)                                                           \
    S(roundf, float)                                                           \
    S(floor, double)                                                           \
    S(floorf, float)                                                           \
    S(ceil, double)                                                            \
    S(ceilf, float)                                                            \
    D(remainder, double)                                                       \
    D(remainderf, float)                                                       \
    D(fmod, double)                                                            \
    D(fmodf, float)                                                            \
    S(trunc, double)                                                           \
    S(truncf, float)                                                           \
    S(sin, double)                                                             \
    S(sinf, float)                                                             \
    S(cos, double)                                                             \
    S(cosf, float)                                                             \
    S(tan, double)                                                             \
    S(tanf, float)                                                             \
    S(asin, double)                                                            \
    S(asinf, float)                                                            \
    S(acos, double)                                                            \
    S(acosf, float)                                                            \
    S(atan, double)                                                            \
    S(atanf, float)                                                            \
    D(atan2, double)                                                           \
    D(atan2f, float)                                                           \
    S(exp2, double)                                                            \
    S(exp2f, float)                                                            \
    S(log2, double)                                                            \
    S(log2f, float)                                                            \
    S(exp, double)                                                             \
    S(expf, float)                                                             \
    N(exp10, double)                                                           \
    N(exp10f, float)                                                           \
    S(log, double)                                                             \
    S(logf, float)                                                             \
    S(log10, double)                                                           \
    S(log10f, float)                                                           \
    D(pow, double)                                                             \
    D(powf, float)                                                             \
    S(sinh, double)                                                            \
    S(sinhf, float)                                                            \
    S(cosh, double)                                                            \
    S(coshf, float)                                                            \
    S(tanh, double)                                                            \
    S(tanhf, float)                                                            \
    S(asinh, double)                                                           \
    S(asinhf, float)                                                           \
    S(acosh, double)                                                           \
    S(acoshf, float)                                                           \
    S(atanh, double)                                                           \
    S(atanhf, float)                                                           \
    S(erf, double)                                                             \
    S(erff, float)                                                             \
    S(erfc, double)                                                            \
    S(erfcf, float)

#ifdef __cplusplus
}
#endif
#endif /* PT_MATH_H */

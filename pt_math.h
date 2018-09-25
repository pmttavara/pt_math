/* pt_math.h - public domain branchless scalar math routines */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PT_ /* Customize the decorator. */           
#define PT_(name) PT_##name
#endif

#ifndef PT_MATH_H
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
static unsigned int PT__dclass(double x);
static unsigned int PT__fclass(float x);

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
#endif /* PT_MATH_H */

static const double PT_(tau) = 6.28318530717958648;
static const double PT_(e) = 2.71828182845904523536;
static const float PT_(infinity) = (float)(1e300 * 1e300);
static const float PT_(nan) = (float)((1e300 * 1e300)) * 0;
static int(PT_(abs))(int x) {
    return (x < 0 ? -x : x);
}
static long(PT_(labs))(long x) {
    return (x < 0 ? -x : x);
}
static long long(PT_(llabs))(long long x) {
    return (x < 0 ? -x : x);
}
static double(PT_(fabs))(double x) {
    return (x < 0 ? -x : x);
}
static float(PT_(fabsf))(float x) {
    return (x < 0 ? -x : x);
}
#if defined(_MSC_VER) || defined(__GNUC__)
#include <emmintrin.h>
static double(PT_(sqrt))(double x) {
    __m128d m;
    m = _mm_set_sd(x);
    m = _mm_sqrt_pd(m);
    x = _mm_cvtsd_f64(m);
    return x;
}
static float(PT_(sqrtf))(float x) {
    __m128 m;
    m = _mm_set_ss(x);
    m = _mm_sqrt_ps(m);
    x = _mm_cvtss_f32(m);
    return x;
}
static double(PT_(rsqrt))(double x) {
    return 1 / PT_(sqrt)(x);
}
static float(PT_(rsqrtf))(float x) {
    __m128 m;
    m = _mm_set_ss(x);          
    m = _mm_rsqrt_ps(m);
    x = _mm_cvtss_f32(m);
    return x;
}
#else
static double(PT_(sqrt))(double x) {
    PT__ll y;
    double z;
    PT__FPCOPY64(y, x);
    y -= 0x0010000000000000ll;
    y >>= 1;
    y += 0x2000000000000000ll;
    PT__FPCOPY64(z, y);
    z = (x / z + z) * 0.5;
    z = (x / z + z) * 0.5;
    return z;
}
static float(PT_(sqrtf))(float x) {
    int y;
    float z;
    PT__FPCOPY32(y, x);
    y -= 0x00800000;
    y >>= 1;
    y += 0x20000000;
    PT__FPCOPY32(z, y);
    z = (x / z + z) * 0.5f;
    z = (x / z + z) * 0.5f;
    return z;
}
static double(PT_(rsqrt))(double x) {
    PT__ll y;
    double z;
    z = x * 0.5;
    PT__FPCOPY64(y, x);
    y = 0x5fe6eb50c7b537a9 - (y >> 1);
    PT__FPCOPY64(x, y);
    x *= 1.5 - z * x * x;
    x *= 1.5 - z * x * x;
    return x;
}
static float(PT_(rsqrtf))(float x) {
    int y;
    double z;
    z = x * 0.5f;
    PT__FPCOPY32(y, x);
    y = 0x5f375a86 - (y >> 1);
    PT__FPCOPY32(x, y);
    x *= 1.5f - z * x * x;
    x *= 1.5f - z * x * x;
    return x;
}
#endif

static double(PT_(round))(double x) {
    x += 6755399441055744.0;
    x -= 6755399441055744.0;
    return x;
}
static float(PT_(roundf))(float x) {
    x += 12582912.0f;
    x -= 12582912.0f;
    return x;
}
static double(PT_(fmod))(double x, double y) {
    return x - (PT__ll)(x / y) * y;
}
static float(PT_(fmodf))(float x, float y) {
    return x - (PT__ll)(x / y) * y;
}
static double(PT_(trunc))(double x) {
    return (double)(PT__ll)x;
}
static float(PT_(truncf))(float x) {
    return (float)(PT__ll)x;
}
static double(PT_(sin))(double x) {
    double y;
    x *= -0.31830988618379067;
    y = x + 13510798882111488.0;
    x -= y - 13510798882111488.0;
    x *= (x < 0 ? -x : x) - 1;
    return x * (3.5841304553896 * (x < 0 ? -x : x) + 3.1039673861526);
}
static float(PT_(sinf))(float x) {
    float y;
    x *= -0.318309886f;
    y = x + 25165824.0f;
    x -= y - 25165824.0f;
    x *= (x < 0 ? -x : x) - 1;
    return x * (3.5841304553896f * (x < 0 ? -x : x) + 3.1039673861526f);
}
static double(PT_(cos))(double x) {
    x += 1.57079632679489662;
    return PT_(sin)(x);
}
static float(PT_(cosf))(float x) {
    x += 1.570796327f;
    return PT_(sinf)(x);
}
static double(PT_(asin))(double x) {
    double y;
    y = 1 + x;
    x = 1 - x;
    x = PT_(sqrt)(y) - PT_(sqrt)(x);
    x *= (0.131754508171 * (x < 0 ? -x : x) + 0.924391722181);
    return x;
}
static float(PT_(asinf))(float x) {
    float y;
    y = 1 + x;
    x = 1 - x;
    x = PT_(sqrtf)(y) - PT_(sqrtf)(x);
    x *= (0.131754508171f * (x < 0 ? -x : x) + 0.924391722181f);
    return x;
}
static double(PT_(acos))(double x) {
    return 1.570796327f - PT_(asin)(x);
}
static float(PT_(acosf))(float x) {
    return 1.570796327f - PT_(asinf)(x);
}
static double(PT_(atan))(double x) {
    x /= (x < 0 ? -x : x) + 1;
    return x * ((x < 0 ? -x : x) *
                    (-1.45667498914 * (x < 0 ? -x : x) + 2.18501248371) +
                0.842458832225);
}
static float(PT_(atanf))(float x) {
    x /= (x < 0 ? -x : x) + 1;
    return x * ((x < 0 ? -x : x) *
                    (-1.45667498914f * (x < 0 ? -x : x) + 2.18501248371f) +
                0.842458832225f);
}
static double(PT_(exp2))(double x) {
    double y, z;
    PT__ll exponent;
    exponent = (PT__ll)(x + 1023);
    x += 1023 - exponent;
    exponent <<= 52;
    PT__FPCOPY64(y, exponent);
    z = x;
    z *= 0.339766027260413688582;
    z += 0.660233972739586311418;
    x *= z;
    x += 1;
    x *= y;
    return x;
}
static float(PT_(exp2f))(float x) {
    float y, z;
    int exponent;
    exponent = (int)(x + 127);
    x += 127 - exponent;
    exponent <<= 23;
    PT__FPCOPY32(y, exponent);
    z = x;
    z *= 0.339766027f;
    z += 0.660233972f;
    x *= z;
    x += 1;
    x *= y;
    return x;
}
static double(PT_(log2))(double x) {
    PT__ll y;
    double result, z;
    PT__FPCOPY64(y, x);
    y >>= 52;
    result = (double)y;
    PT__FPCOPY64(y, x);
    y &= 0x000fffffffffffff;
    y |= 0x3ff0000000000000;
    PT__FPCOPY64(x, y);
    z = x;
    x *= -0.33333333333333333;
    x += 2;
    z *= x;
    result -= 1024 + 0.66666666666666666;
    result += z;
    return result;
}
static float(PT_(log2f))(float x) {
    int y;
    float result, z;
    PT__FPCOPY32(y, x);
    y >>= 23;
    result = (float)y;
    PT__FPCOPY32(y, x);
    y &= 0x007fffff;
    y |= 0x3f800000;
    PT__FPCOPY32(x, y);
    z = x;
    x *= -0.333333333f;
    x += 2;
    z *= x;
    result -= 128 + 0.666666666f;
    result += z;
    return result;
}
static double(PT_(floor))(double x) {
    x -= 0.5;
    return PT_(round)(x);
}
static float(PT_(floorf))(float x) {
    x -= 0.5f;
    return PT_(roundf)(x);
}
static double(PT_(ceil))(double x) {
    x += 0.5;
    return PT_(round)(x);
}
static float(PT_(ceilf))(float x) {
    x += 0.5f;
    return PT_(roundf)(x);
}
static double(PT_(remainder))(double x, double y) {
    double z;
    z = x / y;
    return x - PT_(floor)(z) * y;
}
static float(PT_(remainderf))(float x, float y) {
    float z;
    z = x / y;
    return x - PT_(floorf)(z) * y;
}
static double(PT_(tan))(double x) {
    return PT_(sin)(x) / PT_(cos)(x);
}
static float(PT_(tanf))(float x) {
    return PT_(sinf)(x) / PT_(cosf)(x);
}
static double(PT_(exp))(double x) {
    x *= 1.44269504088896341;
    return PT_(exp2)(x);
}
static float(PT_(expf))(float x) {
    x *= 1.442695041f;
    return PT_(exp2f)(x);
}
static double(PT_(exp10))(double x) {
    x *= 3.321928094887362348;
    return PT_(exp2)(x);
}
static float(PT_(exp10f))(float x) {
    x *= 3.32192809f;
    return PT_(exp2f)(x);
}
static double(PT_(log))(double x) {
    return PT_(log2)(x) * 0.6931471805599453094;
}
static float(PT_(logf))(float x) {
    return PT_(log2f)(x) * 0.693147181f;
}
static double(PT_(log10))(double x) {
    return PT_(log2)(x) * 0.3010299956639811952;
}
static float(PT_(log10f))(float x) {
    return PT_(log2f)(x) * 0.301029996f;
}
static double(PT_(pow))(double a, double b) {
    b *= PT_(log2)(a);
    return PT_(exp2)(b);
}
static float(PT_(powf))(float a, float b) {
    b *= PT_(log2f)(a);
    return PT_(exp2f)(b);
}

static double(PT_(sinh))(double x) {
    x = PT_(exp)(x);
    return (x - 1 / x) * 0.5;
}
static float(PT_(sinhf))(float x) {
    x = PT_(expf)(x);
    return (x - 1 / x) * 0.5f;
}
static double(PT_(cosh))(double x) {
    x = PT_(exp)(x);
    return (x + 1 / x) * 0.5;
}
static float(PT_(coshf))(float x) {
    x = PT_(expf)((x));
    return (x + 1 / x) * 0.5f;
}
static double(PT_(tanh))(double x) {
    x *= -2;
    x = PT_(exp)(x);
    return (1 - x) / (1 + x);
}
static float(PT_(tanhf))(float x) {
    x *= -2;
    x = PT_(expf)(x);
    return (1 - x) / (1 + x);
}
static double(PT_(asinh))(double x) {
    double y;
    y = x * x + 1;
    x += PT_(sqrt)(y);
    return PT_(log)(x);
}
static float(PT_(asinhf))(float x) {
    float y;
    y = x * x + 1;
    x += PT_(sqrtf)(y);
    return PT_(logf)(x);
}
static double(PT_(acosh))(double x) {
    double y;
    y = x * x - 1;
    x += PT_(sqrt)(y);
    return PT_(log)(x);
}
static float(PT_(acoshf))(float x) {
    float y = x * x - 1;
    x += PT_(sqrtf)(y);
    return PT_(logf)(x);
}
static double(PT_(atanh))(double x) {
    x = (1 + x) / (1 - x);
    return PT_(log)(x) * 0.5;
}
static float(PT_(atanhf))(float x) {
    x = (1 + x) / (1 - x);
    return PT_(logf)(x) * 0.5f;
}
static double(PT_(erf))(double x) {
    x *= 3.47203417614113462733;
    x = PT_(exp2)(x);
    return x / ((x < 0 ? -x : x) + 1) * 2 - 1;
}
static float(PT_(erff))(float x) {
    x *= 3.472034176f;
    x = PT_(exp2f)(x);
    return x / ((x < 0 ? -x : x) + 1) * 2 - 1;
}
static double(PT_(erfc))(double x) {
    return 1 - PT_(erf)(x);
}
static float(PT_(erfcf))(float x) {
    return 1 - PT_(erff)(x);
}
static double(PT_(atan2))(double y, double x) {
    double t, result;
    t = 3.1415926535897932 - (y < 0) * 6.28318530717958648;
    y /= x + !x;
    result = PT_(atan)(y) + (x < 0) * t;
    return result + !x * (t * 0.5 - result);
}
static float(PT_(atan2f))(float y, float x) {
    float t, result;
    t = 3.141592653f - (y < 0) * 6.283185307f;
    y /= x + !x;
    result = PT_(atanf)(y) + (x < 0) * t;
    return result + !x * (t * 0.5f - result);
}                                      

#ifndef PT_MATH_H
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
#define PT_MATH_H
#endif /* PT_MATH_H */

#ifdef __cplusplus
}
#endif

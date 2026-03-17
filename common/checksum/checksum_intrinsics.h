/*
Copyright 2022 The Photon Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#else
#error "Unsupported architecture"
#endif

#if defined(__x86_64__)
#define CHECKSUM_SIMD_HAS_AVX512 1
#if defined(__clang__)
#define CHECKSUM_SIMD_TARGET_BEGIN \
    _Pragma("clang attribute push (__attribute__((target(\"crc32,sse4.1,pclmul\"))), apply_to=function)")
#define CHECKSUM_SIMD_TARGET_END \
    _Pragma("clang attribute pop")
#define CHECKSUM_SIMD_TARGET_AVX512_BEGIN \
    _Pragma("clang attribute push (__attribute__((target(\"crc32,sse4.1,pclmul,avx512f,avx512dq,avx512vl,vpclmulqdq\"))), apply_to=function)")
#define CHECKSUM_SIMD_TARGET_AVX512_END \
    _Pragma("clang attribute pop")
#else
#define CHECKSUM_SIMD_TARGET_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC target (\"crc32,sse4.1,pclmul\")") \
    _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"") \
    _Pragma("GCC diagnostic ignored \"-Wuninitialized\"") \
    _Pragma("GCC diagnostic ignored \"-Winit-self\"")
#define CHECKSUM_SIMD_TARGET_END \
    _Pragma("GCC pop_options")
#define CHECKSUM_SIMD_TARGET_AVX512_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC target (\"crc32,sse4.1,pclmul,avx512f,avx512dq,avx512vl,vpclmulqdq\")")
#define CHECKSUM_SIMD_TARGET_AVX512_END \
    _Pragma("GCC pop_options")
#endif
#else
#define CHECKSUM_SIMD_HAS_AVX512 0
#define CHECKSUM_SIMD_TARGET_BEGIN
#define CHECKSUM_SIMD_TARGET_END
#define CHECKSUM_SIMD_TARGET_AVX512_BEGIN
#define CHECKSUM_SIMD_TARGET_AVX512_END
#endif

#if defined(__clang__)
#define CHECKSUM_SIMD_STRICT_ALIASING_DIAG_PUSH \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wstrict-aliasing\"")
#define CHECKSUM_SIMD_STRICT_ALIASING_DIAG_POP \
    _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#define CHECKSUM_SIMD_STRICT_ALIASING_DIAG_PUSH \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wstrict-aliasing\"")
#define CHECKSUM_SIMD_STRICT_ALIASING_DIAG_POP \
    _Pragma("GCC diagnostic pop")
#else
#define CHECKSUM_SIMD_STRICT_ALIASING_DIAG_PUSH
#define CHECKSUM_SIMD_STRICT_ALIASING_DIAG_POP
#endif

namespace checksum_simd {
struct Runtime {
    static bool supports_hw_crc32c() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.2");
#elif defined(__aarch64__)
#ifdef __APPLE__
    return true;
#elif defined(__linux__)
    return getauxval(AT_HWCAP) & HWCAP_CRC32;
#else
    return false;
#endif
#else
    return false;
#endif
    }


    static bool supports_crc64_simd128() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse") &&
           __builtin_cpu_supports("pclmul");
#elif defined(__aarch64__)
#ifdef __APPLE__
    return true;
#elif defined(__linux__)
    return getauxval(AT_HWCAP) & HWCAP_PMULL;
#else
    return false;
#endif
#else
    return false;
#endif
    }

    static bool supports_crc64_avx512() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("vpclmulqdq");
#else
    return false;
#endif
    }
};

struct CRC32CIntrinsic {
    static uint32_t extend(uint32_t crc, uint8_t data) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#if defined(__clang__)
    return __builtin_arm_crc32cb(crc, data);
#else
    return __builtin_aarch64_crc32cb(crc, data);
#endif
#elif defined(__aarch64__)
    __asm__("crc32cb %w[c], %w[c], %w[v]" : [c] "+r"(crc) : [v] "r"(data));
    return crc;
#else
    return __builtin_ia32_crc32qi(crc, data);
#endif
    }

    static uint32_t extend(uint32_t crc, uint16_t data) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#if defined(__clang__)
    return __builtin_arm_crc32ch(crc, data);
#else
    return __builtin_aarch64_crc32ch(crc, data);
#endif
#elif defined(__aarch64__)
    __asm__("crc32ch %w[c], %w[c], %w[v]" : [c] "+r"(crc) : [v] "r"(data));
    return crc;
#else
    return __builtin_ia32_crc32hi(crc, data);
#endif
    }

    static uint32_t extend(uint32_t crc, uint32_t data) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#if defined(__clang__)
    return __builtin_arm_crc32cw(crc, data);
#else
    return __builtin_aarch64_crc32cw(crc, data);
#endif
#elif defined(__aarch64__)
    __asm__("crc32cw %w[c], %w[c], %w[v]" : [c] "+r"(crc) : [v] "r"(data));
    return crc;
#else
    return __builtin_ia32_crc32si(crc, data);
#endif
    }

    static uint32_t extend(uint32_t crc, uint64_t data) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#if defined(__clang__)
    return static_cast<uint32_t>(__builtin_arm_crc32cd(crc, data));
#else
    return static_cast<uint32_t>(__builtin_aarch64_crc32cx(crc, data));
#endif
#elif defined(__aarch64__)
    __asm__("crc32cx %w[c], %w[c], %x[v]" : [c] "+r"(crc) : [v] "r"(data));
    return crc;
#else
    asm volatile ("crc32q %1, %q0" : "+r"(crc) : "rm"(data));
    return crc;
#endif
    }
};

#ifdef __x86_64__
struct SIMD128 {
    using v128 = __m128i;

    static v128 loadu(const void* ptr) {
        return _mm_loadu_si128(reinterpret_cast<const v128*>(ptr));
    }

    static v128 set_low_u32(uint32_t value) {
        return _mm_setr_epi32(static_cast<int>(value), 0, 0, 0);
    }

    static v128 set_low_u64(uint64_t value) {
        return _mm_set_epi64x(0, static_cast<long long>(value));
    }

    static v128 set_u64x2(uint64_t low, uint64_t high) {
        return _mm_set_epi64x(static_cast<long long>(high),
                              static_cast<long long>(low));
    }

    static v128 shuffle_bytes(v128 x, const v128& mask) {
        return _mm_shuffle_epi8(x, mask);
    }

    static v128 blend_bytes(v128 x, v128 y, v128 mask) {
        return _mm_blendv_epi8(x, y, mask);
    }

    // Keep the CLMUL lane selector as a template parameter instead of a plain
    // int argument. Although the x86 intrinsic is spelled as
    // _mm_clmulepi64_si128(x, y, imm8), the imm8 operand is an instruction
    // immediate and must be compile-time constant in practice. Using a
    // template here also keeps the ARM implementation aligned with x86: both
    // backends can specialize directly for the selected 64-bit lane pair with
    // no runtime switch or fallback path.
    template<uint8_t imm>
    static v128 clmul(v128 x, const uint64_t* rk) {
        return _mm_clmulepi64_si128(x, loadu(rk), imm);
    }

    template<uint8_t imm>
    static v128 clmul(v128 x, v128 y) {
        return _mm_clmulepi64_si128(x, y, imm);
    }

    static v128 bsl8(v128 x) {
        return _mm_bslli_si128(x, 8);
    }

    static v128 bsr8(v128 x) {
        return _mm_bsrli_si128(x, 8);
    }

    static uint64_t low_u64(v128 x) {
        return static_cast<uint64_t>(_mm_cvtsi128_si64(x));
    }

    static uint64_t high_u64(v128 x) {
        return static_cast<uint64_t>(_mm_extract_epi64(x, 1));
    }

};

struct SIMD512 {
    using v512 = __m512i;

    static v512 loadu(const void* ptr) {
        return _mm512_loadu_si512(ptr);
    }

    static v512 set_low_u64(uint64_t value) {
        return _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0,
                                static_cast<long long>(value));
    }

    static v512 broadcast_128(SIMD128::v128 value) {
        return _mm512_broadcast_i32x4(value);
    }

    static v512 broadcast_128(const void* ptr) {
        return broadcast_128(SIMD128::loadu(ptr));
    }

    static v512 set_low_128(SIMD128::v128 value) {
        return _mm512_castsi128_si512(value);
    }

    static SIMD128::v128 upper_128(v512 x) {
        return _mm512_extracti64x2_epi64(x, 0x03);
    }

};
#elif defined(__aarch64__)
struct SIMD128 {
    using v128 = uint8x16_t;

    static v128 loadu(const void* ptr) {
        return vld1q_u8(reinterpret_cast<const uint8_t*>(ptr));
    }

    static v128 set_low_u32(uint32_t value) {
        auto lanes = vdupq_n_u32(0);
        lanes = vsetq_lane_u32(value, lanes, 0);
        return vreinterpretq_u8_u32(lanes);
    }

    static v128 set_low_u64(uint64_t value) {
        auto lanes = vdupq_n_u64(0);
        lanes = vsetq_lane_u64(value, lanes, 0);
        return vreinterpretq_u8_u64(lanes);
    }

    static v128 set_u64x2(uint64_t low, uint64_t high) {
        auto lanes = vdupq_n_u64(0);
        lanes = vsetq_lane_u64(low, lanes, 0);
        lanes = vsetq_lane_u64(high, lanes, 1);
        return vreinterpretq_u8_u64(lanes);
    }

    static v128 shuffle_bytes(v128 x, const v128& mask) {
        return vqtbl1q_u8(x, mask);
    }

    static v128 blend_bytes(v128 x, v128 y, v128 mask) {
        auto select = vcltq_s8(vreinterpretq_s8_u8(mask), vdupq_n_s8(0));
        return vbslq_u8(select, y, x);
    }

    static v128 pmull_low(v128 x, v128 y) {
        v128 result;
        __asm__("pmull %0.1q, %1.1d, %2.1d"
                : "=w"(result)
                : "w"(x), "w"(y));
        return result;
    }

    template<uint8_t imm>
    static v128 clmul(v128 x, const uint64_t* rk) {
        return clmul<imm>(x, loadu(rk));
    }

    template<uint8_t imm>
    static v128 clmul(v128 x, v128 y);

    static v128 bsl8(v128 x) {
        return vextq_u8(vdupq_n_u8(0), x, 8);
    }

    static v128 bsr8(v128 x) {
        return vextq_u8(x, vdupq_n_u8(0), 8);
    }

    static uint64_t low_u64(v128 x) {
        return vgetq_lane_u64(vreinterpretq_u64_u8(x), 0);
    }

    static uint64_t high_u64(v128 x) {
        return vgetq_lane_u64(vreinterpretq_u64_u8(x), 1);
    }

};

template<uint8_t imm>
inline SIMD128::v128 SIMD128::clmul(v128, v128) {
    static_assert(imm == 0x00 || imm == 0x01 || imm == 0x10 || imm == 0x11,
                  "unsupported clmul lane selector");
    __builtin_unreachable();
}

// AArch64 only ever sees compile-time CLMUL lane selectors in this codebase.
// Use explicit specializations instead of data-dependent bit tests so each imm
// maps directly to one fixed vextq_u8 + pmull sequence.
template<>
inline SIMD128::v128 SIMD128::clmul<0x00>(v128 x, v128 y) {
    return pmull_low(x, y);
}

template<>
inline SIMD128::v128 SIMD128::clmul<0x01>(v128 x, v128 y) {
    return pmull_low(vextq_u8(x, x, 8), y);
}

template<>
inline SIMD128::v128 SIMD128::clmul<0x10>(v128 x, v128 y) {
    return pmull_low(x, vextq_u8(y, y, 8));
}

template<>
inline SIMD128::v128 SIMD128::clmul<0x11>(v128 x, v128 y) {
    return pmull_low(vextq_u8(x, x, 8), vextq_u8(y, y, 8));
}
#endif

}  // namespace checksum_simd
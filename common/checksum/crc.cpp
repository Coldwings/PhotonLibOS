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

#include "crc32c.h"
#include "crc64ecma.h"
#include "crc_shift_tables.h"
#include "checksum_intrinsics.h"
#include <stdlib.h>
#include <photon/common/utility.h>
#include <photon/common/alog.h>

template<typename T, typename F1, typename F8> __attribute__((always_inline))
inline T do_crc(const uint8_t *data, size_t nbytes, T crc, F1 f1, F8 f8) {
    size_t offset = 0;
    // Process bytes one at a time until we reach an 8-byte boundary and can
    // start doing aligned 64-bit reads.
    static uintptr_t ALIGN_MASK = sizeof(uint64_t) - 1;
    size_t mask = (size_t)((uintptr_t)data & ALIGN_MASK);
    if (mask != 0) {
        size_t limit = std::min(nbytes, sizeof(uint64_t) - mask);
        while (offset < limit) {
            crc = f1(crc, data[offset]);
            offset++;
        }
    }

    // Process 8 bytes at a time until we have fewer than 8 bytes left.
    while (offset + sizeof(uint64_t) <= nbytes) {
        crc = f8(crc, *(uint64_t*)(data + offset));
        offset += sizeof(uint64_t);
    }
    // Process any bytes remaining after the last aligned 8-byte block.
    while (offset < nbytes) {
        crc = f1(crc, data[offset]);
        offset++;
    }
    return crc;
}

template<size_t begin, size_t end, ssize_t step, typename F,
        typename = typename std::enable_if<begin != end>::type>
inline __attribute__((always_inline))
void static_loop(const F& f) {
    f(begin);
    static_loop<begin + step, end, step>(f);
}

template<size_t begin, size_t end, ssize_t step, typename F, typename = void,
        typename = typename std::enable_if<begin == end>::type>
inline __attribute__((always_inline))
void static_loop(const F& f) {
    f(begin);
}

// gcc sometimes doesn't allow always_inline
#define BODY(i) [&](size_t i) /*__attribute__((always_inline))*/


#define CVAL(c, val) (-!!(c) & val)

const uint32_t CRC32C_POLY = 0x82f63b78;
const uint64_t CRC64ECMA_POLY = 0xc96c5795d7870f42;

template<typename T>
struct TableCRC {
    typedef T (*Table)[256];
    Table table = (Table) malloc(sizeof(*table) * 8);
    ~TableCRC() { free(table); }
    TableCRC(T POLY) {
        for (int n = 0; n < 256; n++) {
            T crc = n;
            static_loop<0, 7, 1>(BODY(k) {
                crc = CVAL(crc&1, POLY) ^ (crc >> 1);
            });
            table[0][n] = crc;
        }
        for (int n = 0; n < 256; n++) {
            T crc = table[0][n];
            static_loop<1, 7, 1>(BODY(k) {
                crc = table[0][crc & 0xff] ^ (crc >> 8);
                table[k][n] = crc;
            });
        }
    }
    __attribute__((always_inline))
    T operator()(const uint8_t *buffer, size_t nbytes, T crc) const {
        auto f1 = [&](T crc, uint8_t b) {
            return table[0][(crc ^ b) & 0xff] ^ (crc >> 8);
        };
        auto f8 = [&](T crc, uint64_t x) {
            x ^= crc; crc = 0;
            static_loop<0, 7, 1>(BODY(i) {
                crc ^= table[7-i][(x >> (i*8)) & 0xff];
            });
            return crc;
        };
        return do_crc(buffer, nbytes, crc, f1, f8);
    }
};

uint32_t crc32c_sw(const uint8_t *buffer, size_t nbytes, uint32_t crc) {
    const static TableCRC<uint32_t> calc(CRC32C_POLY);
    return calc(buffer, nbytes, crc);
}

uint64_t crc64ecma_sw(const uint8_t *buffer, size_t nbytes, uint64_t crc) {
    const static TableCRC<uint64_t> calc(CRC64ECMA_POLY);
    return ~calc(buffer, nbytes, ~crc);
}

uint64_t crc64ecma_hw_simd128(const uint8_t *buf, size_t len, uint64_t crc);
#if CHECKSUM_SIMD_HAS_AVX512
uint64_t crc64ecma_hw_avx512(const uint8_t *buf, size_t len, uint64_t crc);
#endif
uint32_t (*crc32c_auto)(const uint8_t*, size_t, uint32_t) = nullptr;
uint32_t (*crc32c_combine_auto)(uint32_t crc1, uint32_t crc2, uint32_t len2);
uint32_t (*crc32c_combine_series_auto)(uint32_t* crc, uint32_t part_size, uint32_t n_parts);
void (*crc32c_series_auto)(const uint8_t *buffer, uint32_t part_size, uint32_t n_parts, uint32_t* crc_parts);
uint64_t (*crc64ecma_auto)(const uint8_t *data, size_t nbytes, uint64_t crc);
uint64_t (*crc64ecma_combine_auto)(uint64_t crc1, uint64_t crc2, uint32_t len2);
uint64_t (*crc64ecma_combine_series_auto)(uint64_t* crc, uint32_t part_size, uint32_t n_parts);
void (*crc64ecma_series_auto)(const uint8_t *buffer, uint32_t part_size, uint32_t n_parts, uint64_t* crc_parts);
uint32_t (*crc32c_trim_auto)(CRC32C_Component all, CRC32C_Component prefix, CRC32C_Component suffix);
uint64_t (*crc64ecma_trim_auto)(CRC64ECMA_Component all, CRC64ECMA_Component prefix,
                                CRC64ECMA_Component suffix);

__attribute__((constructor))
static void crc_init() {
    auto tie32 = std::tie(crc32c_auto, crc32c_series_auto, crc32c_combine_auto, crc32c_combine_series_auto, crc32c_trim_auto);
    auto hw32 = std::make_tuple(crc32c_hw, crc32c_series_hw, crc32c_combine_hw, crc32c_combine_series_hw, crc32c_trim_hw); (void)hw32;
    auto sw32 = std::make_tuple(crc32c_sw, crc32c_series_sw, crc32c_combine_sw, crc32c_combine_series_sw, crc32c_trim_sw); (void)sw32;
    tie32 = checksum_simd::Runtime::supports_hw_crc32c() ? hw32 : sw32;
#if CHECKSUM_SIMD_HAS_AVX512
    if (checksum_simd::Runtime::supports_crc64_avx512()) {
        crc64ecma_auto = crc64ecma_hw_avx512;
    } else
#endif
    if (checksum_simd::Runtime::supports_crc64_simd128()) {
        crc64ecma_auto = crc64ecma_hw_simd128;
    } else {
        crc64ecma_auto = crc64ecma_sw;
    }
    crc64ecma_combine_auto = (crc64ecma_auto == crc64ecma_sw) ?
                              crc64ecma_combine_sw :
                              crc64ecma_combine_hw ;
    crc64ecma_trim_auto = (crc64ecma_auto == crc64ecma_sw) ?
                           crc64ecma_trim_sw :
                           crc64ecma_trim_hw;
}

CHECKSUM_SIMD_TARGET_BEGIN

using SIMD = checksum_simd::SIMD128;
using CRC32 = checksum_simd::CRC32CIntrinsic;
using v128 = SIMD::v128;

inline __attribute__((always_inline))
v128 fold_16(v128 x, const uint64_t* rk) {
    return SIMD::clmul<0x10>(x, rk) ^ SIMD::clmul<0x01>(x, rk);
}

CHECKSUM_SIMD_STRICT_ALIASING_DIAG_PUSH
inline __attribute__((always_inline))
v128 load_small(const void* data, size_t n) {
    assert(n < 16);
    const auto* end = reinterpret_cast<const uint8_t*>(data) + n;
    uint64_t tail = 0;
    if (n & 4) {
        end -= 4;
        tail = *reinterpret_cast<const uint32_t*>(end);
    }
    if (n & 2) {
        end -= 2;
        tail = (tail << 16) | *reinterpret_cast<const uint16_t*>(end);
    }
    if (n & 1) {
        --end;
        tail = (tail << 8) | *end;
    }
    if (n & 8) {
        auto head = *reinterpret_cast<const uint64_t*>(data);
        return SIMD::set_u64x2(head, tail);
    }
    return SIMD::set_u64x2(tail, 0);
}
CHECKSUM_SIMD_STRICT_ALIASING_DIAG_POP

inline __attribute__((always_inline))
v128 barrett_reduce_to_u64(v128 x, const uint64_t* rk) {
    auto t = SIMD::clmul<0x00>(x, rk);
    return x ^ SIMD::clmul<0x10>(t, rk) ^ SIMD::bsl8(t);
}

template<size_t blksz, typename T> inline __attribute__((always_inline))
void crc32c_hw_block(const uint8_t*& data, size_t& nbytes, uint32_t& crc) {
    if (nbytes & blksz) {
        static_loop<0, blksz - sizeof(T), sizeof(T)>(BODY(i) {
            crc = CRC32::extend(crc, *(T*)(data + i));
        });
        nbytes -= blksz;
        data += blksz;
    }
}

inline __attribute__((always_inline))
void crc32c_hw_tiny(uint64_t d, size_t nbytes, uint32_t& crc) {
    assert(nbytes <= 7);
    if (nbytes & 1) { crc = CRC32::extend(crc, (uint8_t)d); d >>= 8; }
    if (nbytes & 2) { crc = CRC32::extend(crc, (uint16_t)d); d >>= 16; }
    if (nbytes & 4) { crc = CRC32::extend(crc, (uint32_t)d); }
}

inline __attribute__((always_inline))
void crc32c_hw_small(const uint8_t*& data, size_t nbytes, uint32_t& crc) {
    assert(nbytes < 256);
    if (unlikely(!nbytes || !data)) return;
    crc32c_hw_block<128, uint64_t>(data, nbytes, crc);
    crc32c_hw_block<64,  uint64_t>(data, nbytes, crc);
    crc32c_hw_block<32,  uint64_t>(data, nbytes, crc);
    crc32c_hw_block<16,  uint64_t>(data, nbytes, crc);
    crc32c_hw_block<8,   uint64_t>(data, nbytes, crc);
    if (unlikely(nbytes)) {
        auto x = 8 - nbytes; // buffer size >= 8
        auto d = *(uint64_t*)(data - x);
        d >>= x * 8;
        data += nbytes;
        crc32c_hw_tiny(d, nbytes, crc);
    }
}

template<uint16_t blksz> inline __attribute__((always_inline))
bool crc32c_3way_ILP(const uint8_t*& data, size_t& nbytes, uint32_t& crc) {
    if (nbytes < blksz * 3) return false;
    auto ptr = (const uint64_t*)data;
    const size_t blksz_8 = blksz / 8;
    uint32_t crc1 = 0, crc2 = 0;
    static_loop<0, blksz_8 - 2, 1>(BODY(i) {
        if (i < blksz_8 * 3 / 4)
            __builtin_prefetch(data + blksz*3 + i*8*4, 0, 0);
        crc  = CRC32::extend(crc,  ptr[i]);
        crc1 = CRC32::extend(crc1, ptr[i + blksz_8]);
        crc2 = CRC32::extend(crc2, ptr[i + blksz_8 * 2]);
    });
    crc = CRC32::extend(crc, ptr[blksz_8 - 1]);
    crc1 = CRC32::extend(crc1, ptr[blksz_8 * 2 - 1]);
    // crc2 = crc32c(crc2, ptr[blksz_8 * 3 - 1]);

    auto k = checksum_tables::crc32_merge_constants<blksz>();
    auto c0 = SIMD::set_low_u64(crc);
    auto c1 = SIMD::set_low_u64(crc1);
    auto t = SIMD::clmul<0x00>(c0, k) ^
             SIMD::clmul<0x10>(c1, k);
    crc = CRC32::extend(crc2, ptr[blksz_8 * 3 - 1] ^ SIMD::low_u64(t));
    data += blksz * 3;
    nbytes -= blksz * 3;
    return true;
}

// This is a portable re-imlementation of crc32_iscsi_00()
// in pure x86_64 assembly in ISA-L. It is as fast as the
// latter one for both small and big data blocks. And it is
// event faster than the ARMv8 counterpart in ISA-L, i.e.
// crc32_iscsi_crc_ext().
uint32_t crc32c_hw_portable(const uint8_t *data, size_t nbytes, uint32_t crc) {
    if (unlikely(!nbytes)) return crc;
    if (unlikely(nbytes < 8)) {
        while(nbytes--)
            crc = CRC32::extend(crc, *data++);
        return crc;
    }
    uint8_t l = (~((uint64_t)data) + 1) & 7;
    if (unlikely(l)) {
        auto d = *(uint64_t*)data; // nbytes >= 8
        data += l; nbytes -= l;
        crc32c_hw_tiny(d, l, crc);
    }
    while(crc32c_3way_ILP<512>(data, nbytes, crc));
    crc32c_3way_ILP<256>(data, nbytes, crc);
    crc32c_3way_ILP<128>(data, nbytes, crc);
    crc32c_3way_ILP<64 >(data, nbytes, crc);
    crc32c_hw_small(data, nbytes, crc);
    return crc;
}

uint32_t crc32c_hw_simple(const uint8_t *data, size_t nbytes, uint32_t crc) {
    auto f1 = [](uint32_t crc, uint8_t x)  { return CRC32::extend(crc, x); };
    auto f8 = [](uint32_t crc, uint64_t x) { return CRC32::extend(crc, x); };
    return do_crc(data, nbytes, crc, f1, f8);
}

uint32_t crc32c_hw(const uint8_t *data, size_t nbytes, uint32_t crc) {
    return crc32c_hw_portable(data, nbytes, crc);
}

// *virtually* pad or remove `len` bytes of trailing 0s
// to source data, and return resulting crc value
template<typename T> inline
T crc_apply_shifts(T crc, uint32_t len, const T* table,
                   T (*apply_shift)(T, T)) {
    for (; len; len &= len - 1) {
        auto x = table[__builtin_ctzll(len)];
        crc = apply_shift(crc, x);
    }
    return crc;
}

static uint32_t clmul_modp_crc32c_hw(uint32_t crc1, uint32_t x) {
    auto crc1x = SIMD::set_low_u32(crc1);
    auto cnstx = SIMD::set_low_u32(x);
    crc1x = SIMD::clmul<0x00>(crc1x, cnstx);
    auto dat64 = SIMD::low_u64(crc1x);
    return CRC32::extend((uint32_t)0, dat64);
}

uint32_t crc32c_combine_hw(uint32_t crc1, uint32_t crc2, uint32_t len2) {
    if (unlikely(!crc1)) return crc2;
    if (unlikely(!len2)) return crc1;
    if (unlikely(len2 & 15)) {
        if (unlikely(len2 & 8)) crc1 = CRC32::extend(crc1, (uint64_t)0);
        if (unlikely(len2 & 4)) crc1 = CRC32::extend(crc1, (uint32_t)0);
        if (unlikely(len2 & 2)) crc1 = CRC32::extend(crc1, (uint16_t)0);
        if (unlikely(len2 & 1)) crc1 = CRC32::extend(crc1, (uint8_t)0);
    }
    crc1 = crc_apply_shifts(crc1, len2>>4,
        checksum_tables::crc32c_lshift_table_hw.data() + 4, clmul_modp_crc32c_hw);
    return crc1 ^ crc2;
}

uint32_t crc32c_combine_series_hw(uint32_t* crc, uint32_t part_size, uint32_t n_parts) {
    if (unlikely(!n_parts)) return 0;
    auto res = crc[0];
    for (uint32_t i = 1; i < n_parts; ++i)
        res = crc32c_combine_hw(res, crc[i], part_size);
    return res;
}

// (a*b) % poly
template<typename T, T POLY> inline
T clmul_modp_sw(T a, T b) {
    T pd = 0;
    for(uint64_t i = 0; i < sizeof(T)*8; i++, b>>=1)
        pd = (pd>>1) ^ CVAL(pd&1, POLY) ^ CVAL(b&1, a);
    return pd;
}

uint32_t crc32c_combine_sw(uint32_t crc1, uint32_t crc2, uint32_t len2) {
    if (unlikely(!crc1)) return crc2;
    if (unlikely(!len2)) return crc1;
    crc1 = crc_apply_shifts(crc1, len2, checksum_tables::crc32c_lshift_table_sw.data(),
            &clmul_modp_sw<uint32_t, CRC32C_POLY>);
    return crc1 ^ crc2;
}

static uint32_t crc32c_rshift_hw(uint32_t crc, size_t len) {
    return crc_apply_shifts(crc, len, checksum_tables::crc32c_rshift_table_hw.data(),
            &clmul_modp_crc32c_hw);
}

static uint32_t crc32c_rshift_sw(uint32_t crc, size_t len) {
    return crc_apply_shifts(crc, len, checksum_tables::crc32c_rshift_table_sw.data(),
            &clmul_modp_sw<uint32_t, CRC32C_POLY>);
}

template<typename T, typename F1, typename F2> static inline
auto do_crc_trim(T all, T prefix, T suffix, F1 rm_prefix, F2 rm_suffix) -> decltype(all.crc) {
    if (all.size < prefix.size + suffix.size)
        LOG_ERRNO_RETURN(EINVAL, 0, "total size (`) must be > summed sizes of prefix (`) + suffix (`)", all.size, prefix.size, suffix.size);
    if (unlikely(!prefix.size && !suffix.size))
        return all.crc;
    auto crc = all.crc;
    if (prefix.size)
        // crc ^= prefix.crc << (all.size - prefix.size);
        crc = rm_prefix(prefix.crc, crc, all.size - prefix.size);
    if (suffix.size)
        // crc = (crc ^ suffix.crc) >> suffix.size;
        crc = rm_suffix(crc ^ suffix.crc, suffix.size);
    return crc;
}

uint32_t crc32c_trim_sw(CRC32C_Component all, CRC32C_Component prefix, CRC32C_Component suffix) {
    return do_crc_trim(all, prefix, suffix, &crc32c_combine_sw, &crc32c_rshift_sw);
}

uint32_t crc32c_trim_hw(CRC32C_Component all, CRC32C_Component prefix, CRC32C_Component suffix) {
    return do_crc_trim(all, prefix, suffix, &crc32c_combine_hw, &crc32c_rshift_hw);
}

uint32_t crc32c_combine_series_sw(uint32_t* crc, uint32_t part_size, uint32_t n_parts) {
    if (unlikely(!n_parts)) return 0;
    auto res = crc[0];
    for (uint32_t i = 1; i < n_parts; ++i)
        res = crc32c_combine_sw(res, crc[i], part_size);
    return res;
}

void crc32c_series_sw(const uint8_t *buffer, uint32_t part_size, uint32_t n_parts, uint32_t* crc_parts) {
    for (uint32_t i = 0; i < n_parts; ++i)
        crc_parts[i] = crc32c_sw(buffer + i * part_size, part_size, 0);
}

void crc32c_series_hw(const uint8_t *buffer, uint32_t part_size, uint32_t n_parts, uint32_t* crc_parts) {
    const size_t BATCH = 4;
    auto part_main = part_size / 8 * 8;
    auto part_remain = part_size % 8;
    auto batch_crc = [&](auto ptr, size_t batch) __attribute__((always_inline)) {
        #pragma GCC unroll 4
        for (size_t k = 0; k < batch; ++k) {
            crc_parts[k] = CRC32::extend(crc_parts[k], *ptr);
            ptr = decltype(ptr)((char*)ptr + part_size);
        }
    };
    auto batch_blocks_crc = [&](const uint8_t* ptr, auto batch) __attribute__((always_inline)) {
        for (size_t j = 0; j < batch; ++j)
            crc_parts[j] = 0;
        for (; ptr < buffer + part_main; ptr += 8) {
            batch_crc((uint64_t*)ptr, batch);
        }
        if (unlikely(part_main)) {
            if (part_remain & 4) { batch_crc((uint32_t*)ptr, batch); ptr += 4; }
            if (part_remain & 2) { batch_crc((uint16_t*)ptr, batch); ptr += 2; }
            if (part_remain & 1) { batch_crc((uint8_t*)ptr, batch); ptr += 1; }
        }
    };
    for (; n_parts >= BATCH; n_parts -= BATCH) {
        batch_blocks_crc(buffer, BATCH);
        buffer += part_size * BATCH;
        crc_parts += BATCH;
    }
    if (unlikely(n_parts))
        batch_blocks_crc(buffer, n_parts);
}

#define RK(i) &checksum_tables::crc64ecma_rk[(i)-1]

__attribute__((aligned(16), used))
const static uint64_t mask[6] = {
    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
    0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF,
    0x8080808080808080, 0x8080808080808080,
};

#define MASK(i) SIMD::loadu(&mask[((i)-1)*2])

const static uint64_t pshufb_shf_table[4] = {
    0x8786858483828100, 0x8f8e8d8c8b8a8988,
    0x0706050403020100, 0x000e0d0c0b0a0908};

inline void* get_shf_table(size_t i) {
    return (char*)pshufb_shf_table + i;
}

inline __attribute__((always_inline))
v128 crc64ecma_hw_big_simd128(const uint8_t*& data, size_t& nbytes, uint64_t crc) {
    v128 xmm[8];
    auto& ptr = (const v128*&)data;
    static_loop<0, 7, 1>(BODY(i){ xmm[i] = SIMD::loadu(ptr+i); });
    xmm[0] ^= SIMD::set_low_u64(~crc); ptr += 8; nbytes -= 128;
    do {
        static_loop<0, 7, 1>(BODY(i) {
            xmm[i] = fold_16(xmm[i], RK(3)) ^ SIMD::loadu(ptr+i);
        });
        ptr += 8; nbytes -= 128;
    } while (nbytes >= 128);
    static_loop<0, 6, 1>(BODY(i) {
        auto I = (i == 6) ? 1 : (9 + i * 2);
        xmm[7] ^= fold_16(xmm[i], RK(I));
    });
    return xmm[7];
}

template<typename F> inline __attribute__((always_inline))
uint64_t crc64ecma_hw_portable(const uint8_t *data, size_t nbytes, uint64_t crc, F hw_big) {
    if (unlikely(!nbytes || !data)) return crc;
    v128 xmm7 = SIMD::set_low_u64(~crc);
    auto& ptr = (const v128*&)data;
    if (nbytes >= 256) {
        xmm7 = hw_big(data, nbytes, crc);
    } else if (nbytes >= 16) {
        xmm7 ^= SIMD::loadu(ptr++);
        nbytes -= 16;
    } else /* 0 < nbytes < 16*/ {
        xmm7 ^= load_small(data, nbytes);
        if (nbytes >= 8) {
            auto shf = SIMD::loadu(get_shf_table(nbytes));
            xmm7 = SIMD::shuffle_bytes(xmm7, shf);
            goto _128_done;
        } else {
            auto shf = SIMD::loadu(get_shf_table(nbytes + 8));
            xmm7 = SIMD::shuffle_bytes(xmm7, shf);
            goto _barrett;
        }
    }

    while (nbytes >= 16) {
        xmm7 = fold_16(xmm7, RK(1)) ^ SIMD::loadu(ptr++);
        nbytes -= 16;
    }

    if (nbytes) {
        auto p = data + nbytes - 16;
        auto remainder = SIMD::loadu((v128*)p);
        auto xmm0 = SIMD::loadu(get_shf_table(nbytes));
        auto xmm2 = xmm7;
        xmm7 = SIMD::shuffle_bytes(xmm7, xmm0);
        xmm0 ^= MASK(3);
        xmm2 = SIMD::shuffle_bytes(xmm2, xmm0);
        xmm2 = SIMD::blend_bytes(xmm2, remainder, xmm0);
        xmm7 = xmm2 ^ fold_16(xmm7, RK(1));
    }
_128_done:
    xmm7  =  SIMD::clmul<0x00>(xmm7, RK(5)) ^ SIMD::bsr8(xmm7);
_barrett:
    xmm7 = barrett_reduce_to_u64(xmm7, RK(7));
    crc = ~SIMD::high_u64(xmm7);
    return crc;
}

uint64_t crc64ecma_hw_simd128(const uint8_t *buf, size_t len, uint64_t crc) {
    return crc64ecma_hw_portable(buf, len, crc, crc64ecma_hw_big_simd128);
}

static uint64_t clmul_modp_crc64ecma_hw(uint64_t crc, uint64_t x) {
    auto crc1x = SIMD::set_low_u64(crc);
    auto constx = SIMD::set_low_u64(x);
    crc1x = SIMD::clmul<0x00>(crc1x, constx);
    return SIMD::high_u64(barrett_reduce_to_u64(crc1x, RK(7)));
}

uint64_t crc64ecma_combine_hw(uint64_t crc1, uint64_t crc2, uint32_t len2) {
    if (unlikely(!crc1)) return crc2;
    return crc2 ^ crc_apply_shifts(crc1, len2,
    checksum_tables::crc64ecma_lshift_table.data(), clmul_modp_crc64ecma_hw);
}

// (a*b) % poly
inline uint64_t clmul_modp64_sw(uint64_t a, uint64_t b) {
    uint64_t pd = 0;
    for(uint64_t i = 0; i <= sizeof(a)*8; i++, b>>=1)
        pd = (pd>>1) ^ CVAL(pd&1, CRC64ECMA_POLY) ^ CVAL(b&1, a);
    return pd;
}

uint64_t crc64ecma_combine_sw(uint64_t crc1, uint64_t crc2, uint32_t len2) {
    if (unlikely(!crc1)) return crc2;
    return crc2 ^ crc_apply_shifts(crc1, len2,
        checksum_tables::crc64ecma_lshift_table.data(), &clmul_modp64_sw);
}

static uint64_t crc64ecma_rshift_hw(uint64_t crc, uint64_t n) {
    return crc_apply_shifts(crc, n, checksum_tables::crc64ecma_rshift_table.data(),
        clmul_modp_crc64ecma_hw);
}

static uint64_t crc64ecma_rshift_sw(uint64_t crc, uint64_t n) {
    return crc_apply_shifts(crc, n, checksum_tables::crc64ecma_rshift_table.data(),
    &clmul_modp64_sw);
}

uint64_t crc64ecma_trim_hw(CRC64ECMA_Component all,
                           CRC64ECMA_Component prefix,
                           CRC64ECMA_Component suffix) {
    return do_crc_trim(all, prefix, suffix, &crc64ecma_combine_hw, &crc64ecma_rshift_hw);
}

uint64_t crc64ecma_trim_sw(CRC64ECMA_Component all,
                           CRC64ECMA_Component prefix,
                           CRC64ECMA_Component suffix) {
    return do_crc_trim(all, prefix, suffix, &crc64ecma_combine_sw, &crc64ecma_rshift_sw);
}

#if CHECKSUM_SIMD_HAS_AVX512
CHECKSUM_SIMD_TARGET_END
CHECKSUM_SIMD_TARGET_AVX512_BEGIN
#define _RK(i) &checksum_tables::crc64ecma_rk512[(i)+1]

using SIMD512 = checksum_simd::SIMD512;
using v512 = SIMD512::v512;

inline __attribute__((always_inline))
v512 fold_128(v512 x, v512 rk, v512 next) {
    auto lo = _mm512_clmulepi64_epi128(x, rk, 0x01);
    auto hi = _mm512_clmulepi64_epi128(x, rk, 0x10);
    return _mm512_ternarylogic_epi64(lo, hi, next, 0x96);
}

inline __attribute__((always_inline))
v128 xor_reduce_to_128(v512 x) {
    auto swapped = _mm512_shuffle_i64x2(x, x, 0x4e);
    auto folded = _mm256_xor_si256(_mm512_castsi512_si256(swapped),
                                   _mm512_castsi512_si256(x));
    return _mm256_extracti64x2_epi64(folded, 0) ^
           _mm256_extracti64x2_epi64(folded, 1);
}

inline __attribute__((always_inline))
v128 crc64ecma_hw_big_avx512(const uint8_t*& data, size_t& nbytes, uint64_t crc) {
    assert(nbytes >= 256);
    auto crc0 = SIMD512::set_low_u64(~crc);
    auto& ptr = (const v512*&)data;
    auto zmm0 = SIMD512::loadu(ptr++); zmm0 ^= crc0;
    auto zmm4 = SIMD512::loadu(ptr++);
    nbytes -= 128;
    if (nbytes < 384) {
        auto rk3 = SIMD512::broadcast_128(_RK(3));
        do { // fold 128 bytes each iteration
            zmm0 = fold_128(zmm0, rk3, SIMD512::loadu(ptr++));
            zmm4 = fold_128(zmm4, rk3, SIMD512::loadu(ptr++));
            nbytes -= 128;
        } while (nbytes >= 128);
    } else { // nbytes >= 384
        auto rk_1_2 = SIMD512::broadcast_128(&checksum_tables::crc64ecma_rk512[0]);
        auto zmm7   = SIMD512::loadu(ptr++);
        auto zmm8   = SIMD512::loadu(ptr++);
        nbytes -= 128;
        do { // fold 256 bytes each iteration
            zmm0 = fold_128(zmm0, rk_1_2, SIMD512::loadu(ptr++));
            zmm4 = fold_128(zmm4, rk_1_2, SIMD512::loadu(ptr++));
            zmm7 = fold_128(zmm7, rk_1_2, SIMD512::loadu(ptr++));
            zmm8 = fold_128(zmm8, rk_1_2, SIMD512::loadu(ptr++));
            nbytes -= 256;
        } while (nbytes >= 256);
        auto rk3 = SIMD512::broadcast_128(_RK(3));
        zmm0 = fold_128(zmm0, rk3, zmm7);
        zmm4 = fold_128(zmm4, rk3, zmm8);
    }
    auto zmm7 = SIMD512::set_low_128(SIMD512::upper_128(zmm4));
    auto zmm1 = fold_128(zmm0, SIMD512::loadu(_RK(9)), zmm7);
         zmm1 = fold_128(zmm4, SIMD512::loadu(_RK(17)), zmm1);
    return xor_reduce_to_128(zmm1);
}

uint64_t crc64ecma_hw_avx512(const uint8_t *buf, size_t len, uint64_t crc) {
    return crc64ecma_hw_portable(buf, len, crc, crc64ecma_hw_big_avx512);
}
CHECKSUM_SIMD_TARGET_AVX512_END
CHECKSUM_SIMD_TARGET_BEGIN
#endif

uint64_t crc64ecma_hw(const uint8_t *buffer, size_t nbytes, uint64_t crc) {
    return crc64ecma_auto(buffer, nbytes, crc);
}

CHECKSUM_SIMD_TARGET_END


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

#include "crc_shift_tables.h"

#include <limits>
#include <type_traits>
#include <utility>

namespace checksum_tables {
namespace detail {

template <typename T>
constexpr T cval(T c, T val) {
    return (c ? static_cast<T>(~T(0)) : T(0)) & val;
}

template <typename T, T Poly>
constexpr T clmul_modp(T a, T b) {
    static_assert(std::is_unsigned<T>::value, "T must be unsigned");
    constexpr std::size_t kBits = std::numeric_limits<T>::digits;
    T pd = 0;
    for (std::size_t i = 0; i < kBits; ++i) {
        pd = static_cast<T>((pd >> 1) ^ cval<T>(pd & 1, Poly) ^ cval<T>(b & 1, a));
        b >>= 1;
    }
    return pd;
}

template <typename T, T Poly>
constexpr T pow_mod_unsigned(T base, uint64_t exp) {
    constexpr std::size_t kBits = std::numeric_limits<T>::digits;
    T prd = static_cast<T>(T(1) << (kBits - 1));
    while (exp) {
        if (exp & 1) prd = clmul_modp<T, Poly>(prd, base);
        exp >>= 1;
        if (exp) base = clmul_modp<T, Poly>(base, base);
    }
    return prd;
}

template <typename T, T Poly, T Base>
constexpr T inverse_base() {
    return pow_mod_unsigned<T, Poly>(Base, static_cast<uint64_t>(std::numeric_limits<T>::max() - 1));
}

template <typename T, T Poly, T Base>
constexpr T pow_mod_signed(int64_t exp) {
    return exp >= 0
               ? pow_mod_unsigned<T, Poly>(Base, static_cast<uint64_t>(exp))
               : pow_mod_unsigned<T, Poly>(inverse_base<T, Poly, Base>(), static_cast<uint64_t>(-exp));
}

template <typename T, T Poly, T Base, T InverseBase>
constexpr T pow_mod_signed_with_inverse(int64_t exp) {
    return exp >= 0
               ? pow_mod_unsigned<T, Poly>(Base, static_cast<uint64_t>(exp))
               : pow_mod_unsigned<T, Poly>(InverseBase, static_cast<uint64_t>(-exp));
}

template <std::size_t... I>
constexpr std::array<uint32_t, sizeof...(I)> make_crc32c_lshift_hw(std::index_sequence<I...>) {
    return {{pow_mod_signed_with_inverse<uint32_t, 0x82f63b78u, 0x40000000u, 0x05ec76f1u>(
        static_cast<int64_t>((uint64_t(1) << (I + 3)) - 33))...}};
}

template <std::size_t... I>
constexpr std::array<uint32_t, sizeof...(I)> make_crc32c_rshift_hw(std::index_sequence<I...>) {
    return {{pow_mod_signed_with_inverse<uint32_t, 0x82f63b78u, 0x40000000u, 0x05ec76f1u>(
        -static_cast<int64_t>((uint64_t(1) << (I + 3)) + 33))...}};
}

template <std::size_t... I>
constexpr std::array<uint32_t, sizeof...(I)> make_crc32c_lshift_sw(std::index_sequence<I...>) {
    return {{pow_mod_signed<uint32_t, 0x82f63b78u, 0x40000000u>(
        static_cast<int64_t>(uint64_t(1) << (I + 3)))...}};
}

template <std::size_t... I>
constexpr std::array<uint32_t, sizeof...(I)> make_crc32c_rshift_sw(std::index_sequence<I...>) {
    return {{pow_mod_signed_with_inverse<uint32_t, 0x82f63b78u, 0x40000000u, 0x05ec76f1u>(
        -static_cast<int64_t>(uint64_t(1) << (I + 3)))...}};
}

template <std::size_t... I>
constexpr std::array<uint64_t, sizeof...(I)> make_crc64ecma_lshift(std::index_sequence<I...>) {
    return {{pow_mod_signed<uint64_t, 0xc96c5795d7870f42ull, 0x4000000000000000ull>(
        static_cast<int64_t>((uint64_t(1) << (I + 3)) - 1))...}};
}

template <std::size_t... I>
constexpr std::array<uint64_t, sizeof...(I)> make_crc64ecma_rshift(std::index_sequence<I...>) {
    return {{pow_mod_signed_with_inverse<uint64_t, 0xc96c5795d7870f42ull, 0x4000000000000000ull, 0x92d8af2baf0e1e85ull>(
        -static_cast<int64_t>((uint64_t(1) << (I + 3)) + 1))...}};
}

constexpr uint32_t crc32_pow_mod_for_merge(int64_t exp) {
    return pow_mod_signed_with_inverse<uint32_t, 0x82f63b78u, 0x40000000u, 0x05ec76f1u>(exp);
}

constexpr std::array<uint64_t, 8> make_crc32_merge_constants() {
    return {{
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 10) - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 9)  - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 11) - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 10) - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 12) - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 11) - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 13) - 33)),
        static_cast<uint64_t>(crc32_pow_mod_for_merge((1ll << 12) - 33)),
    }};
}

constexpr uint64_t crc64ecma_pow_mod_for_rk(int64_t exp) {
    return pow_mod_signed_with_inverse<uint64_t, 0xc96c5795d7870f42ull, 0x4000000000000000ull, 0x92d8af2baf0e1e85ull>(exp);
}

constexpr uint64_t crc64ecma_rk_value(std::size_t i) {
    return
        (i == 0) ? crc64ecma_pow_mod_for_rk((1ll << 7) - 1) :
        (i == 1) ? crc64ecma_pow_mod_for_rk((1ll << 7) - 1 + 64) :
        (i == 2) ? crc64ecma_pow_mod_for_rk((1ll << 10) - 1) :
        (i == 3) ? crc64ecma_pow_mod_for_rk((1ll << 10) - 1 + 64) :
        (i == 4) ? crc64ecma_pow_mod_for_rk((1ll << 7) - 1) :
        (i == 5) ? 0ull :
        (i == 6) ? 0x9c3e466c172963d5ull :
        (i == 7) ? (0x92d8af2baf0e1e85ull - 1) :
                   crc64ecma_pow_mod_for_rk(895 - ((static_cast<int64_t>(i) - 8) / 2) * 128 +
                                            ((i - 8) & 1 ? 64 : 0));
}

template<std::size_t... I>
constexpr std::array<uint64_t, sizeof...(I)> make_crc64ecma_rk(std::index_sequence<I...>) {
    return {{crc64ecma_rk_value(I)...}};
}

constexpr uint64_t crc64ecma_rk512_value(std::size_t i) {
    return
        (i == 0) ? 0xf31fd9271e228b79ull :
        (i == 1) ? 0x8260adf2381ad81cull :
        (i >= 2 && i <= 21) ? crc64ecma_rk_value(i - 2) :
        (i == 22) ? crc64ecma_rk_value(0) :
        (i == 23) ? crc64ecma_rk_value(1) :
                    0ull;
}

template<std::size_t... I>
constexpr std::array<uint64_t, sizeof...(I)> make_crc64ecma_rk512(std::index_sequence<I...>) {
    return {{crc64ecma_rk512_value(I)...}};
}

}  // namespace detail

const std::array<uint32_t, 32> crc32c_lshift_table_hw =
    detail::make_crc32c_lshift_hw(std::make_index_sequence<32>{});
const std::array<uint32_t, 32> crc32c_rshift_table_hw =
    detail::make_crc32c_rshift_hw(std::make_index_sequence<32>{});
const std::array<uint32_t, 32> crc32c_lshift_table_sw =
    detail::make_crc32c_lshift_sw(std::make_index_sequence<32>{});
const std::array<uint32_t, 32> crc32c_rshift_table_sw =
    detail::make_crc32c_rshift_sw(std::make_index_sequence<32>{});

const std::array<uint64_t, 33> crc64ecma_lshift_table =
    detail::make_crc64ecma_lshift(std::make_index_sequence<33>{});
const std::array<uint64_t, 32> crc64ecma_rshift_table =
    detail::make_crc64ecma_rshift(std::make_index_sequence<32>{});

const std::array<uint64_t, 8> crc32_merge_constants_table =
    detail::make_crc32_merge_constants();

const std::array<uint64_t, 20> crc64ecma_rk =
    detail::make_crc64ecma_rk(std::make_index_sequence<20>{});

const std::array<uint64_t, 26> crc64ecma_rk512 =
    detail::make_crc64ecma_rk512(std::make_index_sequence<26>{});

const uint64_t* crc32_merge_constants_by_block(uint16_t blksz) {
    switch (blksz) {
    case 64:
        return crc32_merge_constants_table.data();
    case 128:
        return crc32_merge_constants_table.data() + 2;
    case 256:
        return crc32_merge_constants_table.data() + 4;
    case 512:
        return crc32_merge_constants_table.data() + 6;
    default:
        return nullptr;
    }
}

}  // namespace checksum_tables

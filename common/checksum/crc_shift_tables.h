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

#include <array>
#include <cstddef>
#include <cstdint>

namespace checksum_tables {

extern const std::array<uint32_t, 32> crc32c_lshift_table_hw;
extern const std::array<uint32_t, 32> crc32c_rshift_table_hw;
extern const std::array<uint32_t, 32> crc32c_lshift_table_sw;
extern const std::array<uint32_t, 32> crc32c_rshift_table_sw;

extern const std::array<uint64_t, 33> crc64ecma_lshift_table;
extern const std::array<uint64_t, 32> crc64ecma_rshift_table;

extern const std::array<uint64_t, 8> crc32_merge_constants_table;
extern const std::array<uint64_t, 20> crc64ecma_rk;
extern const std::array<uint64_t, 26> crc64ecma_rk512;

const uint64_t* crc32_merge_constants_by_block(uint16_t blksz);

template<uint16_t blksz>
inline const uint64_t* crc32_merge_constants() {
    static_assert(blksz == 64 || blksz == 128 || blksz == 256 || blksz == 512,
                  "unsupported block size");
    return crc32_merge_constants_by_block(blksz);
}

}  // namespace checksum_tables

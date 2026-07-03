// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Ternary (base-3) weight unpacking utilities.
// ut1_5: 5 trits packed per byte using base-3 encoding.
//   byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4, where t_i in {0, 1, 2}
// ut2: 4 values packed per byte using 2-bit encoding (shift+mask).
//   byte = v0 | (v1<<2) | (v2<<4) | (v3<<6), where v_i in {0, 1, 2, 3}

// Unpack a single trit from a base-3 packed byte (ut1_5).
// trit_idx must be in [0, 4].
//inline char unpack_trit(uchar packed_byte, uint trit_idx) {
//    uint val = (uint)packed_byte;
//    switch (trit_idx) {
//        case 0: return (char)(val % 3);
//        case 1: return (char)((val / 3) % 3);
//        case 2: return (char)((val / 9) % 3);
//        case 3: return (char)((val / 27) % 3);
//        case 4: return (char)((val / 81) % 3);
//        default: return 0;
//    }
//}

inline char unpack_trit(uchar packed_byte, uint trit_idx) {
    uint val = (uint)packed_byte; // Promote to 32-bit uint so multiplication doesn't overflow
    
    // Magic division constants for 8-bit inputs
    // div3  = val / 3
    // div9  = val / 9
    // div27 = val / 27
    // div81 = val / 81

    switch (trit_idx) {
        case 0: {
            uint div3 = (val * 171) >> 9;
            return (char)(val - (div3 * 3));
        }
        case 1: {
            uint div3 = (val * 171) >> 9;
            uint div9 = (val * 57) >> 9;
            return (char)(div3 - (div9 * 3));
        }
        case 2: {
            uint div9 = (val * 57) >> 9;
            uint div27 = (val * 19) >> 9;
            return (char)(div9 - (div27 * 3));
        }
        case 3: {
            uint div27 = (val * 19) >> 9;
            uint div81 = (val * 203) >> 14;
            return (char)(div27 - (div81 * 3));
        }
        case 4: {
            uint div81 = (val * 203) >> 14;
            return (char)div81;
        }
        default: return 0;
    }
}

// Unpack a single 2-bit value from a ut2 packed byte.
// val_idx must be in [0, 3].
inline char unpack_ut2(uchar packed_byte, uint val_idx) {
    return (char)((packed_byte >> (val_idx * 2)) & 0x3);
}
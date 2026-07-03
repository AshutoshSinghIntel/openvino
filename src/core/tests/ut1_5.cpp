// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include "gtest/gtest.h"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov;
using namespace ov::element;
using namespace testing;

// --- Iterator tests ---
// These validate the base-3 BitProxy and Iterator in element_iterator.hpp:
//   - BitProxy<T, ut1_5>::operator=() packs a trit into a byte via base-3 arithmetic
//   - BitProxy<T, ut1_5>::operator value_type() unpacks via (byte / pow3[idx]) % 3
//   - Iterator::operator++/--/+=/- navigates trits within and across bytes (5 trits/byte)

TEST(ut1_5_iterator, write_data) {
    // 5 trits: {2, 0, 1, 2, 1} -> byte = 2 + 0*3 + 1*9 + 2*27 + 1*81 = 146
    // 5 trits: {0, 1, 2, 0, 2} -> byte = 0 + 1*3 + 2*9 + 0*27 + 2*81 = 183
    constexpr size_t elements_count = 10;
    auto input = std::array<int8_t, elements_count>{2, 0, 1, 2, 1, 0, 1, 2, 0, 2};
    auto output = std::array<uint8_t, 2>{};
    auto iter = element::iterator<element::ut1_5>(reinterpret_cast<int8_t*>(output.data()));

    std::copy(input.begin(), input.end(), iter);

    EXPECT_EQ(output[0], 146);  // 2 + 0*3 + 1*9 + 2*27 + 1*81
    EXPECT_EQ(output[1], 183);  // 0 + 1*3 + 2*9 + 0*27 + 2*81
}

TEST(ut1_5_iterator, read_data) {
    // byte 146 = trits {2, 0, 1, 2, 1}
    // byte 183 = trits {0, 1, 2, 0, 2}
    auto input = std::array<int8_t, 2>{static_cast<int8_t>(146), static_cast<int8_t>(183)};
    auto iter = element::iterator<element::ut1_5>(input.data());

    std::vector<int8_t> result(iter, iter + 10);
    EXPECT_EQ(result, (std::vector<int8_t>{2, 0, 1, 2, 1, 0, 1, 2, 0, 2}));
}

TEST(ut1_5_iterator, increment_decrement) {
    // byte = 2 + 1*3 + 0*9 + 2*27 + 1*81 = 140
    auto input = std::array<int8_t, 1>{static_cast<int8_t>(140)};
    auto iter = element::iterator<element::ut1_5>(input.data());

    EXPECT_EQ(*iter, 2);    // trit 0
    ++iter;
    EXPECT_EQ(*iter, 1);    // trit 1
    ++iter;
    EXPECT_EQ(*iter, 0);    // trit 2
    --iter;
    EXPECT_EQ(*iter, 1);    // back to trit 1
    --iter;
    EXPECT_EQ(*iter, 2);    // back to trit 0
}

TEST(ut1_5_iterator, advance_by_offset) {
    // byte 0: 146 = {2, 0, 1, 2, 1}
    // byte 1: 183 = {0, 1, 2, 0, 2}
    auto input = std::array<int8_t, 2>{static_cast<int8_t>(146), static_cast<int8_t>(183)};
    auto iter = element::iterator<element::ut1_5>(input.data());

    EXPECT_EQ(*(iter + 0), 2);   // byte 0, trit 0
    EXPECT_EQ(*(iter + 4), 1);   // byte 0, trit 4
    EXPECT_EQ(*(iter + 5), 0);   // byte 1, trit 0
    EXPECT_EQ(*(iter + 9), 2);   // byte 1, trit 4
}

TEST(ut1_5_iterator, write_single_value) {
    auto data = std::array<uint8_t, 1>{0};
    auto iter = element::iterator<element::ut1_5>(reinterpret_cast<int8_t*>(data.data()));

    // Write trit index 0 = 1
    *iter = 1;
    EXPECT_EQ(data[0], 1);  // 1 + 0 + 0 + 0 + 0

    // Write trit index 2 = 2 (advance 2 positions)
    auto iter2 = iter + 2;
    *iter2 = 2;
    EXPECT_EQ(data[0], 19);  // 1 + 0*3 + 2*9 + 0*27 + 0*81
}

TEST(ut1_5_iterator, all_zeros) {
    auto input = std::array<int8_t, 1>{0};  // all trits = 0
    auto iter = element::iterator<element::ut1_5>(input.data());

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(*(iter + i), 0);
    }
}

TEST(ut1_5_iterator, all_twos) {
    // All trits = 2: byte = 2 + 2*3 + 2*9 + 2*27 + 2*81 = 242
    auto input = std::array<int8_t, 1>{static_cast<int8_t>(242)};
    auto iter = element::iterator<element::ut1_5>(input.data());

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(*(iter + i), 2);
    }
}

// --- Constant tests ---
// These validate that ov::op::v0::Constant correctly handles ut1_5:
//   - Constant constructor calls element::Iterator to pack input values into base-3 bytes
//     (src/core/src/op/constant.cpp — fill_or_write via SUPPORTED_ET dispatch)
//   - cast_vector<T>() uses element::Iterator to unpack bytes back to trit values
//   - get_byte_size() uses ov::util::get_memory_size() which returns ceil(N/5)
//     (src/core/src/memory_util.cpp — is_base3_type branch)
//   - in_t_range<ut1_5>() rejects values outside [0, 2]
//     (src/core/src/op/constant.cpp — in_t_range)

TEST(ut1_5_constant, create_and_read) {
    Shape shape{10};
    std::vector<int8_t> input{2, 0, 1, 2, 1, 0, 1, 2, 0, 2};
    // Constructor: Iterator packs 10 trits into 2 bytes via base-3 BitProxy::operator=
    op::v0::Constant c(element::ut1_5, shape, input);

    // Verify buffer size: ceil(10/5) = 2 bytes
    EXPECT_EQ(c.get_byte_size(), 2);
    EXPECT_EQ(c.get_element_type(), element::ut1_5);
    EXPECT_EQ(c.get_shape(), shape);

    // Verify raw packed bytes are correct
    // byte 0: 2 + 0*3 + 1*9 + 2*27 + 1*81 = 146
    // byte 1: 0 + 1*3 + 2*9 + 0*27 + 2*81 = 183
    auto raw = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(raw[0], 146) << "Packed byte 0 should be 2+0+9+54+81=146";
    EXPECT_EQ(raw[1], 183) << "Packed byte 1 should be 0+3+18+0+162=183";

    // cast_vector uses Iterator to unpack: BitProxy::operator value_type() does (byte/pow3[idx])%3
    auto v = c.cast_vector<int8_t>();
    EXPECT_EQ(v, input) << "Unpacked trits should match original input";
}

TEST(ut1_5_constant, create_broadcast) {
    Shape shape{5};
    // Broadcast: single value 1 is replicated to all 5 positions by Constant constructor
    op::v0::Constant c(element::ut1_5, shape, std::vector<int8_t>{1});

    // Verify raw byte: 1 + 1*3 + 1*9 + 1*27 + 1*81 = 121
    auto raw = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(raw[0], 121) << "All-ones byte should be 1+3+9+27+81=121";

    // cast_vector unpacks via Iterator
    auto v = c.cast_vector<int8_t>();
    EXPECT_EQ(v, (std::vector<int8_t>{1, 1, 1, 1, 1}));
}

TEST(ut1_5_constant, byte_size_various_shapes) {
    // 5 trits = 1 byte
    {
        op::v0::Constant c(element::ut1_5, Shape{5}, std::vector<int8_t>{0, 1, 2, 0, 1});
        EXPECT_EQ(c.get_byte_size(), 1);
    }
    // 6 trits = 2 bytes (ceil(6/5) = 2)
    {
        op::v0::Constant c(element::ut1_5, Shape{6}, std::vector<int8_t>{0, 1, 2, 0, 1, 2});
        EXPECT_EQ(c.get_byte_size(), 2);
    }
    // 10 trits = 2 bytes
    {
        op::v0::Constant c(element::ut1_5, Shape{10}, std::vector<int8_t>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0});
        EXPECT_EQ(c.get_byte_size(), 2);
    }
    // 11 trits = 3 bytes (ceil(11/5) = 3)
    {
        op::v0::Constant c(element::ut1_5, Shape{11}, std::vector<int8_t>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1});
        EXPECT_EQ(c.get_byte_size(), 3);
    }
}

TEST(ut1_5_constant, value_out_of_range) {
    Shape shape{1};
    // Value 3 is out of range for ternary (only 0, 1, 2 valid)
    EXPECT_ANY_THROW(op::v0::Constant c(element::ut1_5, shape, 3));
    // Value -1 is out of range
    EXPECT_ANY_THROW(op::v0::Constant c(element::ut1_5, shape, -1));
}

// --- Dequantization subgraph test ---
// Validates the full ternary dequant chain that the GPU plugin will later fuse:
//   Constant(ut1_5) → Convert(f32) → Subtract(zp=1) → Multiply(scale)
//
// Each step is evaluated individually because Node::evaluate() does NOT
// auto-trace upstream ops — it requires explicit input tensors.
//
// Code flow per step:
//   Convert::evaluate  → src/core/src/op/convert.cpp (CONVERT_ET_LIST dispatch)
//                       → reference::convert() uses Iterator<ut1_5> to unpack trits
//   Subtract::evaluate → src/core/src/op/subtract.cpp (standard f32 arithmetic)
//   Multiply::evaluate → src/core/src/op/multiply.cpp (standard f32 arithmetic)

TEST(ut1_5_dequant, constant_convert_subtract_multiply) {
    // Step 1: Create Constant with known ternary values
    // Values {0, 1, 2, 0, 2} represent offset-encoded {-1, 0, +1, -1, +1}
    auto weights = op::v0::Constant::create(element::ut1_5, Shape{5}, std::vector<int8_t>{0, 1, 2, 0, 2});

    // Verify Constant was created correctly
    ASSERT_EQ(weights->get_element_type(), element::ut1_5);
    ASSERT_EQ(weights->get_byte_size(), 1);  // ceil(5/5) = 1 byte

    // Step 2: Convert ut1_5 → f32
    // This exercises: convert.cpp CONVERT_ET_LIST → Iterator<ut1_5> reads trits → static_cast to float
    // Input: 1 byte of packed ut1_5 data (from Constant's buffer)
    // Expected output: {0.0, 1.0, 2.0, 0.0, 2.0}
    auto convert_node = std::make_shared<op::v0::Convert>(weights, element::f32);
    ov::Tensor weights_tensor(element::ut1_5, Shape{5}, const_cast<void*>(weights->get_data_ptr()));
    ov::TensorVector convert_inputs{weights_tensor};
    ov::TensorVector convert_outputs{ov::Tensor(element::f32, Shape{5})};
    ASSERT_TRUE(convert_node->evaluate(convert_outputs, convert_inputs));

    auto converted = convert_outputs[0].data<float>();
    EXPECT_FLOAT_EQ(converted[0], 0.0f) << "trit 0 → float 0.0";
    EXPECT_FLOAT_EQ(converted[1], 1.0f) << "trit 1 → float 1.0";
    EXPECT_FLOAT_EQ(converted[2], 2.0f) << "trit 2 → float 2.0";
    EXPECT_FLOAT_EQ(converted[3], 0.0f) << "trit 0 → float 0.0";
    EXPECT_FLOAT_EQ(converted[4], 2.0f) << "trit 2 → float 2.0";

    // Step 3: Subtract zero_point (zp=1)
    // This shifts offset encoding {0,1,2} to signed {-1,0,+1}
    // Expected: {0-1, 1-1, 2-1, 0-1, 2-1} = {-1, 0, 1, -1, 1}
    auto sub_node = std::make_shared<op::v1::Subtract>(convert_node,
        op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.0f}));
    ov::Tensor zp_tensor(element::f32, Shape{});
    *zp_tensor.data<float>() = 1.0f;
    ov::TensorVector sub_inputs{convert_outputs[0], zp_tensor};
    ov::TensorVector sub_outputs{ov::Tensor(element::f32, Shape{5})};
    ASSERT_TRUE(sub_node->evaluate(sub_outputs, sub_inputs));

    auto subtracted = sub_outputs[0].data<float>();
    EXPECT_FLOAT_EQ(subtracted[0], -1.0f) << "0 - 1 = -1";
    EXPECT_FLOAT_EQ(subtracted[1],  0.0f) << "1 - 1 =  0";
    EXPECT_FLOAT_EQ(subtracted[2],  1.0f) << "2 - 1 = +1";
    EXPECT_FLOAT_EQ(subtracted[3], -1.0f) << "0 - 1 = -1";
    EXPECT_FLOAT_EQ(subtracted[4],  1.0f) << "2 - 1 = +1";

    // Step 4: Multiply by scale (scale=0.5)
    // This applies the per-row/per-group dequantization scale
    // Expected: {-1*0.5, 0*0.5, 1*0.5, -1*0.5, 1*0.5} = {-0.5, 0, 0.5, -0.5, 0.5}
    auto mul_node = std::make_shared<op::v1::Multiply>(sub_node,
        op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{0.5f}));
    ov::Tensor scale_tensor(element::f32, Shape{});
    *scale_tensor.data<float>() = 0.5f;
    ov::TensorVector mul_inputs{sub_outputs[0], scale_tensor};
    ov::TensorVector mul_outputs{ov::Tensor(element::f32, Shape{5})};
    ASSERT_TRUE(mul_node->evaluate(mul_outputs, mul_inputs));

    auto result = mul_outputs[0].data<float>();
    EXPECT_FLOAT_EQ(result[0], -0.5f) << "Final dequantized weight: -0.5";
    EXPECT_FLOAT_EQ(result[1],  0.0f) << "Final dequantized weight:  0.0";
    EXPECT_FLOAT_EQ(result[2],  0.5f) << "Final dequantized weight: +0.5";
    EXPECT_FLOAT_EQ(result[3], -0.5f) << "Final dequantized weight: -0.5";
    EXPECT_FLOAT_EQ(result[4],  0.5f) << "Final dequantized weight: +0.5";
}

// --- Serialization test ---
// Dumps the dequant subgraph to .xml/.bin files for inspection.
// The .xml shows the IR graph structure with element_type="ut1_5".
// The .bin contains the raw base-3 packed weight bytes + scale constant.
//
// Uses ov::pass::Serialize (src/core/src/pass/serialize.cpp).
// The serializer calls get_data_ptr() on the Constant to write raw bytes to .bin,
// and emits element_type="ut1_5" in the .xml attribute.

TEST(ut1_5_serialize, dump_dequant_subgraph) {
    const std::string out_dir = "C:/Users/asingh13/OneDrive - Intel Corporation/Documents/workspace/client_ai/Sightings/binary-log3";
    const std::string xml_path = out_dir + "/ut1_5_dequant_test.xml";
    const std::string bin_path = out_dir + "/ut1_5_dequant_test.bin";

    // Build a small dequant subgraph: Constant(ut1_5) → Convert → Subtract(1) → Multiply(scale)
    // Shape [5, 3] = 15 trits = 3 bytes packed
    std::vector<int8_t> weight_data{
        2, 0, 1, 2, 1,   // row 0: byte = 146
        0, 1, 2, 0, 2,   // row 1: byte = 183
        1, 1, 1, 1, 1    // row 2: byte = 121
    };
    auto weights = op::v0::Constant::create(element::ut1_5, Shape{5, 3}, weight_data);
    weights->set_friendly_name("ternary_weights");

    auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
    convert->set_friendly_name("convert_to_f32");

    auto zp = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.0f});
    zp->set_friendly_name("zero_point");
    auto sub = std::make_shared<op::v1::Subtract>(convert, zp);
    sub->set_friendly_name("subtract_zp");

    auto scale = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{0.25f});
    scale->set_friendly_name("scale");
    auto mul = std::make_shared<op::v1::Multiply>(sub, scale);
    mul->set_friendly_name("multiply_scale");

    auto model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{});
    model->set_friendly_name("ut1_5_dequant_test");

    // Serialize to disk
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);

    // Verify files were created
    std::ifstream xml_file(xml_path);
    ASSERT_TRUE(xml_file.good()) << "Failed to create " << xml_path;
    std::ifstream bin_file(bin_path, std::ios::binary);
    ASSERT_TRUE(bin_file.good()) << "Failed to create " << bin_path;

    // Verify .bin size: 3 bytes (ut1_5 weights) + 4 bytes (f32 zp) + 4 bytes (f32 scale) = 11 bytes
    bin_file.seekg(0, std::ios::end);
    auto bin_size = bin_file.tellg();
    EXPECT_EQ(bin_size, 11) << ".bin should contain 3 (weights) + 4 (zp) + 4 (scale) = 11 bytes";
}

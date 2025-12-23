/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>
#include "cfu.h"
#include <cstdio>
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {
inline uint32_t PackInt8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
    uint32_t packed = ( (uint32_t)static_cast<uint8_t>(v3)) | 
                      ( (uint32_t)static_cast<uint8_t>(v2) << 8 ) | 
                      ( (uint32_t)static_cast<uint8_t>(v1) << 16 ) | 
                      ( (uint32_t)static_cast<uint8_t>(v0) << 24 );
    return packed;
}

inline unsigned int read_cycles() {
    unsigned int cycles;
    // 使用 rdcycle 讀取機器週期計數器
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
								  
 // unsigned int start = read_cycles();
  
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = -128;
  static const int32_t quantized_max = 127;


  for (int i = 0; i < 256; ++i) {
    int32_t input_val_lookup = quantized_min + i;
    const int32_t input_value = input_val_lookup - params.input_offset;
    int32_t unclamped_output;

    if (input_value >= 0) {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_identity,
                             params.output_shift_identity);
    } else {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_alpha,
                             -1);
    }
    const T clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
		cfu_op(0,8,i,static_cast<T>(clamped_output));
  }

  for (int i = 0; i < flat_size; i+=4) {
	uint32_t index = PackInt8(input_data[i],input_data[i+1],input_data[i+2],input_data[i+3]);
    *(int32_t*)(output_data + i) = cfu_op(0, 9, index, 0);
  }
  //  unsigned int end = read_cycles();
  //printf("relu Cycles: %u\n", end - start);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

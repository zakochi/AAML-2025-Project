/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include "cfu.h"
#include "perf.h"
#include <cstdio>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

namespace tflite {
namespace reference_integer_ops {



inline uint32_t PackInt8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
    uint32_t packed = ( (uint32_t)static_cast<uint8_t>(v3)) | 
                      ( (uint32_t)static_cast<uint8_t>(v2) << 8 ) | 
                      ( (uint32_t)static_cast<uint8_t>(v1) << 16 ) | 
                      ( (uint32_t)static_cast<uint8_t>(v0) << 24 );
    return packed;
}

inline unsigned int read_cycles() {
    unsigned int cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}


// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
 		// unsigned int start = read_cycles();
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  // const int stride_height = params.stride_height;
  //const int dilation_width_factor = params.dilation_width_factor;
  //const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  //const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = -128;
  const int32_t output_activation_max = 127;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  //const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
//cfu_op(1,0, input_offset, 0);
  // Check dimensions of the tensors.
  const int input_height = 1;
  const int input_width = input_shape.Dims(2);
  //const int filter_height = 1;
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  //const int groups = 1;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  //const int filters_per_group = output_depth / groups;
  //const int output_height = 1;
  // const int output_width = 148;
  
  int32_t K = filter_width * filter_input_depth;
  const int32_t N = 148; 
  const int32_t T = 4;
  const int32_t M = output_depth;
  const int32_t in_w = input_shape.Dims(2);
  const int32_t in_d = input_shape.Dims(3);
  
  cfu_op(0,1,K,input_offset);
   
  if(M == 250){
      for(int32_t row = 0; row < M; row+=128){
          int32_t m_rest = M-row;
          int32_t m_size;
          int32_t m_rest_tile;

          if(m_rest > 128){
              m_size = 128;
              m_rest_tile = 128;
          }
          else{
              m_size = 124;
              m_rest_tile = 122; 
          }
         
          cfu_op(0,10,m_size,0);

          // Weight Loading
          for(int32_t m_tile = 0; m_tile < m_size; m_tile+=T){
              int32_t new_rest = m_rest - m_tile;
              int32_t new_row = row + m_tile;
             
              const int8_t* w0_ptr = filter_data + Offset(filter_shape, new_row, 0, 0, 0);
              
              if(new_rest>3){
                  int32_t new_row1 = new_row + 1;
                  int32_t new_row2 = new_row + 2;
                  int32_t new_row3 = new_row + 3;
                  const int8_t* w1_ptr = filter_data + Offset(filter_shape, new_row1, 0, 0, 0);
                  const int8_t* w2_ptr = filter_data + Offset(filter_shape, new_row2, 0, 0, 0);
                  const int8_t* w3_ptr = filter_data + Offset(filter_shape, new_row3, 0, 0, 0);
                  cfu_op(0,20,bias_data[new_row],bias_data[new_row1]);
                  cfu_op(1,20,bias_data[new_row2],bias_data[new_row3]);
                 
                  for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                      uint32_t val_a1 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                      uint32_t val_a2 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                      cfu_op(0,2,val_a1,val_a2);
                  }
              } else {
                  int32_t new_row1 = new_row + 1;
                  const int8_t* w1_ptr = filter_data + Offset(filter_shape, new_row1, 0, 0, 0);
                  cfu_op(0,20,bias_data[new_row],bias_data[new_row1]);
                  cfu_op(1,20,0,0);

                  for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                      uint32_t val_a1 = PackInt8(*w0_ptr++, *w1_ptr++, 0, 0);
                      uint32_t val_a2 = PackInt8(*w0_ptr++, *w1_ptr++, 0, 0);
                      cfu_op(0,2,val_a1,val_a2);
                  }
              }
          }

          // Image loading + systolic
          for(int32_t col = 0; col < N; col+=T){
              int32_t in_x_org[4];
              for(int i=0; i<4; ++i) {
                  int32_t cur = col + i;
                  if (cur < 148) {
                      in_x_org[i] = cur * stride_width - pad_width;
                  } else { 
                      in_x_org[i] = -9999; 
                  }
              }
              
              int32_t ic = 0, fy = 0, fx = 0;
              const int8_t* i_ptr[4];
              bool is_pad[4];
              
              for(int i=0; i<4; ++i) {
                  int32_t in_y = fy; 
                  int32_t in_x = in_x_org[i] + fx; 
                  is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                  if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d; 
              }

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  int8_t vals[4];
                  for(int i=0; i<4; ++i) vals[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
                  uint32_t val_b1 = PackInt8(vals[0], vals[1], vals[2], vals[3]);
                  
                  ic++; 
                  if(ic == filter_input_depth) { 
                      ic=0; fx++; 
                      if(fx == filter_width) { fx=0; fy++; }
                      
                      for(int i=0; i<4; ++i) {
                          int32_t in_y = fy; 
                          int32_t in_x = in_x_org[i] + fx;
                          is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                          if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                      }
                  }
				  
                  int8_t vals_n[4];
                  for(int i=0; i<4; ++i) vals_n[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
                  uint32_t val_b2 = PackInt8(vals_n[0], vals_n[1], vals_n[2], vals_n[3]);
                 
                  ic++; 
                  if(ic == filter_input_depth) { 
                      ic=0; fx++; 
                      if(fx == filter_width) { fx=0; fy++; }
                      for(int i=0; i<4; ++i) {
                          int32_t in_y = fy; 
                          int32_t in_x = in_x_org[i] + fx;
                          is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                          if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                      }
                  }

                  cfu_op(0,3,val_b1,val_b2);
              }
              cfu_op(0,5,0,0);
          }

          // Write back
          cfu_op(0,7,0,0);
          for(int32_t t = 0; t < m_rest_tile; t++){
              int32_t out_channel = row + t;
              int32_t out_mult = output_multiplier[out_channel];
              int32_t out_shift = output_shift[out_channel];
              int8_t* out_ptr_base = output_data + out_channel;

              for(int32_t col = 0; col < N; col++){
                    int32_t val = cfu_op(0,6,t,col);
                    val = MultiplyByQuantizedMultiplier(val, out_mult, out_shift);
                    val += output_offset;
                    val = std::max(val, output_activation_min);
                    val = std::min(val, output_activation_max);
   
                    *out_ptr_base = static_cast<int8_t>(val);
                    out_ptr_base += output_depth;
              }
          }
      }     
  }

  else if(K == 8000){
      for(int32_t row = 0; row < M; row+=32){

          cfu_op(0,10,32,0);
          for(int32_t m_tile = 0; m_tile < 32; m_tile+=T){

              int32_t new_row = row + m_tile;
              const int8_t* w0_ptr = filter_data + Offset(filter_shape, new_row, 0, 0, 0);
              int32_t new_row1 = new_row + 1;
              int32_t new_row2 = new_row + 2;
              int32_t new_row3 = new_row + 3;
              const int8_t* w1_ptr = filter_data + Offset(filter_shape, new_row1, 0, 0, 0);
              const int8_t* w2_ptr = filter_data + Offset(filter_shape, new_row2, 0, 0, 0);
              const int8_t* w3_ptr = filter_data + Offset(filter_shape, new_row3, 0, 0, 0);
              cfu_op(0,20,bias_data[new_row],bias_data[new_row+1]);
              cfu_op(1,20,bias_data[new_row+2],bias_data[new_row+3]);

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  uint32_t val_a1 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                  uint32_t val_a2 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                  cfu_op(0,2,val_a1,val_a2);
              }
          }

          for(int32_t col = 0; col < N; col+=T){
              int32_t in_x_org[4];
              for(int i=0; i<4; ++i) {
                  int32_t cur = col + i;
                  if (cur < 148) in_x_org[i] = cur * stride_width - pad_width;
                  else in_x_org[i] = -9999;
              }
			  
              int32_t ic = 0, fy = 0, fx = 0;
              const int8_t* i_ptr[4];
              bool is_pad[4];

              for(int i=0; i<4; ++i) {
                  int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                  is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                  if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
              }

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  int8_t vals[4];
                  for(int i=0; i<4; ++i) vals[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
                  uint32_t val_b1 = PackInt8(vals[0], vals[1], vals[2], vals[3]);

                  ic++; 
				  if(ic == filter_input_depth) { 
                      ic=0; fx++; if(fx == filter_width) { fx=0; fy++; }
                      for(int i=0; i<4; ++i) {
                          int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                          is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                          if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                      }
                  }

                  int8_t vals_n[4];
                  for(int i=0; i<4; ++i) vals_n[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
                  uint32_t val_b2 = PackInt8(vals_n[0], vals_n[1], vals_n[2], vals_n[3]);

                  ic++; 
				  if(ic == filter_input_depth) { 
                      ic=0; fx++; if(fx == filter_width) { fx=0; fy++; }
                      for(int i=0; i<4; ++i) {
                          int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                          is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                          if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                      }
                  }
                  cfu_op(0,3,val_b1,val_b2);
              }
              cfu_op(0,5,0,0);
          }

          cfu_op(0,7,0,0);
          for(int32_t t = 0; t < 32; t++){
              int32_t out_channel = row + t;
              if (out_channel >= output_depth) continue;
              int32_t out_mult = output_multiplier[out_channel];
              int32_t out_shift = output_shift[out_channel];
              int8_t* out_ptr = output_data + out_channel;

              for(int32_t col = 0; col < N; col++){
                    int32_t val = cfu_op(0,6,t,col);
                    val = MultiplyByQuantizedMultiplier(val, out_mult, out_shift);
                    val += output_offset;
                    val = std::max(val, output_activation_min);
                    val = std::min(val, output_activation_max);
                    *out_ptr = static_cast<int8_t>(val);
                    out_ptr += output_depth;
              }
          }
      }     
  }

  else if(M == 2000){
      for(int32_t row = 0; row < M; row+=128){
          int32_t m_rest = M-row;
          int32_t m_size = (m_rest > 128) ? 128 : 80;
          cfu_op(0,10,m_size,0);
		  
          for(int32_t m_tile = 0; m_tile < m_size; m_tile+=T){
              int32_t new_row = row + m_tile;
              const int8_t* w0_ptr = filter_data + Offset(filter_shape, new_row, 0, 0, 0);
              int32_t new_row1 = new_row + 1;
              int32_t new_row2 = new_row + 2;
              int32_t new_row3 = new_row + 3;
			  
              const int8_t* w1_ptr = filter_data + Offset(filter_shape, new_row1, 0, 0, 0);
              const int8_t* w2_ptr = filter_data + Offset(filter_shape, new_row2, 0, 0, 0);
              const int8_t* w3_ptr = filter_data + Offset(filter_shape, new_row3, 0, 0, 0);
              cfu_op(0,20,bias_data[new_row],bias_data[new_row+1]);
              cfu_op(1,20,bias_data[new_row+2],bias_data[new_row+3]);

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  uint32_t val_a1 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                  uint32_t val_a2 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                  cfu_op(0,2,val_a1,val_a2);
              }
          }
          
          for(int32_t col = 0; col < N; col+=T){
              int32_t in_x_org[4];
              for(int i=0; i<4; ++i) {
                  int32_t cur = col + i;
                  if (cur < 148) in_x_org[i] = cur * stride_width - pad_width;
                  else in_x_org[i] = -9999;
              }
			  
              int32_t ic = 0, fy = 0, fx = 0;
              const int8_t* i_ptr[4];
              bool is_pad[4];

              for(int i=0; i<4; ++i) {
                  int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                  is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                  if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
              }

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  int8_t vals[4];

                  for(int i=0; i<4; ++i) vals[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
                  uint32_t val_b1 = PackInt8(vals[0], vals[1], vals[2], vals[3]);

                  ic++; if(ic == filter_input_depth) { 
                      ic=0; fx++; if(fx == filter_width) { fx=0; fy++; }
                      for(int i=0; i<4; ++i) {
                          int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                          is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                          if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                      }
                  }

                  int8_t vals_n[4];
                  for(int i=0; i<4; ++i) vals_n[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
                  uint32_t val_b2 = PackInt8(vals_n[0], vals_n[1], vals_n[2], vals_n[3]);

                  ic++; 
				  if(ic == filter_input_depth) { 
                      ic=0; fx++; if(fx == filter_width) { fx=0; fy++; }
                      for(int i=0; i<4; ++i) {
                          int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                          is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                          if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                      }
                  }
                  cfu_op(0,3,val_b1,val_b2);
              }
              cfu_op(0,5,0,0);
          }
          cfu_op(0,7,0,0);

          for(int32_t t = 0; t < m_size; t++){
              int32_t out_channel = row + t;
              int32_t out_mult = output_multiplier[out_channel];
              int32_t out_shift = output_shift[out_channel];
              int8_t* out_ptr = output_data + out_channel;

              for(int32_t col = 0; col < 148; col++){
                    int32_t val = cfu_op(0,6,t,col);
                    val = MultiplyByQuantizedMultiplier(val, out_mult, out_shift);
                    val += output_offset;
                    val = std::max(val, output_activation_min);
                    val = std::min(val, output_activation_max);
                    *out_ptr = static_cast<int8_t>(val);
                    out_ptr += output_depth;
              }
          }
      }     
  }
  else{
      int32_t m_rest = 29;
      cfu_op(0,10,32,0);

      for(int32_t m_tile = 0; m_tile < 32; m_tile+=T){
          int32_t new_row = m_tile;
          int32_t new_rest = m_rest - m_tile;
          const int8_t* w0_ptr = filter_data + Offset(filter_shape, new_row, 0, 0, 0);

          if(new_rest>3){
              const int8_t* w1_ptr = filter_data + Offset(filter_shape, new_row + 1, 0, 0, 0);
              const int8_t* w2_ptr = filter_data + Offset(filter_shape, new_row + 2, 0, 0, 0);
              const int8_t* w3_ptr = filter_data + Offset(filter_shape, new_row + 3, 0, 0, 0);

              cfu_op(0,20,bias_data[new_row],bias_data[new_row+1]);
              cfu_op(1,20,bias_data[new_row+2],bias_data[new_row+3]);

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  uint32_t val_a1 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                  uint32_t val_a2 = PackInt8(*w0_ptr++, *w1_ptr++, *w2_ptr++, *w3_ptr++);
                  cfu_op(0,2,val_a1,val_a2);
              }
          }

          else{

              cfu_op(0,20,bias_data[new_row],0);
              cfu_op(1,20,0,0);

              for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
                  uint32_t val_a1 = PackInt8(*w0_ptr++, 0, 0, 0);
                  uint32_t val_a2 = PackInt8(*w0_ptr++, 0, 0, 0);
                  cfu_op(0,2,val_a1,val_a2);
              }
          }
      }

      for(int32_t col = 0; col < 148; col+=T){
          int32_t in_x_org[4];
          for(int i=0; i<4; ++i) {
              int32_t cur = col + i;
              if (cur < 148) in_x_org[i] = cur * stride_width - pad_width;
              else in_x_org[i] = -9999;
          }

          int32_t ic = 0, fy = 0, fx = 0;
          const int8_t* i_ptr[4];
          bool is_pad[4];

          for(int i=0; i<4; ++i) {
              int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
              is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
              if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
          }



          for(int32_t k_dim = 0; k_dim < K; k_dim+=2){
              int8_t vals[4];

              for(int i=0; i<4; ++i) vals[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
              uint32_t val_b1 = PackInt8(vals[0], vals[1], vals[2], vals[3]);

              ic++; 
			  if(ic == filter_input_depth) { 

                  ic=0; fx++; if(fx == filter_width) { fx=0; fy++; }
                  for(int i=0; i<4; ++i) {
                      int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                      is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                      if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                  }
              }

              int8_t vals_n[4];

              for(int i=0; i<4; ++i) vals_n[i] = is_pad[i] ? -input_offset : *i_ptr[i]++;
              uint32_t val_b2 = PackInt8(vals_n[0], vals_n[1], vals_n[2], vals_n[3]);             

              ic++; 
			  if(ic == filter_input_depth) { 
                  ic=0; fx++; if(fx == filter_width) { fx=0; fy++; }
                  for(int i=0; i<4; ++i) {
                      int32_t in_y = fy; int32_t in_x = in_x_org[i] + fx;
                      is_pad[i] = !(in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
                      if(!is_pad[i]) i_ptr[i] = input_data + (in_y * in_w + in_x) * in_d;
                  }
              }
              cfu_op(0,3,val_b1,val_b2);
          }
          cfu_op(0,5,0,0);
      }

      cfu_op(0,7,0,0);

      for(int32_t t = 0; t < 29; t++){
          int32_t out_channel = t;
          int32_t out_mult = output_multiplier[out_channel];
          int32_t out_shift = output_shift[out_channel];
          int8_t* out_ptr = output_data + out_channel;

          for(int32_t col = 0; col < 148; col++){
                int32_t val = cfu_op(0,6,t,col);
                val = MultiplyByQuantizedMultiplier(val, out_mult, out_shift);
                val += output_offset;
                val = std::max(val, output_activation_min);
                val = std::min(val, output_activation_max);
                *out_ptr = static_cast<int8_t>(val);
                out_ptr += output_depth;
          }
      }
  }

  // printf("input_offset = %ld, k_dim = %ld, M = %ld, N = %ld \n",input_offset,K,M,N);
  // unsigned int end = read_cycles();
  // printf("conv total Cycles: %u\n", end - start);
}















inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }const int32_t N = output_height * output_width;
	const int32_t M = output_depth;
	const int32_t k_dim = filter_height * filter_width * filter_input_depth;
	printf("k_dim = %ld, M = %ld, N = %ld \n",k_dim,M,N);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

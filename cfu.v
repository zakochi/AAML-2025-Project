// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "./TPU.v"
`include "./systolic_array.v"
`include "LUT.v"
module Cfu (
  input               cmd_valid,
  output     reg   cmd_ready,
  input      [9:0]    cmd_payload_function_id, // func3 func7
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg             rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);
	
	// CFU fundamentalnessage
	wire[2:0] func3 = cmd_payload_function_id[2:0]; // conv mode or relu mode
	wire[6:0] func7 = cmd_payload_function_id[9:3]; // operation
	
	// BUFFER & TPU
	reg A_wr_en, B0_wr_en, B1_wr_en; //B0 func3==1, B1 func3 == 2
	wire C_wr_en;
	reg [31:0] A_data_in; 
	wire [31:0] A_data_out;
	reg [31:0] B_data_in;
	wire [31:0] B0_data_out, B1_data_out;
	wire [31:0] TPU_B_DATA,TPU_A_DATA;
	wire [127:0] C_data_in0, C_data_in1, C_data_in2, C_data_in3;
	wire [127:0] C0_data_out, C1_data_out, C2_data_out, C3_data_out;
	wire [31:0]A_index,B_index;
	wire [31:0] AR_idx, BR_idx;
	wire [31:0] CR_idx;
	reg [31:0] C_index; //write index
	reg in_valid;
	reg [31:0]C_row_ofs;
	reg [31:0]K;
	reg [31:0]M;// 32/4 = 8 =>故最多左移3 ,外部應當padding好如果不為4的倍數
	wire [6:0]M_ROUND = M[8:2];
	reg [8:0]inputoffset;
	reg B_IN_LRU;
	reg [31:0]CFU_AIDX;
	reg [31:0]CFU_BIDX;
	reg CFU_AIDX_PLUS;
	reg CFU_BIDX_PLUS;
	
	reg [31:0]data_temp;

	reg C_IDX_RESET;
	wire busy;
	
	reg [127:0]C_buf_out;
	reg [31:0]C_data_out;
	
	reg [31:0]c_idx_temp;
	reg [3:0]c_row_temp;
	
    // Relu

	reg        lut_wen;
	reg [31:0] lut_out;
	reg [7:0] lut_idx0;
	reg [7:0] lut_idx1;
	reg [7:0] lut_idx2;
	reg [7:0] lut_idx3;
	reg [7:0] lut_data_in;
	wire [1:0] lut_ofs = cmd_payload_inputs_0[1:0];
	wire [7:0] quant_lut0;
	wire [7:0] quant_lut1;
	wire [7:0] quant_lut2;
	wire [7:0] quant_lut3;

	// bias
	reg [6:0]bias_addr;
	reg CFU_BIAS_PLUS;
	reg CFU_BIAS_RST;
	reg signed [13:0]bias0;
	reg signed [13:0]bias1;
	reg signed [13:0]bias2;
	reg signed [13:0]bias3;
	reg [1:0]bias_en;
	wire [6:0]bias_tpu;
	wire [13:0]bias_out0;
	wire [13:0]bias_out1;
	wire [13:0]bias_out2;
	wire [13:0]bias_out3;
	
	parameter [8:0]qunat_min = 128;
	
	LUT lut0(lut_data_in,lut_idx0,lut_wen,quant_lut0,clk);
	LUT lut1(lut_data_in,lut_idx1,lut_wen,quant_lut1,clk);
	LUT lut2(lut_data_in,lut_idx2,lut_wen,quant_lut2,clk);
	LUT lut3(lut_data_in,lut_idx3,lut_wen,quant_lut3,clk);
	
	
	always@(posedge clk)begin
		if(reset)begin
			CFU_AIDX <= 0;
		end
		else if(CFU_AIDX == (K*M_ROUND))
			CFU_AIDX <= 0;
		else if(CFU_AIDX_PLUS)begin
			CFU_AIDX <= CFU_AIDX + 1;
		end
	end
	
	always@(posedge clk)begin
		if(reset)begin
			CFU_BIDX <= 0;
		end
		else if(CFU_BIDX == K)
			CFU_BIDX <= 0;
		else if(CFU_BIDX_PLUS)begin
			CFU_BIDX <= CFU_BIDX + 1;
		end
	end
	
	always@(posedge clk)begin
		if(reset)
			C_index <= 0;
		else if(C_IDX_RESET)
			C_index <= 0;
		else if(C_wr_en)
			C_index <= C_index + 1;
	end

	always@(posedge clk)begin
		if(reset)begin
			bias_addr <= 0;
		end
		else if(CFU_BIAS_RST)
			bias_addr <= 0;
		else if(CFU_BIAS_PLUS)begin
			bias_addr <= bias_addr + 1;
		end
	end
	
	// FSM parameter
	parameter IDLE = 0;
	
	// CONV
	parameter CONV_KOFS = 1;
	parameter CONV_SETA = 2;
	parameter CONV_SETB = 3;
	parameter CONV_SETAB = 4;
	parameter CONV_CAL = 5;
	parameter CONV_GET = 6;
	parameter CONV_WAIT = 7;
	parameter CONV_SETM = 10;
	parameter CONV_SETA2 = 11;
	parameter CONV_SETB2 = 12;
	// RELU
	parameter RELU_SET = 8;
	parameter RELU_GET = 9;

	parameter BIAS_SET = 20;

	reg [6:0]cs;
	
	
	always@(posedge clk)begin
		if(reset)begin
			cs <= IDLE;
			
			cmd_ready <= 1;
			rsp_valid <= 0;
			
			{A_wr_en, B0_wr_en, B1_wr_en} <= 0;
			in_valid <= 0;
			
			C_row_ofs <= 0;

			A_data_in <= 0;
			B_data_in <= 0;
			data_temp <= 0;
			K <= 0;
			M <= 0;
			
			B_IN_LRU <= 0;
			CFU_AIDX_PLUS <= 0;
			CFU_BIDX_PLUS <= 0;
			
			C_IDX_RESET <= 0;
			
			c_idx_temp <= 0;
			
			lut_out <= 0;
			lut_idx0 <= 0;
			lut_idx1 <= 0;
			lut_idx2 <= 0;
			lut_idx3 <= 0;
			lut_wen <= 0;
			lut_data_in <= 0;
			
			bias0 <= 0;
			bias1 <= 0;
			bias2 <= 0;
			bias3 <= 0;
			CFU_BIAS_PLUS <= 0;
			CFU_BIAS_RST <= 0;
			bias_en <= 0;
		end
		else begin
			case(cs)
				IDLE : begin
					if(cmd_valid & cmd_ready)begin
						cs <= func7;
						cmd_ready <= 0;
						case(func7)
							IDLE : begin rsp_valid <= 1; end 
							CONV_KOFS  : begin K <= cmd_payload_inputs_0; inputoffset <= cmd_payload_inputs_1; rsp_valid <= 1; end
							CONV_SETM  : begin M <= cmd_payload_inputs_0; rsp_valid <= 1; end
							CONV_SETA  : begin {A_wr_en, A_data_in} <= {1'b1 , cmd_payload_inputs_0}; CFU_AIDX_PLUS <= 1; rsp_valid <= 0;data_temp<=cmd_payload_inputs_1; end
							CONV_SETB  : begin {B1_wr_en,B0_wr_en} <= {B_IN_LRU,~B_IN_LRU}; B_data_in <= cmd_payload_inputs_0;CFU_BIDX_PLUS <= 1; rsp_valid <= 0;data_temp<=cmd_payload_inputs_1; end
							CONV_SETAB : begin CFU_AIDX_PLUS <= 1;CFU_BIDX_PLUS <= 1;{B1_wr_en,B0_wr_en} = {B_IN_LRU,~B_IN_LRU}; A_wr_en <= 1; A_data_in <= cmd_payload_inputs_0; B_data_in <= cmd_payload_inputs_1; rsp_valid <= 1; end
							CONV_CAL   : if(busy)
											rsp_valid <= 0;
										 else begin
											 in_valid <= 1; rsp_valid <= 1; B_IN_LRU <= ~B_IN_LRU;
										 end
							CONV_GET   : begin
											rsp_valid <= 1; C_row_ofs <= cmd_payload_inputs_0; C_IDX_RESET <= 1;  c_idx_temp <= cmd_payload_inputs_1;CFU_BIAS_RST <= 1;
										 end
							CONV_WAIT  : begin rsp_valid <= ~busy; end
							RELU_SET : begin
								lut_wen <= 1;
								rsp_valid <= 1;
								lut_idx0 <= cmd_payload_inputs_0[7:0];
								lut_idx1 <= cmd_payload_inputs_0[7:0];
								lut_idx2 <= cmd_payload_inputs_0[7:0];
								lut_idx3 <= cmd_payload_inputs_0[7:0];
								lut_data_in <= cmd_payload_inputs_1;
							end
							RELU_GET : begin
								lut_wen <= 0;
								rsp_valid <= 1;
								lut_idx0 <= cmd_payload_inputs_0[31:24]+qunat_min;
								lut_idx1 <= cmd_payload_inputs_0[23:16]+qunat_min;
								lut_idx2 <= cmd_payload_inputs_0[15:8] +qunat_min;
								lut_idx3 <= cmd_payload_inputs_0[7:0]  +qunat_min;
							end
							BIAS_SET : begin
								rsp_valid <= 1;
								if(func3 == 1)begin
									bias2 <= cmd_payload_inputs_0;
									bias3 <= cmd_payload_inputs_1;
									CFU_BIAS_PLUS <= 1;
									bias_en <= 2'b10;
								end
								else begin
									bias0 <= cmd_payload_inputs_0;
									bias1 <= cmd_payload_inputs_1;	
									bias_en <= 2'b01;
								end
							end
						endcase			
						
					end 
					else if(rsp_valid & rsp_ready)begin
						cs <= IDLE;
						rsp_valid <= 0;
						cmd_ready <= 1;
						C_IDX_RESET <= 0;
					end
				end
				CONV_SETA : begin
					A_wr_en <= 1;
					A_data_in <= data_temp;
					CFU_AIDX_PLUS <= 1;
					cs <= CONV_SETA2;
					rsp_valid <= 1;
				end
				CONV_SETB : begin
					{B1_wr_en,B0_wr_en} <= {B_IN_LRU,~B_IN_LRU};
					B_data_in <= data_temp;
					CFU_BIDX_PLUS <= 1;
					cs <= CONV_SETB2;
					rsp_valid <= 1;
				end
				CONV_KOFS, CONV_SETA2, CONV_SETB2, CONV_SETAB, CONV_SETM, RELU_SET, RELU_GET, BIAS_SET : begin
					A_wr_en <= 0;
					B0_wr_en <= 0;
					B1_wr_en <= 0;
					bias_en <= 0;
					CFU_AIDX_PLUS <= 0;
					CFU_BIDX_PLUS <= 0;
					CFU_BIAS_PLUS <= 0;
					C_IDX_RESET <= 0;
					if(rsp_valid & rsp_ready)begin
						cs <= IDLE;
						rsp_valid <= 0;
						cmd_ready <= 1;
					end
				end
				CONV_WAIT : begin
					if(rsp_valid & rsp_ready)begin
						rsp_valid <= 0;
						cmd_ready <= 1;
						cs <= IDLE;
					end
					else begin
						if(!busy)begin
							cs <= IDLE;
							rsp_valid <= 1;
							cmd_ready <= 0;
						end
					end
				end
				CONV_CAL : begin
					if(rsp_valid & rsp_ready)begin
						rsp_valid <= 0;
						cmd_ready <= 1;
						in_valid <= 0;
						cs <= IDLE;
					end
					else begin
						if(!busy)begin
							cs <= CONV_CAL;
							rsp_valid <= 1;
							cmd_ready <= 0;
							in_valid <= 1;
							B_IN_LRU <= ~B_IN_LRU;
						end
						else begin
							cs <= CONV_CAL;
						end
					end
					
				end
				CONV_GET : begin
					C_IDX_RESET <= 0;
					CFU_BIAS_RST <= 0;
					if(rsp_valid & rsp_ready)begin
						cs <= IDLE;
						rsp_valid <= 0;
						cmd_ready <= 1;
					end	
				end
				
			endcase
		end
	end

	//這裡有問題=>存放順序的問題，計算上還有沒有問題不知道，存放方式很像n在外

	assign CR_idx =  cmd_payload_inputs_0[31:2]+(cmd_payload_inputs_1[31:2]*(M_ROUND)); //這邊邏輯要修正
	wire [1:0]c_ofs = c_idx_temp[1:0];
	
	always@(*)begin
		case(c_ofs[1:0])
			0 : C_data_out <= C_buf_out[127:96];
			1 : C_data_out <= C_buf_out[95:64];
			2 : C_data_out <= C_buf_out[63:32];
			3 : C_data_out <= C_buf_out[31:0];
		endcase
	end
	
	always@(*)begin
		case(C_row_ofs[1:0])
			0 : C_buf_out <= C0_data_out;
			1 : C_buf_out <= C1_data_out;
			2 : C_buf_out <= C2_data_out;
			3 : C_buf_out <= C3_data_out;
		endcase
	end
	
	assign rsp_payload_outputs_0 =  (cs == CONV_GET) ? C_data_out : {quant_lut3,quant_lut2,quant_lut1,quant_lut0};
	//wire [31:0]b_i = cmd_payload_inputs_0;//AR_idx;

	// TPU 
	assign TPU_A_DATA = A_data_out;
	
	
	//assign TPU_B_DATA = cs == IDLE ? (B_IN_LRU ? B1_data_out : B0_data_out) : (B_IN_LRU ? B0_data_out : B1_data_out); //這個結果是延後了1個row得到0     真的不行就改回單個B0 
	assign TPU_B_DATA = B_IN_LRU ? B0_data_out : B1_data_out; //這個結果是延後了1個col得到0  真的不行就改回單個B0 
	assign B_index = CFU_BIDX;
	assign A_index = CFU_AIDX;






  global_buffer_bram_sdp #( //M最大4然後 K=8000
	.ADDR_BITS(16), // ADDR_BITS 12 -> generates 2^12 entries
	.DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
	.clk(clk),
	.en_a(1'b1),
	.we_a(A_wr_en),
	.addr_a(A_index),
	.din_a(A_data_in),
	.en_b(1'b1),
	.addr_b(AR_idx),
	.dout_b(A_data_out)
  );


	// B buf0
	
	
	  global_buffer_bram_sdp #( //K最多可以到3542，但應該會用2048，然後拆成multibank，一個深度1024
		.ADDR_BITS(13), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_B0(
		.clk(clk),
		.en_a(1'b1),
		.we_a(B0_wr_en),
		.addr_a(B_index),
		.din_a(B_data_in),
		.en_b(1'b1),
		.addr_b(BR_idx),
		.dout_b(B0_data_out)
	  );
	  
	// b buf1
	  global_buffer_bram_sdp #( //K最多可以到3542，但應該會用2048，然後拆成multibank，一個深度1024
		.ADDR_BITS(13), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_B1(
		.clk(clk),
		.en_a(1'b1),
		.we_a(B1_wr_en),
		.addr_a(B_index),
		.din_a(B_data_in),
		.en_b(1'b1),
		.addr_b(BR_idx),
		.dout_b(B1_data_out)
	  );
	  
	  
	parameter C_SIZE = 12; //N每做完一輪取一次 => for final  C_SIZE = 6, for lab5 C_size = 
	// C buf0
	  global_buffer_bram_sdp #(
		.ADDR_BITS(C_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_C0(
		.clk(clk),
		.en_a(1'b1),
		.we_a(C_wr_en),
		.addr_a(C_index),
		.din_a(C_data_in0),
		.en_b(1'b1),
		.addr_b(CR_idx),
		.dout_b(C0_data_out)
	  );

	// C buf1
	  global_buffer_bram_sdp #(
		.ADDR_BITS(C_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_C1(
		.clk(clk),
		.en_a(1'b1),
		.we_a(C_wr_en),
		.addr_a(C_index),
		.din_a(C_data_in1),
		.en_b(1'b1),
		.addr_b(CR_idx),
		.dout_b(C1_data_out)
	  );
	  
	// C buf2
	  global_buffer_bram_sdp #(
		.ADDR_BITS(C_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_C2(
		.clk(clk),
		.en_a(1'b1),
		.we_a(C_wr_en),
		.addr_a(C_index),
		.din_a(C_data_in2),
		.en_b(1'b1),
		.addr_b(CR_idx),
		.dout_b(C2_data_out)
	  );
	  
	// C buf3
	  global_buffer_bram_sdp #(
		.ADDR_BITS(C_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_C3(
		.clk(clk),
		.en_a(1'b1),
		.we_a(C_wr_en),
		.addr_a(C_index),
		.din_a(C_data_in3),
		.en_b(1'b1),
		.addr_b(CR_idx),
		.dout_b(C3_data_out)
	  );

	// BIAS buf
	parameter BIAS_SIZE = 7;
	  global_buffer_bram_sdp #(
		.ADDR_BITS(BIAS_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(14)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_BIAS0(
		.clk(clk),
		.en_a(1'b1),
		.we_a(bias_en[0]),
		.addr_a(bias_addr),
		.din_a(bias0),
		.en_b(1'b1),
		.addr_b(bias_tpu),
		.dout_b(bias_out0)
	  );
	  global_buffer_bram_sdp #(
		.ADDR_BITS(BIAS_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(14)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_BIAS1(
		.clk(clk),
		.en_a(1'b1),
		.we_a(bias_en[0]),
		.addr_a(bias_addr),
		.din_a(bias1),
		.en_b(1'b1),
		.addr_b(bias_tpu),
		.dout_b(bias_out1)
	  );
	  global_buffer_bram_sdp #(
		.ADDR_BITS(BIAS_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(14)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_BIAS2(
		.clk(clk),
		.en_a(1'b1),
		.we_a(bias_en[1]),
		.addr_a(bias_addr),
		.din_a(bias2),
		.en_b(1'b1),
		.addr_b(bias_tpu),
		.dout_b(bias_out2)
	  );
	  global_buffer_bram_sdp #(
		.ADDR_BITS(BIAS_SIZE), // ADDR_BITS 12 -> generates 2^12 entries
		.DATA_BITS(14)  // DATA_BITS 32 -> 32 bits for each entries
	  )
	  gbuff_BIAS3(
		.clk(clk),
		.en_a(1'b1),
		.we_a(bias_en[1]),
		.addr_a(bias_addr),
		.din_a(bias3),
		.en_b(1'b1),
		.addr_b(bias_tpu),
		.dout_b(bias_out3)
	  );


TPU u_tpu(
	.clk(clk),
	.rst_n(~reset),

	.in_valid(in_valid),
	.K(K),     
	.M_ROUND(M_ROUND),
	.InputOffset(inputoffset),
	.busy(busy),

	.A_index(AR_idx),
	.A_data_out(TPU_A_DATA),
	
	.B_index(BR_idx),
	.B_data_out(TPU_B_DATA),
	
	.bias_tpu(bias_tpu),
	.bias_out0(bias_out0),
	.bias_out1(bias_out1),
	.bias_out2(bias_out2),
	.bias_out3(bias_out3),

	.C_wr_en(C_wr_en),
	.C_data_in0(C_data_in0),
	.C_data_in1(C_data_in1),
	.C_data_in2(C_data_in2),
	.C_data_in3(C_data_in3)
);  

endmodule


// BRAM def 
module global_buffer_bram_sdp #(
  parameter ADDR_BITS = 8,
  parameter DATA_BITS = 8
)(
  input                       clk,

  // ------ Write Port (A) ------
  input                       we_a,
  input                       en_a,
  input      [ADDR_BITS-1:0]  addr_a,
  input      [DATA_BITS-1:0]  din_a,

  // ------ Read Port (B) ------
  input                       en_b,
  input      [ADDR_BITS-1:0]  addr_b,
  output reg [DATA_BITS-1:0]  dout_b
);

  localparam DEPTH = 2**ADDR_BITS;

  reg [DATA_BITS-1:0] mem [0:DEPTH-1];

  // ------------------------------
  // Write Port  (Port A)
  // ------------------------------
  always @(posedge clk) begin
    if (en_a && we_a) begin
      mem[addr_a] <= din_a;
    end
  end

  // ------------------------------
  // Read Port (Port B)
  // ------------------------------
  always @(posedge clk) begin
    if (en_b) begin
      dout_b <= mem[addr_b];   // synchronous read => BRAM inference
    end
  end

endmodule

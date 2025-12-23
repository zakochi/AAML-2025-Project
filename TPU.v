module TPU(
    input clk,
    input rst_n,
	
    input            in_valid,
    input [12:0]      K,
	input [12:0]      M_ROUND,
	//input [31:0]     C_idx_base;
	input [8:0]      InputOffset,
    output reg       busy,

    output wire [18:0] A_index,
    input      [31:0] A_data_out,

    output wire [18:0] B_index,
    input      [31:0] B_data_out,

    output wire [6:0] bias_tpu,
    input signed [13:0] bias_out0,
    input signed [13:0] bias_out1,
    input signed [13:0] bias_out2,
    input signed [13:0] bias_out3,

    output reg      C_wr_en,
    //output reg [31:0] C_index, //C_index由外部管理
    output reg [127:0] C_data_in0,
	output reg [127:0] C_data_in1,
	output reg [127:0] C_data_in2,
	output reg [127:0] C_data_in3
);

    reg [12:0] k,m_round;
	reg [8:0]inputoffset;
	reg clear;
	reg [12:0]k_cnt;
	reg [12:0]m_cnt;
	reg [12:0]bias_cnt;
	reg [31:0]A,B;
	reg vld_i;
	wire [127:0] C_o[0:3];
	reg k_start_cnt,m_start_cnt,bias_start_cnt;
	reg k_rst_cnt,m_rst_cnt,bias_rst_cnt;
	reg [18:0]a_idx;
	reg [18:0]b_idx;

	reg start_c;
	reg rst_c;
	reg m_end;
	reg n_end;
	reg [31:0] m;
	
	parameter n = 4;
	
	assign A_index = a_idx;
	assign B_index = b_idx;
	assign bias_tpu = bias_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k <= 0;
            m_round <= 0;
			inputoffset <= 0;
        end else if (in_valid) begin
            k <= K;
            m_round <= M_ROUND;
			inputoffset <= InputOffset;
        end
    end
	
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_cnt <= 0;
        end else if (k_start_cnt) begin
            k_cnt <= k_cnt + 1;
        end
        else if (k_rst_cnt) begin
            k_cnt <= 0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_cnt <= 0;
        end else if (m_start_cnt) begin
            m_cnt <= m_cnt + 1;
        end
        else if (m_rst_cnt) begin
            m_cnt <= 0;
        end
    end
	
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_cnt <= 0;
        end else if (bias_start_cnt) begin
            bias_cnt <= bias_cnt + 1;
        end
        else if (bias_rst_cnt) begin
            bias_cnt <= 0;
        end
    end
	

	//要改的東西 
	//          2 BUFFER大小，並將N input改multi bank buffer
	//          3 先算n一輪，再算m一輪
	//          4 output buffer改multibank
	// m_tile==4 , 一次可存m_tile * 8000個
	// n_tile == 4, 一次存m_tile * 8000,1024+1024
	

    parameter IDLE       = 0;
    parameter LOAD_A_B   = 1;
	parameter WAIT0 = 2;
	parameter WAIT1 = 3;
	parameter WAIT2 = 4;
	parameter WAIT3 = 5;
	parameter WAIT4 = 6;
	parameter WAIT5 = 7;
    parameter COMPUTE0    = 8;
	parameter CLEAN = 9;
    reg [3:0] cs, ns;
		
	always@(posedge clk or negedge rst_n)begin
		if(!rst_n)
			cs <= IDLE;
		else
			cs <= ns;
	end
	
	always@(*)begin
		busy = 1;
		clear = 0;
		k_rst_cnt = 0;
		m_rst_cnt = 0;
		m_start_cnt = 0;
		bias_start_cnt = 0;
		bias_rst_cnt = 0;
		C_wr_en = 0;
		A = 0;
		a_idx = k_cnt + m_cnt*k;
		b_idx = k_cnt;
		B = 0;
		k_start_cnt = 0;
		C_data_in0 = 0;
		C_data_in1 = 0;
		C_data_in2 = 0;
		C_data_in3 = 0;
		
		case(cs)
			IDLE : begin
				if(in_valid)begin
					ns = LOAD_A_B;
					clear = 1;
					k_start_cnt = 1;
					m_rst_cnt = 1;
				end
				else begin
					ns = IDLE;
					busy = 0;
					clear = 1;
					k_rst_cnt = 1;
					m_rst_cnt = 1;
					bias_rst_cnt = 1;
				end
			end
			LOAD_A_B : begin
				if(k_cnt == k)begin
					ns = WAIT0;
					A = A_data_out;
					B = B_data_out;
				end
				else begin
					ns = LOAD_A_B;
					A = A_data_out;
					B = B_data_out;
					k_start_cnt = 1;
				end
			end
			WAIT0 : begin
				ns = WAIT1;
				m_start_cnt = 1;
				k_rst_cnt = 1;
			end
			WAIT1 : begin
				ns = WAIT2;
			end
			WAIT2 : begin
				ns = WAIT3;
			end
			WAIT3 : begin
				ns = WAIT4;
			end
			WAIT4 : begin
				ns = WAIT5;
			end
			WAIT5 : begin
				ns = COMPUTE0;
			end
			COMPUTE0 : begin
				ns = (m_cnt == m_round) ? IDLE : LOAD_A_B;//CLEAN;
				clear = 1;
				C_wr_en  = 1;
				C_data_in0 = C_o[0];
				C_data_in1 = C_o[1];
				C_data_in2 = C_o[2];
				C_data_in3 = C_o[3];
				k_start_cnt = 1;
				bias_start_cnt = 1;
			end
			// CLEAN : begin
				// ns = LOAD_A_B;
				// k_start_cnt = 1;
			// end
		endcase
	end



	 systolic_array u_SA(
		.clk(clk),
		.rst_n(rst_n),
		.clear(clear),         
		.inputoffset(inputoffset),
		.A_in(A),
		.B_in(B),
		.bias_out0(bias_out0),
		.bias_out1(bias_out1),
		.bias_out2(bias_out2),
		.bias_out3(bias_out3),

		.C_o0(C_o[0]),
		.C_o1(C_o[1]),
		.C_o2(C_o[2]),
		.C_o3(C_o[3])
	);
endmodule

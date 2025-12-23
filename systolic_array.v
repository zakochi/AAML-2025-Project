`include"./PE.v"
`include"./DE.v"
`include"./DE2.v"
`include"./DE3.v"

module systolic_array(
    input clk,
    input rst_n,
    input clear,         
	input [8:0]inputoffset,
	//input vld_i,
    input [31:0] A_in,
    input [31:0] B_in,
	
    input signed [13:0] bias_out0,
    input signed [13:0] bias_out1,
    input signed [13:0] bias_out2,
    input signed [13:0] bias_out3,
	
    output wire [127:0] C_o0,
	output wire [127:0] C_o1,
	output wire [127:0] C_o2,
	output wire [127:0] C_o3
);

	wire [31:0] C_out[0:3][0:3];
	wire [7:0] A_out[0:3][0:3];
	wire [7:0] B_out[0:3][0:3];
	//reg [31:0] C_out[0:3][0:3];
	
	wire [7:0]A_i[0:3];
	wire [7:0]B_i[0:3];
	
	PE u_pe00(clk,rst_n,clear,A_i[0],B_i[0],inputoffset,A_out[0][0],B_out[0][0],C_out[0][0]);
	PE u_pe01(clk,rst_n,clear,A_out[0][0],B_i[1],inputoffset,A_out[0][1],B_out[0][1],C_out[0][1]);
	PE u_pe02(clk,rst_n,clear,A_out[0][1],B_i[2],inputoffset,A_out[0][2],B_out[0][2],C_out[0][2]);
	PE u_pe03(clk,rst_n,clear,A_out[0][2],B_i[3],inputoffset,A_out[0][3],B_out[0][3],C_out[0][3]);
							 
	PE u_pe10(clk,rst_n,clear,A_i[1],B_out[0][0],inputoffset,A_out[1][0],B_out[1][0],C_out[1][0]);
	PE u_pe11(clk,rst_n,clear,A_out[1][0],B_out[0][1],inputoffset,A_out[1][1],B_out[1][1],C_out[1][1]);
	PE u_pe12(clk,rst_n,clear,A_out[1][1],B_out[0][2],inputoffset,A_out[1][2],B_out[1][2],C_out[1][2]);
	PE u_pe13(clk,rst_n,clear,A_out[1][2],B_out[0][3],inputoffset,A_out[1][3],B_out[1][3],C_out[1][3]);
							 
	PE u_pe20(clk,rst_n,clear,A_i[2],B_out[1][0],inputoffset,A_out[2][0],B_out[2][0],C_out[2][0]);
	PE u_pe21(clk,rst_n,clear,A_out[2][0],B_out[1][1],inputoffset,A_out[2][1],B_out[2][1],C_out[2][1]);
	PE u_pe22(clk,rst_n,clear,A_out[2][1],B_out[1][2],inputoffset,A_out[2][2],B_out[2][2],C_out[2][2]);
	PE u_pe23(clk,rst_n,clear,A_out[2][2],B_out[1][3],inputoffset,A_out[2][3],B_out[2][3],C_out[2][3]);
							 
	PE u_pe30(clk,rst_n,clear,A_i[3],B_out[2][0],inputoffset,A_out[3][0],B_out[3][0],C_out[3][0]);
	PE u_pe31(clk,rst_n,clear,A_out[3][0],B_out[2][1],inputoffset,A_out[3][1],B_out[3][1],C_out[3][1]);
	PE u_pe32(clk,rst_n,clear,A_out[3][1],B_out[2][2],inputoffset,A_out[3][2],B_out[3][2],C_out[3][2]);
	PE u_pe33(clk,rst_n,clear,A_out[3][2],B_out[2][3],inputoffset,A_out[3][3],B_out[3][3],C_out[3][3]);
	
	//SA fifo
	assign A_i[0] =  A_in[31:24];
	assign B_i[0] =  B_in[31:24];
	DE u_de0(clk,rst_n,A_in[23:16],B_in[23:16],A_i[1],B_i[1]);
	DE2 u_de1(clk,rst_n,A_in[15:8],B_in[15:8],A_i[2],B_i[2]);
	DE3 u_de2(clk,rst_n,A_in[7:0],B_in[7:0],A_i[3],B_i[3]);
	
	wire signed [31:0]c00,c01,c02,c03; 
	wire signed [31:0]c10,c11,c12,c13; 
	wire signed [31:0]c20,c21,c22,c23; 
	wire signed [31:0]c30,c31,c32,c33; 
	
	assign c00 = $signed(C_out[0][0])+bias_out0;
	assign c01 = $signed(C_out[0][1])+bias_out0;
	assign c02 = $signed(C_out[0][2])+bias_out0;
	assign c03 = $signed(C_out[0][3])+bias_out0;

	assign c10 = $signed(C_out[1][0])+bias_out1;
	assign c11 = $signed(C_out[1][1])+bias_out1;
	assign c12 = $signed(C_out[1][2])+bias_out1;
	assign c13 = $signed(C_out[1][3])+bias_out1;

	assign c20 = $signed(C_out[2][0])+bias_out2;
	assign c21 = $signed(C_out[2][1])+bias_out2;
	assign c22 = $signed(C_out[2][2])+bias_out2;
	assign c23 = $signed(C_out[2][3])+bias_out2;
	
	assign c30 = $signed(C_out[3][0])+bias_out3;
	assign c31 = $signed(C_out[3][1])+bias_out3;
	assign c32 = $signed(C_out[3][2])+bias_out3;
	assign c33 = $signed(C_out[3][3])+bias_out3;

	assign C_o0 = {c00,c01,c02,c03};
	assign C_o1 = {c10,c11,c12,c13};
	assign C_o2 = {c20,c21,c22,c23};
	assign C_o3 = {c30,c31,c32,c33};
	
endmodule
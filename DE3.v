module DE3(
	input clk,
	input rst_n,
	
	input  [7:0]    A,
	input  [7:0]    B,
	
	output wire [7:0]    A_o,
	output wire [7:0]    B_o
);
	wire [7:0]A_o0,B_o0;
	
	DE u_DE3(clk,rst_n,A,B,A_o0,B_o0);
	DE2 u_DE4(clk,rst_n,A_o0,B_o0,A_o,B_o);
	
endmodule
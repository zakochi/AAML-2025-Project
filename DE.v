module DE(
	input clk,
	input rst_n,
	
	input  [7:0]    A,
	input  [7:0]    B,
	
	output reg [7:0]    A_o,
	output reg [7:0]    B_o
);

	
	always@(posedge clk or negedge rst_n)begin
		if(!rst_n)begin
			A_o <= 0;
			B_o <= 0;
		end
		else begin
			A_o <= A;
			B_o <= B;
		end
	end
endmodule
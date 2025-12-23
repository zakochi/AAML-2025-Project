
module PE(
	input clk,
	input rst_n,
	input clear,

	//input [7:0] id,
	//input vld_i,
	input signed  [7:0]    A_in,
	input signed  [7:0]    B_in,
	input signed  [8:0]    Offset,
	//input  [31:0]   C_in,
	
	output  reg [7:0]    A_out,
	output  reg [7:0]    B_out,
	output  reg signed [31:0]    C_out
);

	always@(posedge clk or negedge rst_n)begin
		if(!rst_n)begin
			A_out <= 0;
			B_out <= 0;
			C_out <= 0;
		end
		else begin
			if(clear)begin
				A_out <= 0;
				B_out <= 0;
				C_out <= 0;		
			end
			else begin
				A_out <= A_in;
				B_out <= B_in;
				//C_out <= C_in + A_in*B_in;
				C_out <= C_out + A_in*(B_in + Offset);
			end
		end
	end
endmodule
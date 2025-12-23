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
module LUT (
  input      [7:0]   data_in,
  input      [7:0]    idx,
  input               wen,
  output     [7:0]   data_out,
  //input               reset,
  input               clk
);
	
	reg [7:0] result[0:255];
	
	assign data_out = result[idx];
	
	always@(posedge clk)begin
		if(wen)
			result[idx] <= data_in;
	end
		
	


endmodule
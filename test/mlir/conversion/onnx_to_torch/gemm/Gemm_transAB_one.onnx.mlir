//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module  {
  func @main_graph(%arg0: tensor<?x15xf32>) -> tensor<?x1xf32> attributes {input_names = ["dense_input"], output_names = ["dense_1"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<15x8xf32>} : () -> tensor<15x8xf32>
    %1 = "onnx.Constant"() {value = dense<[0.00852977391, 0.168265477, 0.00649423245, 0.169964984, 0.00856472272, 0.0122008752, 0.169504166, 0.169468462]> : tensor<8xf32>} : () -> tensor<8xf32>
//CHECK: %[[AVAL:.*]] = torch.constant.int 0
//CHECK: %[[BVAL:.*]] = torch.constant.int 1
//CHECK: torch.aten.transpose.int %arg0, %[[AVAL]], %[[BVAL]] : !torch.vtensor<[?,15],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,8],f32>
    %2 = "onnx.Gemm"(%arg0, %0, %1) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x15xf32>, tensor<15x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
//CHECK: torch.aten.bmm %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[?,8],f32>, !torch.vtensor<[?,8],f32> -> !torch.vtensor<[?,8],f32> 
    %3 = "onnx.Constant"() {value = dense<[[0.00501655508], [0.159436852], [0.00367315393], [0.537981451], [0.00364905875], [0.00456042029], [0.323497862], [0.373646796]]> : tensor<8x1xf32>} : () -> tensor<8x1xf32>
    %4 = "onnx.Constant"() {value = dense<0.17070578> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: %[[AVAL:int0_1]] = torch.constant.int 0
//CHECK: %[[BVAL:int1_2]] = torch.constant.int 1
//CHECK: torch.aten.transpose.int %{{[^,]*}}, %[[AVAL]], %[[BVAL]] : !torch.vtensor<[?,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[8,8],f32>
    %5 = "onnx.Gemm"(%2, %3, %4) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<?x8xf32>, tensor<8x1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
//CHECK: torch.aten.bmm %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[8,8],f32>, !torch.vtensor<[8,8],f32> -> !torch.vtensor<[8,8],f32>  
  return %5 : tensor<?x1xf32>
  }
}

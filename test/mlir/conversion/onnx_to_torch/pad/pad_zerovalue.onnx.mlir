//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x16x12xf32> attributes {input_names = ["input.1"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 2, 2, 0, 0, 2, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
    %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: %[[DIM1:.*]] = torch.constant.int 2
//CHECK: %[[DIM2:int2_0]] = torch.constant.int 2
//CHECK: %[[DIM3:int2_1]] = torch.constant.int 2
//CHECK: %[[DIM4:int2_2]] = torch.constant.int 2
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 0.000000e+00    
//CHECK: [[RES1:%.]] = torch.prim.ListConstruct %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[DIM4]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
//CHECK: [[RES2:%.]] = torch.aten.constant_pad_nd %arg0, [[RES1]], [[ALPHA]] : !torch.vtensor<[1,1,5,5],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,1,9,9],f32>
%2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<1x1x5x5xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x9x9xf32>
    %3 = "onnx.Constant"() {value = dense<[0, 0, 3, 1, 0, 0, 4, 2]> : tensor<8xi64>} : () -> tensor<8xi64>
    %4 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
//CHECK: %[[DIM5:.*]] = torch.constant.int 1
//CHECK: %[[DIM6:int2_3]] = torch.constant.int 2
//CHECK: %[[DIM7:.*]] = torch.constant.int 3
//CHECK: %[[DIM8:.*]] = torch.constant.int 4
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 0.000000e+00
//CHECK: [[RES3:%.]] = torch.prim.ListConstruct %[[DIM5]], %[[DIM6]], %[[DIM7]], %[[DIM8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
//CHECK: [[RES4:%.]] = torch.aten.constant_pad_nd [[RES2]], [[RES3]], [[ALPHA]] : !torch.vtensor<[1,1,9,9],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[1,1,16,12],f32>
    %5 = "onnx.Pad"(%2, %3, %4) {mode = "constant"} : (tensor<1x1x9x9xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<1x1x16x12xf32>
    return %5 : tensor<1x1x16x12xf32>
  }
}

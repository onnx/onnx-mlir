module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @main_graph(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[1,4],f32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
    %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %1 = torch.aten.linear %arg0, %arg1, %0 : !torch.vtensor<[1,5],f32>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
    return %1 : !torch.vtensor<[1,4],f32>
  }
}

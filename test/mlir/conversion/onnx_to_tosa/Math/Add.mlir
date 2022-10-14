func.func @test_add_int(%arg0: tensor<10x10xi32>, %arg1: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  "func.return"(%0) : (tensor<10x10xi32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xi32>, [[PARAM_1_:%.+]]: tensor<10x10xi32>) -> tensor<10x10xi32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
}
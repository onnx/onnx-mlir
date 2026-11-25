// Baseline for onnx_recompose_scatter_elements_to_nd.mlir (file-mode grounding,
// see docs/GroundedLitTests.md): the same functions, with each index built by
// onnx.Tile instead of onnx.Expand. Tile computes the same index tensor, but the
// ScatterElements-to-ScatterND recompose only matches an Expand, so these stay
// ScatterElements and lower through the element-by-element path. Repeats: the
// Expand's shape on dims where x has size 1, 1 elsewhere.

// -----

// CHECK-LABEL: func.func @test_scatter_row_add_3d
func.func @test_scatter_row_add_3d(%arg0: tensor<?x8x12xf32> {onnx.name = "data"}, %arg1: tensor<?x1x1xi64> {onnx.name = "x"}, %arg2: tensor<?x8x12xf32> {onnx.name = "upd"}) -> (tensor<?x8x12xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 8, 12]> : tensor<3xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<?x1x1xi64>, tensor<3xi64>) -> tensor<?x8x12xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<?x8x12xf32>, tensor<?x8x12xi64>, tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
  onnx.Return %1 : tensor<?x8x12xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_max_2d
func.func @test_scatter_row_max_2d(%arg0: tensor<6x8xf32> {onnx.name = "data"}, %arg1: tensor<7x1xi64> {onnx.name = "x"}, %arg2: tensor<7x8xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 8]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<7x1xi64>, tensor<2xi64>) -> tensor<7x8xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "max"} : (tensor<6x8xf32>, tensor<7x8xi64>, tensor<7x8xf32>) -> tensor<6x8xf32>
  onnx.Return %1 : tensor<6x8xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_min_neg_axis
func.func @test_scatter_row_min_neg_axis(%arg0: tensor<?x8xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "x"}, %arg2: tensor<?x8xf32> {onnx.name = "upd"}) -> (tensor<?x8xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 8]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x8xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = -2 : si64, reduction = "min"} : (tensor<?x8xf32>, tensor<?x8xi64>, tensor<?x8xf32>) -> tensor<?x8xf32>
  onnx.Return %1 : tensor<?x8xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_none_perm
func.func @test_scatter_row_none_perm(%arg0: tensor<4x2x3xf32> {onnx.name = "data"}, %arg1: tensor<4x2x3xf32> {onnx.name = "upd"}) -> (tensor<4x2x3xf32> {onnx.name = "y"}) {
  %x = onnx.Constant dense<[[[3]], [[0]], [[2]], [[1]]]> : tensor<4x1x1xi64>
  %shape = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
  %0 = "onnx.Tile"(%x, %shape) : (tensor<4x1x1xi64>, tensor<3xi64>) -> tensor<4x2x3xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg1) {axis = 0 : si64, reduction = "none"} : (tensor<4x2x3xf32>, tensor<4x2x3xi64>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
  onnx.Return %1 : tensor<4x2x3xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_broadcast_rows
func.func @test_scatter_row_broadcast_rows(%arg0: tensor<6x3xf32> {onnx.name = "data"}, %arg1: tensor<1x1xi64> {onnx.name = "x"}, %arg2: tensor<5x3xf32> {onnx.name = "upd"}) -> (tensor<6x3xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[5, 3]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<5x3xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<6x3xf32>, tensor<5x3xi64>, tensor<5x3xf32>) -> tensor<6x3xf32>
  onnx.Return %1 : tensor<6x3xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_reject_axis1
func.func @test_scatter_row_reject_axis1(%arg0: tensor<3x6xf32> {onnx.name = "data"}, %arg1: tensor<1x4xi64> {onnx.name = "x"}, %arg2: tensor<3x4xf32> {onnx.name = "upd"}) -> (tensor<3x6xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[3, 1]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<1x4xi64>, tensor<2xi64>) -> tensor<3x4xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 1 : si64, reduction = "add"} : (tensor<3x6xf32>, tensor<3x4xi64>, tensor<3x4xf32>) -> tensor<3x6xf32>
  onnx.Return %1 : tensor<3x6xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_reject_varying_index
func.func @test_scatter_row_reject_varying_index(%arg0: tensor<6x3xf32> {onnx.name = "data"}, %arg1: tensor<1x3xi64> {onnx.name = "x"}, %arg2: tensor<5x3xf32> {onnx.name = "upd"}) -> (tensor<6x3xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[5, 1]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<5x3xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<6x3xf32>, tensor<5x3xi64>, tensor<5x3xf32>) -> tensor<6x3xf32>
  onnx.Return %1 : tensor<6x3xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_reject_narrow_updates
func.func @test_scatter_row_reject_narrow_updates(%arg0: tensor<6x8xf32> {onnx.name = "data"}, %arg1: tensor<4x1xi64> {onnx.name = "x"}, %arg2: tensor<4x5xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 5]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<4x1xi64>, tensor<2xi64>) -> tensor<4x5xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<6x8xf32>, tensor<4x5xi64>, tensor<4x5xf32>) -> tensor<6x8xf32>
  onnx.Return %1 : tensor<6x8xf32>


}
// -----

// CHECK-LABEL: func.func @test_scatter_row_reject_dynamic_row
func.func @test_scatter_row_reject_dynamic_row(%arg0: tensor<?x?xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "x"}, %arg2: tensor<?x?xf32> {onnx.name = "upd"}) -> (tensor<?x?xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 4]> : tensor<2xi64>
  %0 = "onnx.Tile"(%arg1, %shape) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x?xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %1 : tensor<?x?xf32>

}
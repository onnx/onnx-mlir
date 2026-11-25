// RUN: onnx-mlir-opt --recompose-onnx %s -split-input-file | FileCheck %s

// GROUND-ALL: --lower-bound=int64:-6 --upper-bound=int64:5 --rtol=1e-5 --atol=1e-5

// ScatterElements whose index is an Expand of a value constant along every row
// (one index per row, copied along the feature dims, as PyG scatter_add /
// scatter_reduce exports produce) is rewritten to ScatterND with an [E, 1]
// index: out[x[e], :] op= upd[e, :], without the expanded index tensor.
// Indices are drawn from [-6, 5] for data with 6 rows (negative ones count
// from the back), so rows repeat and the reductions see duplicate rows.
// Grounded in file mode against onnx_recompose_scatter_elements_to_nd.baseline.mlir:
// the same functions with the index built by Tile, which the pattern does not
// match, so they keep ScatterElements and its element-by-element lowering.

// GNN aggregation: dynamic rows, 3-D, reduction add. The Expand keeps dim 0 of
// x (shape 1 there); dim 0 of x and of the updates are not statically equal,
// so the [E, 1] index is expanded to the number of update rows.
// GROUND-THIS: --shape-info=0:6x8x12,1:10x1x1,2:10x8x12
func.func @test_scatter_row_add_3d(%arg0: tensor<?x8x12xf32> {onnx.name = "data"}, %arg1: tensor<?x1x1xi64> {onnx.name = "x"}, %arg2: tensor<?x8x12xf32> {onnx.name = "upd"}) -> (tensor<?x8x12xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 8, 12]> : tensor<3xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<?x1x1xi64>, tensor<3xi64>) -> tensor<?x8x12xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<?x8x12xf32>, tensor<?x8x12xi64>, tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
  onnx.Return %1 : tensor<?x8x12xf32>

// CHECK-LABEL:  func.func @test_scatter_row_add_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x8x12xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<?x1x1xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<?x8x12xf32> {onnx.name = "upd"}) -> (tensor<?x8x12xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<?x1x1xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[PARAM_2_]]) <{axis = 0 : si64}> : (tensor<?x8x12xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_0_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Expand"([[VAR_2_]], [[VAR_4_]]) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ScatterND"([[PARAM_0_]], [[VAR_5_]], [[PARAM_2_]]) <{reduction = "add"}> : (tensor<?x8x12xf32>, tensor<?x1xi64>, tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<?x8x12xf32>
// CHECK:         }

}

// -----

// Segment max of a segment softmax: 2-D, static, reduction max. Rows of x and
// of the updates are statically equal: no extra Expand.
func.func @test_scatter_row_max_2d(%arg0: tensor<6x8xf32> {onnx.name = "data"}, %arg1: tensor<7x1xi64> {onnx.name = "x"}, %arg2: tensor<7x8xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[7, 8]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<7x1xi64>, tensor<2xi64>) -> tensor<7x8xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "max"} : (tensor<6x8xf32>, tensor<7x8xi64>, tensor<7x8xf32>) -> tensor<6x8xf32>
  onnx.Return %1 : tensor<6x8xf32>

// CHECK-LABEL:  func.func @test_scatter_row_max_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<7x1xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<7x8xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<7x1xi64>, tensor<2xi64>) -> tensor<7x1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ScatterND"([[PARAM_0_]], [[VAR_1_]], [[PARAM_2_]]) <{reduction = "max"}> : (tensor<6x8xf32>, tensor<7x1xi64>, tensor<7x8xf32>) -> tensor<6x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<6x8xf32>
// CHECK:         }

}

// -----

// Reduction min, axis given as -2 (the same axis 0 for rank 2), dynamic rows.
// GROUND-THIS: --shape-info=0:6x8,1:1x1,2:1x8
func.func @test_scatter_row_min_neg_axis(%arg0: tensor<?x8xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "x"}, %arg2: tensor<?x8xf32> {onnx.name = "upd"}) -> (tensor<?x8xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 8]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x8xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = -2 : si64, reduction = "min"} : (tensor<?x8xf32>, tensor<?x8xi64>, tensor<?x8xf32>) -> tensor<?x8xf32>
  onnx.Return %1 : tensor<?x8xf32>

// CHECK-LABEL:  func.func @test_scatter_row_min_neg_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<?x1xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<?x8xf32> {onnx.name = "upd"}) -> (tensor<?x8xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[PARAM_2_]]) <{axis = 0 : si64}> : (tensor<?x8xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_0_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Expand"([[VAR_2_]], [[VAR_4_]]) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ScatterND"([[PARAM_0_]], [[VAR_5_]], [[PARAM_2_]]) <{reduction = "min"}> : (tensor<?x8xf32>, tensor<?x1xi64>, tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<?x8xf32>
// CHECK:         }

}

// -----

// Reduction none, with a constant permutation as the index (no duplicate rows,
// as the spec requires for none).
func.func @test_scatter_row_none_perm(%arg0: tensor<4x2x3xf32> {onnx.name = "data"}, %arg1: tensor<4x2x3xf32> {onnx.name = "upd"}) -> (tensor<4x2x3xf32> {onnx.name = "y"}) {
  %x = onnx.Constant dense<[[[3]], [[0]], [[2]], [[1]]]> : tensor<4x1x1xi64>
  %shape = onnx.Constant dense<[4, 2, 3]> : tensor<3xi64>
  %0 = "onnx.Expand"(%x, %shape) : (tensor<4x1x1xi64>, tensor<3xi64>) -> tensor<4x2x3xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg1) {axis = 0 : si64, reduction = "none"} : (tensor<4x2x3xf32>, tensor<4x2x3xi64>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
  onnx.Return %1 : tensor<4x2x3xf32>

// CHECK-LABEL:  func.func @test_scatter_row_none_perm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x2x3xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<4x2x3xf32> {onnx.name = "upd"}) -> (tensor<4x2x3xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}{{.}}[3]{{.}}, {{.}}[0]{{.}}, {{.}}[2]{{.}}, {{.}}[1]{{.}}{{.}}> : tensor<4x1x1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[VAR_0_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<4x1x1xi64>, tensor<2xi64>) -> tensor<4x1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ScatterND"([[PARAM_0_]], [[VAR_2_]], [[PARAM_1_]]) <{reduction = "none"}> : (tensor<4x2x3xf32>, tensor<4x1xi64>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<4x2x3xf32>
// CHECK:         }

}

// -----

// One index broadcast to all rows: x is [1, 1] and the Expand also broadcasts
// dim 0, so the [1, 1] index is expanded to the 5 update rows.
func.func @test_scatter_row_broadcast_rows(%arg0: tensor<6x3xf32> {onnx.name = "data"}, %arg1: tensor<1x1xi64> {onnx.name = "x"}, %arg2: tensor<5x3xf32> {onnx.name = "upd"}) -> (tensor<6x3xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[5, 3]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<5x3xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<6x3xf32>, tensor<5x3xi64>, tensor<5x3xf32>) -> tensor<6x3xf32>
  onnx.Return %1 : tensor<6x3xf32>

// CHECK-LABEL:  func.func @test_scatter_row_broadcast_rows
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6x3xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<1x1xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<5x3xf32> {onnx.name = "upd"}) -> (tensor<6x3xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<1x1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Dim"([[PARAM_2_]]) <{axis = 0 : si64}> : (tensor<5x3xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_0_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Expand"([[VAR_2_]], [[VAR_4_]]) : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<5x1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ScatterND"([[PARAM_0_]], [[VAR_5_]], [[PARAM_2_]]) <{reduction = "add"}> : (tensor<6x3xf32>, tensor<5x1xi64>, tensor<5x3xf32>) -> tensor<6x3xf32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<6x3xf32>
// CHECK:         }

}

// -----

// Not rewritten: axis 1.
func.func @test_scatter_row_reject_axis1(%arg0: tensor<3x6xf32> {onnx.name = "data"}, %arg1: tensor<1x4xi64> {onnx.name = "x"}, %arg2: tensor<3x4xf32> {onnx.name = "upd"}) -> (tensor<3x6xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[3, 1]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<1x4xi64>, tensor<2xi64>) -> tensor<3x4xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 1 : si64, reduction = "add"} : (tensor<3x6xf32>, tensor<3x4xi64>, tensor<3x4xf32>) -> tensor<3x6xf32>
  onnx.Return %1 : tensor<3x6xf32>

// CHECK-LABEL:  func.func @test_scatter_row_reject_axis1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<1x4xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<3x4xf32> {onnx.name = "upd"}) -> (tensor<3x6xf32> {onnx.name = "y"}) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[3, 1]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Expand"([[PARAM_1_]], [[VAR_0_]]) : (tensor<1x4xi64>, tensor<2xi64>) -> tensor<3x4xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ScatterElements"([[PARAM_0_]], [[VAR_1_]], [[PARAM_2_]]) <{axis = 1 : si64, reduction = "add"}> : (tensor<3x6xf32>, tensor<3x4xi64>, tensor<3x4xf32>) -> tensor<3x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<3x6xf32>
// CHECK:         }

}

// -----

// Not rewritten: the index varies along a row (x is not 1 on dim 1).
func.func @test_scatter_row_reject_varying_index(%arg0: tensor<6x3xf32> {onnx.name = "data"}, %arg1: tensor<1x3xi64> {onnx.name = "x"}, %arg2: tensor<5x3xf32> {onnx.name = "upd"}) -> (tensor<6x3xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[5, 1]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<5x3xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<6x3xf32>, tensor<5x3xi64>, tensor<5x3xf32>) -> tensor<6x3xf32>
  onnx.Return %1 : tensor<6x3xf32>

// CHECK-LABEL:  func.func @test_scatter_row_reject_varying_index
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6x3xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<1x3xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<5x3xf32> {onnx.name = "upd"}) -> (tensor<6x3xf32> {onnx.name = "y"}) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[5, 1]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Expand"([[PARAM_1_]], [[VAR_0_]]) : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<5x3xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ScatterElements"([[PARAM_0_]], [[VAR_1_]], [[PARAM_2_]]) <{axis = 0 : si64, reduction = "add"}> : (tensor<6x3xf32>, tensor<5x3xi64>, tensor<5x3xf32>) -> tensor<6x3xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<6x3xf32>
// CHECK:         }

}

// -----

// Not rewritten: the updates are narrower than the rows of data (5 of 8).
func.func @test_scatter_row_reject_narrow_updates(%arg0: tensor<6x8xf32> {onnx.name = "data"}, %arg1: tensor<4x1xi64> {onnx.name = "x"}, %arg2: tensor<4x5xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[4, 5]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<4x1xi64>, tensor<2xi64>) -> tensor<4x5xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<6x8xf32>, tensor<4x5xi64>, tensor<4x5xf32>) -> tensor<6x8xf32>
  onnx.Return %1 : tensor<6x8xf32>

// CHECK-LABEL:  func.func @test_scatter_row_reject_narrow_updates
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<4x1xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<4x5xf32> {onnx.name = "upd"}) -> (tensor<6x8xf32> {onnx.name = "y"}) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[4, 5]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Expand"([[PARAM_1_]], [[VAR_0_]]) : (tensor<4x1xi64>, tensor<2xi64>) -> tensor<4x5xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ScatterElements"([[PARAM_0_]], [[VAR_1_]], [[PARAM_2_]]) <{axis = 0 : si64, reduction = "add"}> : (tensor<6x8xf32>, tensor<4x5xi64>, tensor<4x5xf32>) -> tensor<6x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<6x8xf32>
// CHECK:         }

}

// -----

// Not rewritten: dynamic non-axis dims, which recompose cannot compare.
// GROUND-THIS: --shape-info=0:6x4,1:3x1,2:3x4
func.func @test_scatter_row_reject_dynamic_row(%arg0: tensor<?x?xf32> {onnx.name = "data"}, %arg1: tensor<?x1xi64> {onnx.name = "x"}, %arg2: tensor<?x?xf32> {onnx.name = "upd"}) -> (tensor<?x?xf32> {onnx.name = "y"}) {
  %shape = onnx.Constant dense<[1, 4]> : tensor<2xi64>
  %0 = "onnx.Expand"(%arg1, %shape) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x?xi64>
  %1 = "onnx.ScatterElements"(%arg0, %0, %arg2) {axis = 0 : si64, reduction = "add"} : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %1 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_scatter_row_reject_dynamic_row
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: tensor<?x1xi64> {onnx.name = "x"}, [[PARAM_2_:%.+]]: tensor<?x?xf32> {onnx.name = "upd"}) -> (tensor<?x?xf32> {onnx.name = "y"}) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 4]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Expand"([[PARAM_1_]], [[VAR_0_]]) : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ScatterElements"([[PARAM_0_]], [[VAR_1_]], [[PARAM_2_]]) <{axis = 0 : si64, reduction = "add"}> : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<?x?xf32>
// CHECK:         }

}


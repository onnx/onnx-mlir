// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Shape inference tests for onnx.Loop.
//
// For each test, the Loop results are declared with unranked tensor types
// (tensor<*x…>) in the source.  After shape inference the types are updated:
//
//   v_final  — propagated directly from the body's Yield operand type.
//   scan_out — body's scan-element type with M prepended as leading dim:
//                 NoneType cond     : leading dim = M (static)
//                 const-true cond   : leading dim = M (static)
//                 M = 0             : leading dim = 0 (empty tensor)
//                 dynamic M or live condition : leading dim = ? (unknown)
//===----------------------------------------------------------------------===//

// -----

// Basic case: static trip count M=3, NoneType condition.
// v_final  : tensor<i64>   (scalar — same type as body yield)
// scan_out : tensor<3xi64> (M=3 prepended to scalar element)

func.func @test_loop_shape_static_none_cond() -> (tensor<*xi64>, tensor<*xi64>) {
  %trip = onnx.Constant dense<3>   : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0>   : tensor<i64>
  %v_final, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<*xi64>, tensor<*xi64>)
  onnx.Return %v_final, %scan : tensor<*xi64>, tensor<*xi64>
}
// CHECK-LABEL: func.func @test_loop_shape_static_none_cond
// CHECK-SAME:  () -> (tensor<i64>, tensor<3xi64>) {
// CHECK:         }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<3xi64>)

// -----

// M = 0: body never executes.
// v_final  : tensor<i64>   (carries the unchanged v_initial type)
// scan_out : tensor<0xi64> (zero iterations = empty tensor, leading dim = 0)

func.func @test_loop_shape_m0_scan() -> (tensor<*xi64>, tensor<*xi64>) {
  %trip = onnx.Constant dense<0>   : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<7>   : tensor<i64>
  %v_final, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<*xi64>, tensor<*xi64>)
  onnx.Return %v_final, %scan : tensor<*xi64>, tensor<*xi64>
}
// CHECK-LABEL: func.func @test_loop_shape_m0_scan
// CHECK-SAME:  () -> (tensor<i64>, tensor<0xi64>) {
// CHECK:         }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<0xi64>)

// -----

// Dynamic M (runtime function argument): shape inference cannot determine
// the trip count statically, so the scan leading dimension is unknown (?).

func.func @test_loop_shape_dynamic_m(%trip: tensor<i64>) -> (tensor<*xi64>, tensor<*xi64>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0>   : tensor<i64>
  %v_final, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<*xi64>, tensor<*xi64>)
  onnx.Return %v_final, %scan : tensor<*xi64>, tensor<*xi64>
}
// CHECK-LABEL: func.func @test_loop_shape_dynamic_m
// CHECK-SAME:  (%arg0: tensor<i64>) -> (tensor<i64>, tensor<?xi64>) {
// CHECK:         }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<?xi64>)

// -----

// Constant-true initial condition with body always yielding constant true.
// Semantically equivalent to NoneType: shape inference gives a static
// leading dimension equal to M.

func.func @test_loop_shape_const_true_cond() -> (tensor<*xi64>, tensor<*xi64>) {
  %trip = onnx.Constant dense<3>    : tensor<i64>
  %cond = onnx.Constant dense<true> : tensor<i1>
  %init = onnx.Constant dense<0>    : tensor<i64>
  %v_final, %scan = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1>    : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true>  : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> (tensor<*xi64>, tensor<*xi64>)
  onnx.Return %v_final, %scan : tensor<*xi64>, tensor<*xi64>
}
// CHECK-LABEL: func.func @test_loop_shape_const_true_cond
// CHECK-SAME:  () -> (tensor<i64>, tensor<3xi64>) {
// CHECK:         }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> (tensor<i64>, tensor<3xi64>)

// -----

// Live condition (not a compile-time constant): shape inference cannot prove
// the loop runs exactly M times, so the scan leading dimension falls back to
// dynamic (?) even though M=3 is statically known.

func.func @test_loop_shape_live_cond(%cond: tensor<i1>) -> (tensor<*xi64>, tensor<*xi64>) {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %init = onnx.Constant dense<0> : tensor<i64>
  %v_final, %scan = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    onnx.Yield %body_cond, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> (tensor<*xi64>, tensor<*xi64>)
  onnx.Return %v_final, %scan : tensor<*xi64>, tensor<*xi64>
}
// CHECK-LABEL: func.func @test_loop_shape_live_cond
// CHECK-SAME:  (%arg0: tensor<i1>) -> (tensor<i64>, tensor<?xi64>) {
// CHECK:         }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> (tensor<i64>, tensor<?xi64>)

// -----

// Scan element is a 1-D vector tensor<4xi32> (not a scalar).
// After M=5 iterations the scan output has shape [M, D] = tensor<5x4xi32>.
// The v_final type tensor<4xi32> is propagated directly from the body yield.

func.func @test_loop_shape_2d_scan_elem() -> (tensor<*xi32>, tensor<*xi32>) {
  %trip = onnx.Constant dense<5>             : tensor<i64>
  %none = "onnx.NoValue"() {value}           : () -> none
  %init = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %v_final, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<4xi32>):
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %carried, %carried : tensor<i1>, tensor<4xi32>, tensor<4xi32>
  }) : (tensor<i64>, none, tensor<4xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  onnx.Return %v_final, %scan : tensor<*xi32>, tensor<*xi32>
}
// CHECK-LABEL: func.func @test_loop_shape_2d_scan_elem
// CHECK-SAME:  () -> (tensor<4xi32>, tensor<5x4xi32>) {
// CHECK:         }) : (tensor<i64>, none, tensor<4xi32>) -> (tensor<4xi32>, tensor<5x4xi32>)

// -----

// Multiple carried outputs and multiple scan outputs with different element
// types (i64 and f32), M=4, NoneType condition.
// Each v_final gets the type from its body yield; each scan gets M prepended.
//
//   v_final_i : tensor<i64>    v_final_f : tensor<f32>
//   scan_i    : tensor<4xi64>  scan_f    : tensor<4xf32>

func.func @test_loop_shape_multi_carried_and_scan()
    -> (tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>) {
  %trip   = onnx.Constant dense<4>   : tensor<i64>
  %none   = "onnx.NoValue"() {value} : () -> none
  %init_i = onnx.Constant dense<0>   : tensor<i64>
  %init_f = onnx.Constant dense<0.0> : tensor<f32>
  %vi, %vf, %si, %sf =
      "onnx.Loop"(%trip, %none, %init_i, %init_f) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>,
       %ci: tensor<i64>, %cf: tensor<f32>):
    %one  = onnx.Constant dense<1>   : tensor<i64>
    %onef = onnx.Constant dense<1.0> : tensor<f32>
    %ni = "onnx.Add"(%ci, %one)  : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %nf = "onnx.Add"(%cf, %onef) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %ni, %nf, %ni, %nf
        : tensor<i1>, tensor<i64>, tensor<f32>, tensor<i64>, tensor<f32>
  }) : (tensor<i64>, none, tensor<i64>, tensor<f32>)
       -> (tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>)
  onnx.Return %vi, %vf, %si, %sf
      : tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>
}
// CHECK-LABEL: func.func @test_loop_shape_multi_carried_and_scan
// CHECK-SAME:  () -> (tensor<i64>, tensor<f32>, tensor<4xi64>, tensor<4xf32>) {
// CHECK:         }) : (tensor<i64>, none, tensor<i64>, tensor<f32>)
// CHECK-SAME:        -> (tensor<i64>, tensor<f32>, tensor<4xi64>, tensor<4xf32>)

// -----

// Many carried variables AND many scan outputs: 4 of each, M = 4.
// Mirrors the constprop test but verifies that shape inference correctly
// handles the full 4-carried + 4-scan topology.
//
//   Carried (init -> update rule):
//     a (0) : counter    next_a = a + 1
//     b (1) : powers-2   next_b = b * 2
//     c (10): countdown  next_c = c + (-3)
//     d (0) : product    next_d = a * b
//
//   Scan outputs (per-iteration snapshot):
//     sa  : next_a           tensor<4xi64>
//     sb  : next_b           tensor<4xi64>
//     sc  : next_c           tensor<4xi64>
//     sab : next_a + next_b  tensor<4xi64>
//
//   All 4 v_finals -> tensor<i64>; all 4 scan_outs -> tensor<4xi64>.

func.func @test_loop_shape_many_carried_and_scans()
    -> (tensor<*xi64>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>,
        tensor<*xi64>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>) {
  %trip   = onnx.Constant dense<4>  : tensor<i64>
  %none   = "onnx.NoValue"() {value} : () -> none
  %init_a = onnx.Constant dense<0>  : tensor<i64>
  %init_b = onnx.Constant dense<1>  : tensor<i64>
  %init_c = onnx.Constant dense<10> : tensor<i64>
  %init_d = onnx.Constant dense<0>  : tensor<i64>
  %va, %vb, %vc, %vd, %sa, %sb, %sc, %sab =
      "onnx.Loop"(%trip, %none, %init_a, %init_b, %init_c, %init_d) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>,
       %a: tensor<i64>, %b: tensor<i64>,
       %c: tensor<i64>, %d: tensor<i64>):
    %one  = onnx.Constant dense<1>  : tensor<i64>
    %two  = onnx.Constant dense<2>  : tensor<i64>
    %neg3 = onnx.Constant dense<-3> : tensor<i64>
    %next_a = "onnx.Add"(%a, %one)         : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_b = "onnx.Mul"(%b, %two)         : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_c = "onnx.Add"(%c, %neg3)        : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_d = "onnx.Mul"(%a, %b)           : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %sum_ab = "onnx.Add"(%next_a, %next_b) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next_a, %next_b, %next_c, %next_d,
               %next_a, %next_b, %next_c, %sum_ab
        : tensor<i1>,
          tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
          tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>)
       -> (tensor<*xi64>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>,
           tensor<*xi64>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>)
  onnx.Return %va, %vb, %vc, %vd, %sa, %sb, %sc, %sab
      : tensor<*xi64>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>,
        tensor<*xi64>, tensor<*xi64>, tensor<*xi64>, tensor<*xi64>
}
// CHECK-LABEL: func.func @test_loop_shape_many_carried_and_scans
// CHECK-SAME:  () -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
// CHECK-SAME:         tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
// CHECK:         }) : (tensor<i64>, none, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>)
// CHECK-SAME:        -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
// CHECK-SAME:            tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)

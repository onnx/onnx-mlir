// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// LoopUnroll: constant-trip-count loops with NoneType condition are physically
// unrolled so that standard constprop can fold the resulting ops.
//
// Match conditions (see LoopUnroll in ConstProp.cpp):
//   • Loop condition operand is NoneType (loop always runs exactly M times)
//   • Trip count M is a dense scalar constant in (0, 64]
//===----------------------------------------------------------------------===//

// -----

// Simplest case: accumulate an integer sum across 3 iterations.
// Expected: onnx.Loop disappears; result folds to onnx.Constant dense<3>.

func.func @test_loop_unroll_carried_only() -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_carried_only
// CHECK-SAME:   () -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<i64>
// CHECK:         }

// -----

// Loop with one scan output: collect the running sum at each iteration.
// Trip count = 3, body: next = carried + 1, scan_elem = next.
// Expected scan output: [1, 2, 3] folds to a single constant tensor.

func.func @test_loop_unroll_scan_output() -> tensor<3xi64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %carried_out, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<3xi64>)
  onnx.Return %scan : tensor<3xi64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_scan_output
// CHECK-SAME:   () -> tensor<3xi64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3xi64>
// CHECK:         }

// -----

// Loop is NOT unrolled: condition is a live tensor<i1> value, not NoneType.
// The LoopUnroll pattern requires NoneType condition.

func.func @test_loop_no_unroll_with_condition(%cond: tensor<i1>) -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    onnx.Yield %body_cond, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_no_unroll_with_condition
// CHECK:       onnx.Loop

// -----

// Loop is NOT unrolled: trip count is a runtime value, not a compile-time
// constant.

func.func @test_loop_no_unroll_dynamic_trip(%trip: tensor<i64>) -> tensor<i64> {
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_no_unroll_dynamic_trip
// CHECK:       onnx.Loop

// -----

// NOT unrolled: M > kMaxUnrollCount (64). The pattern refuses to unroll
// excessively large trip counts to avoid IR explosion.

func.func @test_loop_no_unroll_too_large() -> tensor<i64> {
  %trip = onnx.Constant dense<65> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_no_unroll_too_large
// CHECK:       onnx.Loop

// -----

// Non-constant-foldable body: the loop still unrolls (Loop op disappears),
// but the cloned Add ops are not further folded because %arg is a runtime value.
// The initial carried value is dense<0>, so the first iteration (0 + %arg)
// is folded by Add-identity to just %arg, leaving 2 Add ops instead of 3.

func.func @test_loop_unroll_non_const_body(%arg: tensor<i64>) -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %next = "onnx.Add"(%carried, %arg) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_non_const_body
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i64>) -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<i64>
// CHECK:         }

// -----

// Constant-true initial condition + body always yields true: semantically
// equivalent to NoneType condition. The loop should be unrolled and the result
// folded to a constant.

func.func @test_loop_unroll_true_cond() -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %cond = onnx.Constant dense<true> : tensor<i1>
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_true_cond
// CHECK-SAME:   () -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<i64>
// CHECK:         }

// -----

// Constant-true initial condition with passthrough body condition: the body
// yields %body_cond unchanged (not a fresh constant). Since the initial
// condition is dense<true> and the body never modifies it, every iteration
// sees true — equivalent to NoneType. The loop should still be unrolled.

func.func @test_loop_unroll_true_cond_passthrough() -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %cond = onnx.Constant dense<true> : tensor<i1>
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    onnx.Yield %body_cond, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_true_cond_passthrough
// CHECK-SAME:   () -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<i64>
// CHECK:         }

// -----

// NOT unrolled: body yields a non-constant condition. Even though the initial
// condition is constant-true, we cannot guarantee the loop runs M times.

func.func @test_loop_no_unroll_non_const_yield_cond(%flag: tensor<i1>) -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %cond = onnx.Constant dense<true> : tensor<i1>
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    onnx.Yield %flag, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_no_unroll_non_const_yield_cond
// CHECK:       onnx.Loop

// -----

// M = 0, carried-only: the loop body never executes; the carried output is
// just the initial value passed through unchanged.

func.func @test_loop_m0_carried_only() -> tensor<i64> {
  %trip = onnx.Constant dense<0> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<42> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_m0_carried_only
// CHECK-SAME:   () -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<42> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<i64>
// CHECK:         }

// -----

// M = 0 with scan output: the loop body never executes; the carried output is
// the initial value and the scan output is an empty tensor.

func.func @test_loop_m0_with_scan() -> (tensor<i64>, tensor<?xi64>) {
  %trip = onnx.Constant dense<0> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<7> : tensor<i64>
  %carried_out, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<?xi64>)
  onnx.Return %carried_out, %scan : tensor<i64>, tensor<?xi64>
}
// Shape inference (M=0) refines the scan output to tensor<0xi64> statically.
// CHECK-LABEL:  func.func @test_loop_m0_with_scan
// CHECK-SAME:   () -> (tensor<i64>, tensor<0xi64>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<7> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<> : tensor<0xi64>
// CHECK:           onnx.Return [[VAR_0_]], [[VAR_1_]] : tensor<i64>, tensor<0xi64>
// CHECK:         }

//===----------------------------------------------------------------------===//
// Scan output: additional shape and content coverage.
//===----------------------------------------------------------------------===//

// -----

// Scan output with 2-D elements: each of 3 iterations contributes a constant
// row vector dense<[10, 20]> : tensor<2xi64>.  LoopUnroll emits three
// Unsqueeze(tensor<1x2xi64>) ops and one Concat(axis=0); subsequent constprop
// folds them into a single constant matrix.
// Expected scan output: [[10, 20], [10, 20], [10, 20]] : tensor<3x2xi64>.

func.func @test_loop_scan_2d_elem() -> tensor<3x2xi64> {
  %trip = onnx.Constant dense<3>    : tensor<i64>
  %none = "onnx.NoValue"() {value}  : () -> none
  %init = onnx.Constant dense<0>    : tensor<i64>
  %carried_out, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1>        : tensor<i64>
    %next = "onnx.Add"(%carried, %one)    : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %row  = onnx.Constant dense<[10, 20]> : tensor<2xi64>
    %true = onnx.Constant dense<true>     : tensor<i1>
    onnx.Yield %true, %next, %row : tensor<i1>, tensor<i64>, tensor<2xi64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<3x2xi64>)
  onnx.Return %scan : tensor<3x2xi64>
}
// CHECK-LABEL: func.func @test_loop_scan_2d_elem
// CHECK-SAME:  () -> tensor<3x2xi64> {
// CHECK:         [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[10, 20], [10, 20], [10, 20]{{.}}> : tensor<3x2xi64>
// CHECK:         onnx.Return [[VAR_0_]] : tensor<3x2xi64>
// CHECK:       }

// -----

// Multiple scan outputs: the body records both the running sum and the raw
// iteration counter at each step.  Two independent scan tapes are built and
// both fold to constants.
// Expected: scan_sum = [1, 2, 3], scan_iter = [0, 1, 2].

func.func @test_loop_scan_multiple() -> (tensor<3xi64>, tensor<3xi64>) {
  %trip = onnx.Constant dense<3>   : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0>   : tensor<i64>
  %carried_out, %scan_sum, %scan_iter = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true>  : tensor<i1>
    onnx.Yield %true, %next, %next, %iter
        : tensor<i1>, tensor<i64>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<3xi64>, tensor<3xi64>)
  onnx.Return %scan_sum, %scan_iter : tensor<3xi64>, tensor<3xi64>
}
// CHECK-LABEL: func.func @test_loop_scan_multiple
// CHECK-SAME:  () -> (tensor<3xi64>, tensor<3xi64>) {
// CHECK-DAG:     [[SUM:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK-DAG:     [[IDX:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK:         onnx.Return [[SUM]], [[IDX]] : tensor<3xi64>, tensor<3xi64>
// CHECK:       }

// -----

// Five simultaneous scan outputs: each tape records a different function of
// the loop state.  Trip = 4, one carried scalar (running sum starting at 0).
//
//   scan_sum    = running sum at end of each iter :  [1, 2, 3, 4]
//   scan_iter   = raw iteration counter            :  [0, 1, 2, 3]
//   scan_sq     = square of the iteration counter  :  [0, 1, 4, 9]
//   scan_double = 2 × running sum                  :  [2, 4, 6, 8]
//   scan_neg    = −1 × running sum                 :  [-1,-2,-3,-4]
//
// All five Unsqueeze+Concat chains fold completely because every contribution
// is a compile-time constant.

func.func @test_loop_scan_many_outputs()
    -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
  %trip = onnx.Constant dense<4>   : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0>   : tensor<i64>
  %carried_out, %scan_sum, %scan_iter, %scan_sq, %scan_double, %scan_neg =
      "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one    = onnx.Constant dense<1>  : tensor<i64>
    %two    = onnx.Constant dense<2>  : tensor<i64>
    %neg1   = onnx.Constant dense<-1> : tensor<i64>
    %next   = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %sq     = "onnx.Mul"(%iter, %iter)   : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %double = "onnx.Mul"(%next, %two)    : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %neg    = "onnx.Mul"(%next, %neg1)   : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true   = onnx.Constant dense<true>  : tensor<i1>
    onnx.Yield %true, %next, %next, %iter, %sq, %double, %neg
        : tensor<i1>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>)
       -> (tensor<i64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
  onnx.Return %scan_sum, %scan_iter, %scan_sq, %scan_double, %scan_neg
      : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}
// CHECK-LABEL: func.func @test_loop_scan_many_outputs
// CHECK-SAME:  () -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
// CHECK-DAG:     [[SUM:%.+]]  = onnx.Constant dense<[1, 2, 3, 4]>      : tensor<4xi64>
// CHECK-DAG:     [[IDX:%.+]]  = onnx.Constant dense<[0, 1, 2, 3]>      : tensor<4xi64>
// CHECK-DAG:     [[SQ:%.+]]   = onnx.Constant dense<[0, 1, 4, 9]>      : tensor<4xi64>
// CHECK-DAG:     [[DBL:%.+]]  = onnx.Constant dense<[2, 4, 6, 8]>      : tensor<4xi64>
// CHECK-DAG:     [[NEG:%.+]]  = onnx.Constant dense<[-1, -2, -3, -4]>  : tensor<4xi64>
// CHECK:         onnx.Return [[SUM]], [[IDX]], [[SQ]], [[DBL]], [[NEG]]
// CHECK-SAME:        : tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
// CHECK:       }

// -----

// Many carried variables AND many scan outputs simultaneously.
// Trip = 4, four v_initials, four scan outputs.
//
//   Carried variables (init → update rule):
//     a  ( 0 ) : counter   — next_a = a + 1
//     b  ( 1 ) : powers-2  — next_b = b × 2
//     c  (10 ) : countdown — next_c = c − 3
//     d  ( 0 ) : product   — next_d = a × b  (uses *input* a and b)
//
//   Scan outputs (what each tape records per iteration):
//     scan_a  : next_a            →  [1,  2,  3,   4]
//     scan_b  : next_b            →  [2,  4,  8,  16]
//     scan_c  : next_c            →  [7,  4,  1,  -2]
//     scan_ab : next_a + next_b   →  [3,  6, 11,  20]
//
//   Final carried values after 4 iters:
//     v_final_a =  4,  v_final_b = 16,  v_final_c = -2,  v_final_d = 24
//
// All eight results (4 carried + 4 scan) fold to constants.

func.func @test_loop_many_carried_and_scans()
    -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
        tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
  %trip   = onnx.Constant dense<4>  : tensor<i64>
  %none   = "onnx.NoValue"() {value} : () -> none
  %init_a = onnx.Constant dense<0>  : tensor<i64>
  %init_b = onnx.Constant dense<1>  : tensor<i64>
  %init_c = onnx.Constant dense<10> : tensor<i64>
  %init_d = onnx.Constant dense<0>  : tensor<i64>
  %v_a, %v_b, %v_c, %v_d, %scan_a, %scan_b, %scan_c, %scan_ab =
      "onnx.Loop"(%trip, %none, %init_a, %init_b, %init_c, %init_d) ({
  ^bb0(%iter : tensor<i64>, %cond : tensor<i1>,
       %a : tensor<i64>, %b : tensor<i64>,
       %c : tensor<i64>, %d : tensor<i64>):
    %one  = onnx.Constant dense<1>  : tensor<i64>
    %two  = onnx.Constant dense<2>  : tensor<i64>
    %neg3 = onnx.Constant dense<-3> : tensor<i64>
    %next_a  = "onnx.Add"(%a, %one)       : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_b  = "onnx.Mul"(%b, %two)       : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_c  = "onnx.Add"(%c, %neg3)      : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_d  = "onnx.Mul"(%a, %b)         : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %sum_ab  = "onnx.Add"(%next_a, %next_b) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next_a, %next_b, %next_c, %next_d,
               %next_a, %next_b, %next_c, %sum_ab
        : tensor<i1>,
          tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
          tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>)
       -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
           tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
  onnx.Return %v_a, %v_b, %v_c, %v_d, %scan_a, %scan_b, %scan_c, %scan_ab
      : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
        tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}
// CHECK-LABEL:  func.func @test_loop_many_carried_and_scans
// CHECK-SAME:   () -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<16> : tensor<i64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<-2> : tensor<i64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<24> : tensor<i64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<[2, 4, 8, 16]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<[7, 4, 1, -2]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<[3, 6, 11, 20]> : tensor<4xi64>
// CHECK:           onnx.Return [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]] : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
// CHECK:         }

// -----

// Partial-fold companion to test_loop_many_carried_and_scans.
// Same 4-carried + 4-scan topology, trip = 4, but two carried variables
// depend on the runtime value %arg, so their folding is impossible.
//
//   Carried (init -> update rule):
//     a  ( 0 ) : counter    — next_a = a + 1          FOLDS  (constant)
//     b  ( 1 ) : powers-2   — next_b = b * 2          FOLDS  (constant)
//     c  ( 0 ) : runtime    — next_c = c + %arg       NO FOLD (runtime)
//     d  ( 0 ) : accumulate — next_d = d + b          FOLDS  (depends only on b)
//
//   Scan outputs:
//     scan_a    : next_a              [1,2,3,4]     FOLDS
//     scan_b    : next_b              [2,4,8,16]    FOLDS
//     scan_c    : next_c              runtime tape  NO FOLD -> Unsqueeze + Concat remain
//     scan_mixed: next_a + next_c     runtime tape  NO FOLD -> Unsqueeze + Concat remain
//
//   Final carried values:
//     v_a =  4   (constant)   v_b = 16   (constant)
//     v_c = runtime           v_d = 15   (1+2+4+8, constant)
//
// The loop is still fully UNROLLED (LoopUnroll fires on static M + NoneType cond).
// The constant-foldable results collapse; the runtime-dependent scan tapes
// remain as Unsqueeze + Concat chains in the IR.

func.func @test_loop_many_partial_fold(%arg: tensor<i64>)
    -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
        tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
  %trip   = onnx.Constant dense<4>  : tensor<i64>
  %none   = "onnx.NoValue"() {value} : () -> none
  %init_a = onnx.Constant dense<0>  : tensor<i64>
  %init_b = onnx.Constant dense<1>  : tensor<i64>
  %init_c = onnx.Constant dense<0>  : tensor<i64>
  %init_d = onnx.Constant dense<0>  : tensor<i64>
  %va, %vb, %vc, %vd, %scan_a, %scan_b, %scan_c, %scan_mixed =
      "onnx.Loop"(%trip, %none, %init_a, %init_b, %init_c, %init_d) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>,
       %a: tensor<i64>, %b: tensor<i64>,
       %c: tensor<i64>, %d: tensor<i64>):
    %one    = onnx.Constant dense<1> : tensor<i64>
    %two    = onnx.Constant dense<2> : tensor<i64>
    %next_a  = "onnx.Add"(%a, %one)        : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_b  = "onnx.Mul"(%b, %two)        : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_c  = "onnx.Add"(%c, %arg)        : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %next_d  = "onnx.Add"(%d, %b)          : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %mixed   = "onnx.Add"(%next_a, %next_c) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next_a, %next_b, %next_c, %next_d,
               %next_a, %next_b, %next_c, %mixed
        : tensor<i1>,
          tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
          tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>)
       -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
           tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
  onnx.Return %va, %vb, %vc, %vd, %scan_a, %scan_b, %scan_c, %scan_mixed
      : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>,
        tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}
// CHECK-LABEL:  func.func @test_loop_many_partial_fold
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i64>) -> (tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 4, 8, 16]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<15> : tensor<i64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<16> : tensor<i64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4> : tensor<i64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<3> : tensor<i64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<2> : tensor<i64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<1> : tensor<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_8_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Add"([[VAR_10_]], [[VAR_7_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Add"([[VAR_10_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Add"([[VAR_12_]], [[VAR_6_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Add"([[VAR_12_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Add"([[VAR_14_]], [[VAR_5_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Unsqueeze"([[VAR_10_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Unsqueeze"([[VAR_12_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Unsqueeze"([[VAR_14_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Concat"([[VAR_16_]], [[VAR_17_]], [[VAR_18_]], [[VAR_19_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Unsqueeze"([[VAR_9_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Unsqueeze"([[VAR_11_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Unsqueeze"([[VAR_13_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Unsqueeze"([[VAR_15_]], [[VAR_2_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_25_:%.+]] = "onnx.Concat"([[VAR_21_]], [[VAR_22_]], [[VAR_23_]], [[VAR_24_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           onnx.Return [[VAR_5_]], [[VAR_4_]], [[VAR_14_]], [[VAR_3_]], [[VAR_1_]], [[VAR_0_]], [[VAR_20_]], [[VAR_25_]] : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
// CHECK:         }

// -----

// Scan output collects the raw iteration counter: trip = 4, the body carries
// nothing useful and yields %iter as the scan element.
// Each Unsqueeze of a constant scalar folds immediately; the resulting four
// tensor<1xi64> slices are then Concat-folded into a single constant.
// Expected scan output: [0, 1, 2, 3] : tensor<4xi64>.

func.func @test_loop_scan_iter_counter() -> tensor<4xi64> {
  %trip = onnx.Constant dense<4>   : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0>   : tensor<i64>
  %carried_out, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %carried, %iter : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<4xi64>)
  onnx.Return %scan : tensor<4xi64>
}
// CHECK-LABEL: func.func @test_loop_scan_iter_counter
// CHECK-SAME:  () -> tensor<4xi64> {
// CHECK:         [[VAR_0_:%.+]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK:         onnx.Return [[VAR_0_]] : tensor<4xi64>
// CHECK:       }

// -----

// Scan output with a non-constant body: the body adds the runtime value %arg,
// so per-iteration contributions are not constants.  The loop is still fully
// unrolled (onnx.Loop disappears) but the Unsqueeze + Concat ops that build
// the scan tape cannot be constant-folded further.
// Note: the first iteration's Add (0 + %arg) is folded by Add-identity to
// %arg directly, leaving 2 Add ops for iterations 1 and 2.

func.func @test_loop_scan_non_const(%arg: tensor<i64>) -> tensor<3xi64> {
  %trip = onnx.Constant dense<3>   : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0>   : tensor<i64>
  %carried_out, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %next = "onnx.Add"(%carried, %arg) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true>  : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<3xi64>)
  onnx.Return %scan : tensor<3xi64>
}
// CHECK-LABEL:  func.func @test_loop_scan_non_const
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<i64>) -> tensor<3xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_0_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[VAR_1_]], [[VAR_0_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Unsqueeze"([[VAR_2_]], [[VAR_0_]]) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<3xi64>
// CHECK:         }

// -----

// Nested loops: outer loop runs 2 trips, each trip runs an inner loop for
// 3 trips. Both loops have NoneType condition and constant bodies.
// Outer body: inner_result = Loop(3){ v = v + 1 }, carried = inner_result.
// Starting carried = 0; after outer iter 0: inner gives 3; after iter 1: 6.
// Expected final result: dense<6>.

func.func @test_loop_unroll_nested() -> tensor<i64> {
  %outer_trip = onnx.Constant dense<2> : tensor<i64>
  %none       = "onnx.NoValue"() {value} : () -> none
  %outer_init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%outer_trip, %none, %outer_init) ({
  ^bb0(%outer_iter: tensor<i64>, %outer_cond: tensor<i1>, %outer_carried: tensor<i64>):
    %inner_trip = onnx.Constant dense<3> : tensor<i64>
    %inner_result = "onnx.Loop"(%inner_trip, %none, %outer_carried) ({
    ^bb0(%inner_iter: tensor<i64>, %inner_cond: tensor<i1>, %inner_carried: tensor<i64>):
      %one = onnx.Constant dense<1> : tensor<i64>
      %next = "onnx.Add"(%inner_carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %true = onnx.Constant dense<true> : tensor<i1>
      onnx.Yield %true, %next : tensor<i1>, tensor<i64>
    }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %inner_result : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_nested
// CHECK-SAME:   () -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<i64>
// CHECK:         }

//===----------------------------------------------------------------------===//
// ConstPropConcatFromSequence: when a ConcatFromSequence op's input is a
// compile-time SequenceEmpty → SequenceInsert … chain with all-constant
// elements (NoneType position), fold the whole thing into a single onnx.Constant.
//===----------------------------------------------------------------------===//

// -----

// Concatenate two constant vectors along axis 0.
// Sequence: empty → insert [1,1,1] → insert [2,2,2]
// ConcatFromSequence(axis=0) → [1,1,1,2,2,2]
// Declare the output type as the expected folded shape so types are consistent.

func.func @test_constprop_concatfromseq_concat() -> tensor<6xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
  %e0   = onnx.Constant dense<1> : tensor<3xi32>
  %e1   = onnx.Constant dense<2> : tensor<3xi32>
  %seq1 = "onnx.SequenceInsert"(%seq0, %e0, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %seq2 = "onnx.SequenceInsert"(%seq1, %e1, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %result = "onnx.ConcatFromSequence"(%seq2) {axis = 0 : si64} : (!onnx.Seq<tensor<*xi32>>) -> tensor<6xi32>
  onnx.Return %result : tensor<6xi32>
}
// CHECK-LABEL:  func.func @test_constprop_concatfromseq_concat
// CHECK-SAME:   () -> tensor<6xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 1, 2, 2, 2]> : tensor<6xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<6xi32>
// CHECK:         }

// -----

// Stack two constant row vectors along a new axis, producing a 2×3 matrix.
// Sequence: empty → insert [1,2,3] → insert [4,5,6]
// ConcatFromSequence(axis=0, new_axis=1) → [[1,2,3],[4,5,6]]

func.func @test_constprop_concatfromseq_stack() -> tensor<2x3xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
  %e0   = onnx.Constant dense<[1, 2, 3]> : tensor<3xi32>
  %e1   = onnx.Constant dense<[4, 5, 6]> : tensor<3xi32>
  %seq1 = "onnx.SequenceInsert"(%seq0, %e0, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %seq2 = "onnx.SequenceInsert"(%seq1, %e1, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %result = "onnx.ConcatFromSequence"(%seq2) {axis = 0 : si64, new_axis = 1 : si64} : (!onnx.Seq<tensor<*xi32>>) -> tensor<2x3xi32>
  onnx.Return %result : tensor<2x3xi32>
}
// CHECK-LABEL:  func.func @test_constprop_concatfromseq_stack
// CHECK-SAME:   () -> tensor<2x3xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1, 2, 3], [4, 5, 6]{{.}}> : tensor<2x3xi32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x3xi32>
// CHECK:         }

// -----

// NOT folded: one sequence element is a runtime value (function argument).
// The ConstPropConcatFromSequence pattern requires all elements to be
// compile-time constants.

func.func @test_constprop_concatfromseq_no_fold(%arg: tensor<3xi32>) -> tensor<6xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
  %e1   = onnx.Constant dense<2> : tensor<3xi32>
  %seq1 = "onnx.SequenceInsert"(%seq0, %arg, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %seq2 = "onnx.SequenceInsert"(%seq1, %e1,  %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %result = "onnx.ConcatFromSequence"(%seq2) {axis = 0 : si64} : (!onnx.Seq<tensor<*xi32>>) -> tensor<6xi32>
  onnx.Return %result : tensor<6xi32>
}
// CHECK-LABEL:  func.func @test_constprop_concatfromseq_no_fold
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>) -> tensor<6xi32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<3xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
// CHECK:           [[VAR_3_:%.+]] = "onnx.SequenceInsert"([[VAR_2_]], [[PARAM_0_]], [[VAR_1_]]) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<3xi32>>
// CHECK:           [[VAR_4_:%.+]] = "onnx.SequenceInsert"([[VAR_3_]], [[VAR_0_]], [[VAR_1_]]) : (!onnx.Seq<tensor<3xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<3xi32>>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ConcatFromSequence"([[VAR_4_]]) {axis = 0 : si64, new_axis = 0 : si64} : (!onnx.Seq<tensor<3xi32>>) -> tensor<6xi32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<6xi32>
// CHECK:         }

//===----------------------------------------------------------------------===//
// LoopUnroll + SequenceInsert + ConcatFromSequence
//
// These tests verify that LoopUnroll and ConstPropConcatFromSequence cooperate
// to fold loops that carry and build sequences into a single constant.
//
// Note: constprop must run before shape-inference (the order used by this
// file's RUN line). Running shape-inference first locks the function return
// type to tensor<1xi64> (one sequence element) before the loop is unrolled;
// constprop then folds ConcatFromSequence to tensor<3xi64>, creating a type
// mismatch that the MLIR verifier rejects.
//===----------------------------------------------------------------------===//

// -----

// Loop carries a growing sequence; each of 3 iterations appends the constant
// [5]. After unrolling, the sequence chain is:
//   SequenceEmpty → Insert([5]) → Insert([5]) → Insert([5])
// ConcatFromSequence(axis=0) folds the whole thing into dense<5> : tensor<3xi64>
// (MLIR splat notation for [5, 5, 5]).

func.func @test_loop_unroll_build_seq_const() -> tensor<*xi64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 7 : si64} : () -> !onnx.Seq<tensor<*xi64>>
  %seq_final = "onnx.Loop"(%trip, %none, %seq0) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %seq: !onnx.Seq<tensor<*xi64>>):
    %elem = onnx.Constant dense<[5]> : tensor<1xi64>
    %pos  = "onnx.NoValue"() {value} : () -> none
    %next = "onnx.SequenceInsert"(%seq, %elem, %pos) : (!onnx.Seq<tensor<*xi64>>, tensor<1xi64>, none) -> !onnx.Seq<tensor<*xi64>>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, !onnx.Seq<tensor<*xi64>>
  }) : (tensor<i64>, none, !onnx.Seq<tensor<*xi64>>) -> !onnx.Seq<tensor<*xi64>>
  %result = "onnx.ConcatFromSequence"(%seq_final) {axis = 0 : si64} : (!onnx.Seq<tensor<*xi64>>) -> tensor<*xi64>
  onnx.Return %result : tensor<*xi64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_build_seq_const
// CHECK-SAME:   () -> tensor<*xi64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<5> : tensor<3xi64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3xi64>
// CHECK:         }

// -----

// Loop carries a growing sequence; each iteration i appends the unsqueezed
// iteration counter [i].  After unrolling, Unsqueeze is constant-folded for
// each cloned iteration (iter=0→[0], iter=1→[1], iter=2→[2]), giving:
//   SequenceEmpty → Insert([0]) → Insert([1]) → Insert([2])
// ConcatFromSequence(axis=0) folds to dense<[0, 1, 2]> : tensor<3xi64>.

func.func @test_loop_unroll_build_seq_iter() -> tensor<*xi64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 7 : si64} : () -> !onnx.Seq<tensor<*xi64>>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %seq_final = "onnx.Loop"(%trip, %none, %seq0) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %seq: !onnx.Seq<tensor<*xi64>>):
    %elem = "onnx.Unsqueeze"(%iter, %axes) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %pos  = "onnx.NoValue"() {value} : () -> none
    %next = "onnx.SequenceInsert"(%seq, %elem, %pos) : (!onnx.Seq<tensor<*xi64>>, tensor<1xi64>, none) -> !onnx.Seq<tensor<*xi64>>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, !onnx.Seq<tensor<*xi64>>
  }) : (tensor<i64>, none, !onnx.Seq<tensor<*xi64>>) -> !onnx.Seq<tensor<*xi64>>
  %result = "onnx.ConcatFromSequence"(%seq_final) {axis = 0 : si64} : (!onnx.Seq<tensor<*xi64>>) -> tensor<*xi64>
  onnx.Return %result : tensor<*xi64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_build_seq_iter
// CHECK-SAME:   () -> tensor<*xi64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3xi64>
// CHECK:         }

// -----

// The loop condition is not a literal dense<true> but is computed via
// onnx.Not(dense<false>), which the constprop pass folds to dense<true>.
// Because the greedy driver applies patterns to a fixpoint, the Not ops are
// folded first (both the outer initial-condition and the body's yield), and
// LoopUnroll fires on the next driver iteration when it sees dense<true>.
//
// Expected: three additions of 1 unroll and fold to dense<3>.

func.func @test_loop_unroll_foldable_cond() -> tensor<i64> {
  %trip  = onnx.Constant dense<3>     : tensor<i64>
  %false = onnx.Constant dense<false> : tensor<i1>
  // Foldable initial condition: Not(false) → true.
  %cond  = "onnx.Not"(%false) : (tensor<i1>) -> tensor<i1>
  %init  = onnx.Constant dense<0> : tensor<i64>
  %res = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one  = onnx.Constant dense<1>     : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %f    = onnx.Constant dense<false> : tensor<i1>
    // Foldable yield condition: Not(false) → true.
    %yield_cond = "onnx.Not"(%f) : (tensor<i1>) -> tensor<i1>
    onnx.Yield %yield_cond, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> tensor<i64>
  onnx.Return %res : tensor<i64>
}
// CHECK-LABEL:  func.func @test_loop_unroll_foldable_cond
// CHECK-SAME:   () -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<i64>
// CHECK:         }

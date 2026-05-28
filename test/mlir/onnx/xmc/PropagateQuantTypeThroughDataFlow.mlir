// RUN: onnx-mlir-opt --propagate-quant-type-through-dataflow %s -split-input-file | FileCheck %s

// -----
// Forward: per-tensor quant operand, f32 result on Reshape -> result becomes quant.
// CHECK-LABEL: func @fwd_reshape_per_tensor
// CHECK: %[[Q:.+]] = quant.scast
// CHECK: %[[R:.+]] = "onnx.Reshape"(%[[Q]], %{{.+}}) {{.+}} -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[R]]
func.func @fwd_reshape_per_tensor(%arg0: tensor<6xi8>) -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %s = "onnx.Constant"() {value = dense<[1,2,3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %s) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<3xi64>) -> tensor<1x2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%r) : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward: per-tensor quant operand, f32 result on Transpose -> result becomes quant.
// CHECK-LABEL: func @fwd_transpose_per_tensor
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[T]]
func.func @fwd_transpose_per_tensor(%arg0: tensor<2x3xi8>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %t = "onnx.Transpose"(%q) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2xf32>
  %back = "builtin.unrealized_conversion_cast"(%t) : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward through every other whitelisted op (Squeeze/Unsqueeze/Flatten/Identity/DepthToSpace/SpaceToDepth).
// CHECK-LABEL: func @fwd_squeeze
// CHECK: %[[S:.+]] = "onnx.Squeeze"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_squeeze(%arg0: tensor<1x2x3xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
  %q = quant.scast %arg0 : tensor<1x2x3xi8> to tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %s = "onnx.Squeeze"(%q, %axes) : (tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%s) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// CHECK-LABEL: func @fwd_unsqueeze
// CHECK: %[[U:.+]] = "onnx.Unsqueeze"(%{{.+}}, %{{.+}}) : {{.+}} -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_unsqueeze(%arg0: tensor<2x3xi8>) -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %u = "onnx.Unsqueeze"(%q, %axes) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<1x2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%u) : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// CHECK-LABEL: func @fwd_flatten
// CHECK: %[[F:.+]] = "onnx.Flatten"(%{{.+}}) {{.+}} -> tensor<2x12x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_flatten(%arg0: tensor<2x3x4xi8>) -> tensor<2x12x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<2x3x4xi8> to tensor<2x3x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %f = "onnx.Flatten"(%q) {axis = 1 : si64} : (tensor<2x3x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x12xf32>
  %back = "builtin.unrealized_conversion_cast"(%f) : (tensor<2x12xf32>) -> tensor<2x12x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x12x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// CHECK-LABEL: func @fwd_identity
// CHECK: %[[I:.+]] = "onnx.Identity"(%{{.+}}) : {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_identity(%arg0: tensor<2x3xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %i = "onnx.Identity"(%q) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%i) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// CHECK-LABEL: func @fwd_depth_to_space
// CHECK: %[[D:.+]] = "onnx.DepthToSpace"(%{{.+}}) {{.+}} -> tensor<1x1x4x4x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_depth_to_space(%arg0: tensor<1x4x2x2xi8>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<1x4x2x2xi8> to tensor<1x4x2x2x!quant.uniform<i8:f32, 5.000000e-01>>
  %d = "onnx.DepthToSpace"(%q) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x2x2x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x1x4x4xf32>
  %back = "builtin.unrealized_conversion_cast"(%d) : (tensor<1x1x4x4xf32>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<1x1x4x4x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// CHECK-LABEL: func @fwd_space_to_depth
// CHECK: %[[S:.+]] = "onnx.SpaceToDepth"(%{{.+}}) {{.+}} -> tensor<1x4x2x2x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_space_to_depth(%arg0: tensor<1x1x4x4xi8>) -> tensor<1x4x2x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<1x1x4x4xi8> to tensor<1x1x4x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %s = "onnx.SpaceToDepth"(%q) {blocksize = 2 : si64} : (tensor<1x1x4x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x2x2xf32>
  %back = "builtin.unrealized_conversion_cast"(%s) : (tensor<1x4x2x2xf32>) -> tensor<1x4x2x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<1x4x2x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Backward: f32 operand from a single-use polymorphic producer, quant result -> producer's result becomes quant.
// CHECK-LABEL: func @bwd_polymorphic_producer
// CHECK: %[[ADD:.+]] = "onnx.Relu"(%{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[ADD]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @bwd_polymorphic_producer(%arg0: tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %relu = "onnx.Relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%relu) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Chain: forward propagates through two transposes.
// CHECK-LABEL: func @fwd_chain_two_transposes
// CHECK: %[[T1:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T2:.+]] = "onnx.Transpose"(%[[T1]]) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_chain_two_transposes(%arg0: tensor<2x3xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %t1 = "onnx.Transpose"(%q) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2xf32>
  %t2 = "onnx.Transpose"(%t1) {perm = [1, 0]} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%t2) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Same operand and result element type: no-op (no rewrite).
// CHECK-LABEL: func @noop_same_quant_type
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK-NOT: quant.scast
// CHECK-SAME-LABEL: return %[[T]]
func.func @noop_same_quant_type(%arg0: tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Both sides f32: no-op (no quant to propagate).
// CHECK-LABEL: func @noop_both_f32
// CHECK: "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2xf32>
func.func @noop_both_f32(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %t : tensor<3x2xf32>
}

// -----
// Two different quant types on operand and result: no rewrite (XCOMPILERRequantize territory).
// CHECK-LABEL: func @noop_different_quant_types
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}}: (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 2.500000e-01>>
func.func @noop_different_quant_types(%arg0: tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 2.500000e-01>> {
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 2.500000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 2.500000e-01>>
}

// -----
// Operand is the result of a quant.scast: forward proceeds (scast is upstream, not downstream/consumer).
// CHECK-LABEL: func @fwd_from_scast
// CHECK: "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @fwd_from_scast(%arg0: tensor<6xi8>) -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %s = "onnx.Constant"() {value = dense<[1,2,3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %s) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<3xi64>) -> tensor<1x2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%r) : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward skip: a non-return type-bound consumer (Cast) blocks the rewrite.
// CHECK-LABEL: func @fwd_skip_cast_consumer
// CHECK: %[[R:.+]] = "onnx.Reshape"({{.+}}) {{.+}} -> tensor<1x2x3xf32>
// CHECK: "onnx.Cast"(%[[R]])
func.func @fwd_skip_cast_consumer(%arg0: tensor<6xi8>) -> tensor<1x2x3xi64> {
  %s = "onnx.Constant"() {value = dense<[1,2,3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %s) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<3xi64>) -> tensor<1x2x3xf32>
  %c = "onnx.Cast"(%r) {to = i64, saturate = 1 : si64} : (tensor<1x2x3xf32>) -> tensor<1x2x3xi64>
  return %c : tensor<1x2x3xi64>
}

// -----
// Graph output bridge: result feeds func.return -> retype result to quant, insert scast + DequantizeLinear.
// CHECK-LABEL: func @fwd_return_bridge
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[S:.+]] = quant.scast %[[T]] : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<3x2xi8>
// CHECK: %[[DQ:.+]] = "onnx.DequantizeLinear"(%[[S]], %{{.+}}, %{{.+}})
// CHECK: return %[[DQ]] : tensor<3x2xf32>
func.func @fwd_return_bridge(%arg0: tensor<2x3xi8>) -> tensor<3x2xf32> {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %t = "onnx.Transpose"(%q) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2xf32>
  return %t : tensor<3x2xf32>
}

// -----
// Graph input bridge: f32 block-arg feeds an op whose result is quant -> insert QuantizeLinear + scast locally before op.
// CHECK-LABEL: func @bwd_block_arg_bridge
// CHECK: %[[Q:.+]] = "onnx.QuantizeLinear"(%arg0, %{{.+}}, %{{.+}})
// CHECK: %[[S:.+]] = quant.scast %[[Q]]
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[S]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[T]]
func.func @bwd_block_arg_bridge(%arg0: tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Backward skip: producer is a type-bound op (Constant). No rewrite expected.
// CHECK-LABEL: func @bwd_skip_constant_producer
// CHECK: %[[C:.+]] = onnx.Constant
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[C]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @bwd_skip_constant_producer() -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %c = onnx.Constant dense<1.0> : tensor<2x3xf32>
  %t = "onnx.Transpose"(%c) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Backward skip: producer is a Cast (type-bound by 'to' attr). No rewrite expected.
// CHECK-LABEL: func @bwd_skip_cast_producer
// CHECK: %[[C:.+]] = "onnx.Cast"(%{{.+}})
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[C]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @bwd_skip_cast_producer(%arg0: tensor<2x3xi64>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %c = "onnx.Cast"(%arg0) {to = f32, saturate = 1 : si64} : (tensor<2x3xi64>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%c) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Per-axis quant on operand: defensive skip (per-axis is weights-only; not on the activation path).
// CHECK-LABEL: func @skip_per_axis
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2xf32>
func.func @skip_per_axis(%arg0: tensor<2x3xi8>) -> tensor<3x2xf32> {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01, 2.000000e-01}>>
  %t = "onnx.Transpose"(%q) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01, 2.000000e-01}>>) -> tensor<3x2xf32>
  return %t : tensor<3x2xf32>
}

// -----
// Idempotency: running the pass on already-propagated IR is a no-op.
// CHECK-LABEL: func @idempotent
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @idempotent(%arg0: tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward + multi-consumer including func.return AND a polymorphic op.
// Retype the f32 result to quant, insert scast+DQ bridge only for the return.
// The polymorphic sibling adopts the new quant operand type directly.
// CHECK-LABEL: func @fwd_return_plus_polymorphic
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[RELU:.+]] = "onnx.Relu"(%[[T]])
// CHECK: %[[S:.+]] = quant.scast %[[T]] : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<3x2xi8>
// CHECK: %[[DQ:.+]] = "onnx.DequantizeLinear"(%[[S]], %{{.+}}, %{{.+}})
// CHECK: return %[[DQ]], %[[RELU]]
func.func @fwd_return_plus_polymorphic(%arg0: tensor<2x3xi8>) -> (tensor<3x2xf32>, tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>) {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %t = "onnx.Transpose"(%q) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2xf32>
  %r = "onnx.Relu"(%t) : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t, %r : tensor<3x2xf32>, tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward + multi-consumer including func.return AND a non-return type-bound
// consumer (Cast). Skip entire rewrite to keep Cast's f32 input intact.
// CHECK-LABEL: func @fwd_return_plus_cast_skips
// CHECK: %[[T:.+]] = "onnx.Transpose"(%{{.+}}) {{.+}} -> tensor<3x2xf32>
// CHECK: %[[C:.+]] = "onnx.Cast"(%[[T]])
// CHECK: return %[[T]], %[[C]]
func.func @fwd_return_plus_cast_skips(%arg0: tensor<2x3xi8>) -> (tensor<3x2xf32>, tensor<3x2xi64>) {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %t = "onnx.Transpose"(%q) {perm = [1, 0]} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2xf32>
  %c = "onnx.Cast"(%t) {to = i64, saturate = 1 : si64} : (tensor<3x2xf32>) -> tensor<3x2xi64>
  return %t, %c : tensor<3x2xf32>, tensor<3x2xi64>
}

// -----
// Backward + producer also feeds func.return as sibling.
// Retype producer; bridge the return sibling with scast+DQ; quant-using op
// (Transpose) sees quant directly.
// CHECK-LABEL: func @bwd_producer_return_sibling
// CHECK: %[[R:.+]] = "onnx.Relu"(%{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[R]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[S:.+]] = quant.scast %[[R]] : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<2x3xi8>
// CHECK: %[[DQ:.+]] = "onnx.DequantizeLinear"(%[[S]], %{{.+}}, %{{.+}})
// CHECK: return %[[T]], %[[DQ]]
func.func @bwd_producer_return_sibling(%arg0: tensor<2x3xf32>) -> (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xf32>) {
  %r = "onnx.Relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%r) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t, %r : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xf32>
}

// -----
// Backward + producer also feeds a non-return type-bound sibling (Cast).
// Whole backward rewrite is skipped to keep Cast's f32 input intact.
// CHECK-LABEL: func @bwd_producer_cast_sibling_skips
// CHECK: %[[R:.+]] = "onnx.Relu"(%{{.+}}) {{.+}} -> tensor<2x3xf32>
// CHECK: %[[C:.+]] = "onnx.Cast"(%[[R]])
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[R]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[T]], %[[C]]
func.func @bwd_producer_cast_sibling_skips(%arg0: tensor<2x3xf32>) -> (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xi64>) {
  %r = "onnx.Relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %c = "onnx.Cast"(%r) {to = i64, saturate = 1 : si64} : (tensor<2x3xf32>) -> tensor<2x3xi64>
  %t = "onnx.Transpose"(%r) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t, %c : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xi64>
}

// -----
// Graph input that also feeds a sibling (and that sibling also goes to return).
// Block-arg branch fires for our op only; sibling is untouched; block-arg type
// stays f32 (ABI preserved).
// CHECK-LABEL: func @bwd_block_arg_with_other_users
// CHECK: %[[Q:.+]] = "onnx.QuantizeLinear"(%arg0, %{{.+}}, %{{.+}})
// CHECK: %[[S:.+]] = quant.scast %[[Q]]
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[S]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[T]], %arg0
func.func @bwd_block_arg_with_other_users(%arg0: tensor<2x3xf32>) -> (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xf32>) {
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t, %arg0 : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xf32>
}

// -----
// Long data-flow-only chain ending at func.return: every op converges to quant
// and a single scast+DequantizeLinear bridge is inserted just before return.
// CHECK-LABEL: func @chain_dataflow_only_to_return
// CHECK: %[[R:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[R]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[F:.+]] = "onnx.Flatten"(%[[T]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[SC:.+]] = quant.scast %[[F]] : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<3x2xi8>
// CHECK: %[[DQ:.+]] = "onnx.DequantizeLinear"(%[[SC]], %{{.+}}, %{{.+}})
// CHECK: return %[[DQ]]
func.func @chain_dataflow_only_to_return(%arg0: tensor<6xi8>) -> tensor<3x2xf32> {
  %sh = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %sh) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%r) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %f = "onnx.Flatten"(%t) {axis = 1 : si64} : (tensor<3x2xf32>) -> tensor<3x2xf32>
  return %f : tensor<3x2xf32>
}

// -----
// Mixed chain: compute op (Relu) interleaved between two data-flow ops. The
// shared-SSA retype propagates through Relu without us touching it: Reshape's
// forward retypes Relu's operand; the trailing Transpose's backward retypes
// Relu's result (single use). Relu naturally ends up quant -> quant.
// CHECK-LABEL: func @chain_with_compute_op
// CHECK: %[[R:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[RL:.+]] = "onnx.Relu"(%[[R]]) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[RL]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[T]]
func.func @chain_with_compute_op(%arg0: tensor<6xi8>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %sh = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %sh) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %rl = "onnx.Relu"(%r) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%rl) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Two data-flow producers feeding the same compute op (Add): both reshapes
// forward-propagate to quant; Add naturally ends up consuming two quant
// operands.
// CHECK-LABEL: func @two_dataflow_to_compute
// CHECK: %[[R1:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[R2:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[A:.+]] = "onnx.Add"(%[[R1]], %[[R2]]) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[A]]
func.func @two_dataflow_to_compute(%lhs: tensor<6xi8>, %rhs: tensor<6xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %sh = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %ql = quant.scast %lhs : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %qr = quant.scast %rhs : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r1 = "onnx.Reshape"(%ql, %sh) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %r2 = "onnx.Reshape"(%qr, %sh) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %a = "onnx.Add"(%r1, %r2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %a : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward propagation cascades through the chain but the last data-flow op
// (Flatten, single-stepping into Cast) is blocked. The earlier ops still
// converge to quant; only the boundary op retains the q->f32 mismatch, which
// downstream Q/DQ rematerialization paths handle.
// CHECK-LABEL: func @chain_blocked_by_terminal_cast
// CHECK: %[[R:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[R]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[F:.+]] = "onnx.Flatten"(%[[T]]) {axis = 1 : si64} : (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2xf32>
// CHECK: %[[C:.+]] = "onnx.Cast"(%[[F]]) {{.+}} : (tensor<3x2xf32>) -> tensor<3x2xi64>
// CHECK: return %[[C]]
func.func @chain_blocked_by_terminal_cast(%arg0: tensor<6xi8>) -> tensor<3x2xi64> {
  %sh = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %sh) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%r) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %f = "onnx.Flatten"(%t) {axis = 1 : si64} : (tensor<3x2xf32>) -> tensor<3x2xf32>
  %c = "onnx.Cast"(%f) {to = i64, saturate = 1 : si64} : (tensor<3x2xf32>) -> tensor<3x2xi64>
  return %c : tensor<3x2xi64>
}

// -----
// Forward then backward in the same chain: data-flow (in=quant, out=f32) feeds
// a non-whitelisted compute op (Relu, polymorphic), which feeds another
// data-flow (in=f32, out=quant). Forward and backward triggers meet on the
// Relu via shared SSA.
// CHECK-LABEL: func @fwd_and_bwd_meet_on_compute
// CHECK: %[[R:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[RL:.+]] = "onnx.Relu"(%[[R]]) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[RL]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[T]]
func.func @fwd_and_bwd_meet_on_compute(%arg0: tensor<6xi8>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %sh = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Reshape"(%q, %sh) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %rl = "onnx.Relu"(%r) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%rl) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %t : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Long mixed chain across multiple whitelisted op kinds plus a compute op
// (Relu), with a quant constraint at BOTH ends (scast feeds the head; the
// terminal Flatten result is quant). Forward propagation from the head meets
// backward propagation from the tail at the Relu, so every op (including the
// non-whitelisted Relu) ends up quant -> quant via shared SSA retyping.
// CHECK-LABEL: func @long_mixed_chain_end_to_end_quant
// CHECK: %[[U:.+]] = "onnx.Unsqueeze"(%{{.+}}, %{{.+}}) : (tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<1x3x4x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[RL:.+]] = "onnx.Relu"(%[[U]]) : (tensor<1x3x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x3x4x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[SQ:.+]] = "onnx.Squeeze"(%[[RL]], %{{.+}}) : (tensor<1x3x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T:.+]] = "onnx.Transpose"(%[[SQ]]) {{.+}} -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[F:.+]] = "onnx.Flatten"(%[[T]]) {{.+}} -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[F]]
func.func @long_mixed_chain_end_to_end_quant(%arg0: tensor<12xi8>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %shA = "onnx.Constant"() {value = dense<[3, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
  %q = quant.scast %arg0 : tensor<12xi8> to tensor<12x!quant.uniform<i8:f32, 5.000000e-01>>
  %rsh = "onnx.Reshape"(%q, %shA) {allowzero = 0 : si64} : (tensor<12x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %u = "onnx.Unsqueeze"(%rsh, %axes) : (tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1xi64>) -> tensor<1x3x4xf32>
  %rl = "onnx.Relu"(%u) : (tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
  %sq = "onnx.Squeeze"(%rl, %axes) : (tensor<1x3x4xf32>, tensor<1xi64>) -> tensor<3x4xf32>
  %t = "onnx.Transpose"(%sq) {perm = [1, 0]} : (tensor<3x4xf32>) -> tensor<4x3xf32>
  %f = "onnx.Flatten"(%t) {axis = 1 : si64} : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %f : tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Diamond: a single quant source feeds two parallel data-flow chains; the
// chains rejoin at an Add. Both branches forward-propagate independently to
// quant; the Add ends up consuming two quant operands.
// CHECK-LABEL: func @diamond_two_chains_into_add
// CHECK: %[[R1:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[T1:.+]] = "onnx.Transpose"(%[[R1]]) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[R2:.+]] = "onnx.Reshape"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[A:.+]] = "onnx.Add"(%[[T1]], %[[R2]]) : (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[A]]
func.func @diamond_two_chains_into_add(%arg0: tensor<6xi8>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>> {
  %shA = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %shB = "onnx.Constant"() {value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %q = quant.scast %arg0 : tensor<6xi8> to tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>
  %r1 = "onnx.Reshape"(%q, %shA) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<2x3xf32>
  %t1 = "onnx.Transpose"(%r1) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %r2 = "onnx.Reshape"(%q, %shB) {allowzero = 0 : si64} : (tensor<6x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<3x2xf32>
  %a = "onnx.Add"(%t1, %r2) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
  return %a : tensor<3x2x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward through Expand: broadcast replicates input values, result becomes quant.
// CHECK-LABEL: func @fwd_expand
// CHECK: %[[E:.+]] = "onnx.Expand"(%{{.+}}, %{{.+}}) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[E]]
func.func @fwd_expand(%arg0: tensor<1x4xi8>) -> tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>> {
  %shape = "onnx.Constant"() {value = dense<[3, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %q = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %e = "onnx.Expand"(%q, %shape) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2xi64>) -> tensor<3x4xf32>
  %back = "builtin.unrealized_conversion_cast"(%e) : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<3x4x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward through Neg: same scale and zp propagated (accepted approximation
// for asymmetric quant; exact for symmetric).
// CHECK-LABEL: func @fwd_neg
// CHECK: %[[N:.+]] = "onnx.Neg"(%{{.+}}) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[N]]
func.func @fwd_neg(%arg0: tensor<2x3xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %n = "onnx.Neg"(%q) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%n) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward through Clip: input range envelopes output range; same params OK.
// CHECK-LABEL: func @fwd_clip
// CHECK: %[[C:.+]] = "onnx.Clip"(%{{.+}}, %{{.+}}, %{{.+}}) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<f32>, tensor<f32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[C]]
func.func @fwd_clip(%arg0: tensor<2x3xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %min = "onnx.Constant"() {value = dense<-1.0> : tensor<f32>} : () -> tensor<f32>
  %max = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %q = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %c = "onnx.Clip"(%q, %min, %max) : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%c) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward through Resize (nearest mode): each output equals an input value.
// CHECK-LABEL: func @fwd_resize_nearest
// CHECK: %[[R:.+]] = "onnx.Resize"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) {{.+}}: (tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-01>>, none, tensor<4xf32>, none) -> tensor<1x3x8x8x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[R]]
func.func @fwd_resize_nearest(%arg0: tensor<1x3x4x4xi8>) -> tensor<1x3x8x8x!quant.uniform<i8:f32, 5.000000e-01>> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %sizes = "onnx.NoValue"() {value} : () -> none
  %q = quant.scast %arg0 : tensor<1x3x4x4xi8> to tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Resize"(%q, %roi, %scales, %sizes) {mode = "nearest", coordinate_transformation_mode = "half_pixel", nearest_mode = "round_prefer_floor"} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-01>>, none, tensor<4xf32>, none) -> tensor<1x3x8x8xf32>
  %back = "builtin.unrealized_conversion_cast"(%r) : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<1x3x8x8x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Resize with non-nearest mode is filtered out - leave mismatch as-is for
// other passes to handle.
// CHECK-LABEL: func @skip_resize_linear
// CHECK: %[[R:.+]] = "onnx.Resize"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) {{.+}} -> tensor<1x3x8x8xf32>
func.func @skip_resize_linear(%arg0: tensor<1x3x4x4xi8>) -> tensor<1x3x8x8xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %sizes = "onnx.NoValue"() {value} : () -> none
  %q = quant.scast %arg0 : tensor<1x3x4x4xi8> to tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %r = "onnx.Resize"(%q, %roi, %scales, %sizes) {mode = "linear", coordinate_transformation_mode = "half_pixel"} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-01>>, none, tensor<4xf32>, none) -> tensor<1x3x8x8xf32>
  return %r : tensor<1x3x8x8xf32>
}

// -----
// Forward through Concat: all input operands share the same quant type, so
// the f32 result is retyped to that common quant.
// CHECK-LABEL: func @fwd_concat
// CHECK: %[[C:.+]] = "onnx.Concat"(%{{.+}}, %{{.+}}) {{.+}}: (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[C]]
func.func @fwd_concat(%a: tensor<2x3xi8>, %b: tensor<2x3xi8>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %qa = quant.scast %a : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %qb = quant.scast %b : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %c = "onnx.Concat"(%qa, %qb) {axis = 0 : si64} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%c) : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Concat with operands carrying different quant types - skip (different
// scales mean a real requantization, not a propagation).
// CHECK-LABEL: func @noop_concat_mismatched_inputs
// CHECK: %[[C:.+]] = "onnx.Concat"(%{{.+}}, %{{.+}}) {{.+}} -> tensor<4x3xf32>
func.func @noop_concat_mismatched_inputs(%a: tensor<2x3xi8>, %b: tensor<2x3xi8>) -> tensor<4x3xf32> {
  %qa = quant.scast %a : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %qb = quant.scast %b : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>
  %c = "onnx.Concat"(%qa, %qb) {axis = 0 : si64} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>) -> tensor<4x3xf32>
  return %c : tensor<4x3xf32>
}

// -----
// Backward through Concat: f32 operands feeding Concat whose result is quant.
// All operand producers (Relu's here) get their results retyped to the common
// quant.
// CHECK-LABEL: func @bwd_concat
// CHECK: %[[RA:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[RB:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[C:.+]] = "onnx.Concat"(%[[RA]], %[[RB]]) {{.+}}: (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[C]]
func.func @bwd_concat(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %ra = "onnx.Relu"(%a) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %rb = "onnx.Relu"(%b) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %c = "onnx.Concat"(%ra, %rb) {axis = 0 : si64} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %c : tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Forward through Where: X and Y share the same quant; condition stays i1.
// CHECK-LABEL: func @fwd_where
// CHECK: %[[W:.+]] = "onnx.Where"(%{{.+}}, %{{.+}}, %{{.+}}) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[W]]
func.func @fwd_where(%cond: tensor<2x3xi1>, %x: tensor<2x3xi8>, %y: tensor<2x3xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %qx = quant.scast %x : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %qy = quant.scast %y : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %w = "onnx.Where"(%cond, %qx, %qy) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3xf32>
  %back = "builtin.unrealized_conversion_cast"(%w) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %back : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Where with X and Y carrying different quant types - skip.
// CHECK-LABEL: func @noop_where_mismatched_xy
// CHECK: %[[W:.+]] = "onnx.Where"(%{{.+}}, %{{.+}}, %{{.+}}) : {{.+}} -> tensor<2x3xf32>
func.func @noop_where_mismatched_xy(%cond: tensor<2x3xi1>, %x: tensor<2x3xi8>, %y: tensor<2x3xi8>) -> tensor<2x3xf32> {
  %qx = quant.scast %x : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %qy = quant.scast %y : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>
  %w = "onnx.Where"(%cond, %qx, %qy) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>) -> tensor<2x3xf32>
  return %w : tensor<2x3xf32>
}

// -----
// Backward through Where: both X and Y producers retyped; condition operand
// (i1) is left alone.
// CHECK-LABEL: func @bwd_where
// CHECK: %[[RX:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[RY:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[W:.+]] = "onnx.Where"(%{{.+}}, %[[RX]], %[[RY]]) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[W]]
func.func @bwd_where(%cond: tensor<2x3xi1>, %x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %rx = "onnx.Relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %ry = "onnx.Relu"(%y) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %w = "onnx.Where"(%cond, %rx, %ry) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %w : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Partial backward through Concat: one operand already quant (matching
// result), the other is f32 from a polymorphic producer. Only the f32
// operand's producer is retyped; the already-quant operand is untouched.
// CHECK-LABEL: func @bwd_concat_partial
// CHECK: %[[Q:.+]] = quant.scast
// CHECK: %[[R:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[C:.+]] = "onnx.Concat"(%[[Q]], %[[R]]) {{.+}}: (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[C]]
func.func @bwd_concat_partial(%a: tensor<2x3xi8>, %b: tensor<2x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %qa = quant.scast %a : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %rb = "onnx.Relu"(%b) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %c = "onnx.Concat"(%qa, %rb) {axis = 0 : si64} : (tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %c : tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Partial backward through Concat with an already-quant operand carrying
// DIFFERENT quant params from the result. The mismatched operand is left
// alone; only the f32 operand is retyped to the result's quant. The pre-pass
// mismatch (q1 input + q2 output) is preserved and surfaces for later passes
// to materialize an explicit requantize.
// CHECK-LABEL: func @bwd_concat_partial_with_mismatched_quant
// CHECK: %[[Q1:.+]] = quant.scast %{{.+}} : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK: %[[R:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[C:.+]] = "onnx.Concat"(%[[Q1]], %[[R]]) {{.+}}: (tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[C]]
func.func @bwd_concat_partial_with_mismatched_quant(%a: tensor<2x3xi8>, %b: tensor<2x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %qa = quant.scast %a : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>
  %rb = "onnx.Relu"(%b) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %c = "onnx.Concat"(%qa, %rb) {axis = 0 : si64} : (tensor<2x3x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<2x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %c : tensor<4x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

// -----
// Partial backward through Where: X is already quant (matching result), Y is
// f32. Only Y's producer is retyped.
// CHECK-LABEL: func @bwd_where_partial
// CHECK: %[[QX:.+]] = quant.scast
// CHECK: %[[RY:.+]] = "onnx.Relu"(%{{.+}}) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: %[[W:.+]] = "onnx.Where"(%{{.+}}, %[[QX]], %[[RY]]) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK: return %[[W]]
func.func @bwd_where_partial(%cond: tensor<2x3xi1>, %x: tensor<2x3xi8>, %y: tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>> {
  %qx = quant.scast %x : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  %ry = "onnx.Relu"(%y) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %w = "onnx.Where"(%cond, %qx, %ry) : (tensor<2x3xi1>, tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
  return %w : tensor<2x3x!quant.uniform<i8:f32, 5.000000e-01>>
}

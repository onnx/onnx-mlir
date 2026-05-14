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

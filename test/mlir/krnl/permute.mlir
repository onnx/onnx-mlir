// RUN: onnx-mlir-opt --lower-krnl %s -split-input-file | FileCheck %s

func @simple_permute() {
  %ii, %jj = krnl.define_loops 2
  krnl.permute(%ii, %jj) [1, 0] : !krnl.loop, !krnl.loop
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %foo = addi %i, %i : index
  }

  // CHECK-LABEL: simple_permute
  // CHECK-NEXT: affine.for [[OUTER_LOOP_IV:%.+]] = 0 to 20 {
  // CHECK-NEXT:   affine.for [[INNER_LOOP_IV:%.+]] = 0 to 10 {
  // CHECK-NEXT:     [[ADD:%.+]] = addi [[INNER_LOOP_IV]], [[INNER_LOOP_IV]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}
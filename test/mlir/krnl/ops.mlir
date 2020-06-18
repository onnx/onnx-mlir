// RUN: onnx-mlir-opt %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s
// RUN: onnx-mlir-opt %s | FileCheck %s

// -----

// GENERIC-DAG: #{{.*}} = affine_map<() -> (0)>
// GENERIC-DAG: #{{.*}} = affine_map<() -> (10)>
// GENERIC-DAG: #{{.*}} = affine_map<() -> (1)>
// GENERIC-DAG: #{{.*}} = affine_map<() -> (11)>
// GENERIC-DAG: #{{.*}} = affine_map<(d0, d1) -> (d0 - d1)>
// GENERIC-DAG: #{{.*}} = affine_map<(d0, d1) -> (d0 + d1)>

func @simple_iterate(%N : index) {
  %ii, %ij, %ik = krnl.define_loops 3
  %oi, %oj, %ok = krnl.optimize_loops  {
    krnl.return_loops %ii, %ij, %ik
  } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
  // GENERIC-NEXT: "krnl.terminate"() : () -> ()
  // GENERIC-NEXT: bounds = [#{{.*}}, #{{.*}}, #{{.*}}, #{{.*}}]

  // CHECK: krnl.iterate(%{{.*}}, %{{.*}}) with (%{{.*}} -> %{{.*}} = 0 to 10, %{{.*}} -> %{{.*}} = 1 to 11) {
  krnl.iterate(%oi, %oj) with (%ii -> %i = 0 to 10, %ij -> %j = 1 to 11) {

  }

  // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
  // CHECK: krnl.iterate(%{{.*}}, %{{.*}}) with (%{{.*}} -> %{{.*}} = 0 to 10, %{{.*}} -> %{{.*}} = 0 to 10) {
  krnl.iterate(%oi, %oj) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 10) {
    // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}) ( {
    // GENERIC-NEXT: ^bb0(%{{.*}}: index):
    // CHECK: krnl.iterate(%{{.*}}) with (%{{.*}} -> %{{.*}} = 0 to 10) {
    krnl.iterate(%ok) with (%ik -> %k = 0 to 10) {

    }
  }

  // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
  // CHECK: krnl.iterate(%{{.*}}, %{{.*}}) with (%{{.*}} -> %{{.*}} = 0 to %{{.*}}, %{{.*}} -> %{{.*}} = 0 to 10) {
  krnl.iterate(%oi, %oj) with (%ii -> %i = 0 to %N, %ij -> %j = 0 to 10) {

  }

  return
}

// -----

func @affine_map_bound(%N : index) {
  %ii, %ij, %ik = krnl.define_loops 3
  %oi, %oj, %ok = krnl.optimize_loops  {
    krnl.return_loops %ii, %ij, %ik
  } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
  // CHECK: krnl.iterate(%{{.*}}, %{{.*}}) with (%{{.*}} -> %{{.*}} = 0 to 10, %{{.*}} -> %{{.*}} = 0 to 10) {
  krnl.iterate(%oi, %oj) with (%ii -> %i = affine_map<()->(0)>() to affine_map<()->(10)>(), %ij -> %j = 0 to 10) {
    // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
    // GENERIC-NEXT: ^bb0(%{{.*}}: index):
    // CHECK: krnl.iterate(%{{.*}}) with (%{{.*}} -> %{{.*}} = #{{.*}}(%{{.*}}, %{{.*}}) to #{{.*}}(%{{.*}}, %{{.*}})) {
    krnl.iterate(%ok) with (%ik -> %k = affine_map<(d0, d1)->(d0 - d1)>(%i, %j) to affine_map<(d0, d1)->(d0 + d1)>(%i, %j)) {

    }

    // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
    // GENERIC-NEXT: ^bb0(%{{.*}}: index):
    // CHECK: krnl.iterate(%{{.*}}) with (%{{.*}} -> %{{.*}} = max #map{{.*}}(%{{.*}}, %{{.*}}) to min #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]) {
    krnl.iterate(%ok) with (%ik -> %k = max affine_map<(d0, d1)->(d0 - d1, 0)>(%i, %j) to min affine_map<(d0, d1)[s0]->(d0 + d1, s0)>(%i, %j)[%N]) {

    }
  }

  return
}

// -----

func @getref(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
  %c42_i64 = constant 42 : i64
  %0 = alloc() : memref<10x10xf32>

  // GENERIC: "krnl.getref"(%{{.*}}, %{{.*}}) ( {
  // GENERIC-NEXT: }) : ({{.*}}, {{.*}}) -> {{.*}}
  %1 = krnl.getref (%0, %c42_i64) : (memref<10x10xf32>, i64) -> memref<2x2xf32>

  return %1 : memref<2x2xf32>

}

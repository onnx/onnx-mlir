// RUN: onnf-opt %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s
// RUN: onnf-opt %s | FileCheck %s

// GENERIC-DAG: #{{.*}} = () -> (0)
// GENERIC-DAG: #{{.*}} = () -> (10)
// GENERIC-DAG: #{{.*}} = () -> (1)
// GENERIC-DAG: #{{.*}} = () -> (11)
// GENERIC-DAG: #{{.*}} = (d0, d1) -> (d0 - d1)
// GENERIC-DAG: #{{.*}} = (d0, d1) -> (d0 + d1)

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

func @affine_map_bound(%N : index) {
  %ii, %ij, %ik = krnl.define_loops 3
  %oi, %oj, %ok = krnl.optimize_loops  {
    krnl.return_loops %ii, %ij, %ik
  } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
  // CHECK: krnl.iterate(%{{.*}}, %{{.*}}) with (%{{.*}} -> %{{.*}} = 0 to 10, %{{.*}} -> %{{.*}} = 0 to 10) {
  krnl.iterate(%oi, %oj) with (%ii -> %i = ()->(0)() to ()->(10)(), %ij -> %j = 0 to 10) {
    // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
    // GENERIC-NEXT: ^bb0(%{{.*}}: index):
    // CHECK: krnl.iterate(%{{.*}}) with (%{{.*}} -> %{{.*}} = #{{.*}}(%{{.*}}, %{{.*}}) to #{{.*}}(%{{.*}}, %{{.*}})) {
    krnl.iterate(%ok) with (%ik -> %k = (d0, d1)->(d0 - d1)(%i, %j) to (d0, d1)->(d0 + d1)(%i, %j)) {

    }

    // GENERIC: "krnl.iterate"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ( {
    // GENERIC-NEXT: ^bb0(%{{.*}}: index):
    // CHECK: krnl.iterate(%{{.*}}) with (%{{.*}} -> %{{.*}} = max #map{{.*}}(%{{.*}}, %{{.*}}) to min #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]) {
    krnl.iterate(%ok) with (%ik -> %k = max (d0, d1)->(d0 - d1, 0)(%i, %j) to min (d0, d1)[s0]->(d0 + d1, s0)(%i, %j)[%N]) {

    }
  }

  return
}
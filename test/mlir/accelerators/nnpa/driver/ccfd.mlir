// RUN: ccfd=$(dirname %s)/ccfd.onnx && curl -L https://github.com/IBM/ai-on-z-fraud-detection/raw/main/onnx%20models/ccf_lstm_static_tf2onnx_OS_new.onnx -o ${ccfd} && onnx-mlir --march=z16 --maccel=NNPA --disable-compiler-stick-unstick --EmitMLIR --printIR -tag="test" ${ccfd} | FileCheck %s && rm -rf ${ccfd}

// COM: This test is to check regression on the IBM CCFD model.
// COM: We expect that there are only one zlow.stick for the input and one zlow.unstick for the output.
// COM: It is the necessary condition to get the best performance.

// CHECK-LABEL: func.func @main_graph
// CHECK-DAG: krnl.global
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-NEXT: zlow.stick

// CHECK-DAG: krnl.global
// CHECK-DAG: krnl.global
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-DAG: memref.alloc
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-NEXT: zlow.lstm

// No stick and unstick between two LSTMs.
// CHECK-NOT: zlow.stick
// CHECK-NOT: zlow.unstick
// 
// CHECK-DAG: krnl.global
// CHECK-DAG: krnl.global
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-DAG: memref.alloc
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-NEXT: zlow.lstm
// 
// No stick and unstick in between.
// CHECK-NOT: zlow.stick
// CHECK-NOT: zlow.unstick
// 
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-DAG: krnl.global
// CHECK-DAG: krnl.global
// CHECK-NEXT: zlow.matmul
// 
// No stick and unstick in between.
// CHECK-NOT: zlow.stick
// CHECK-NOT: zlow.unstick
// 
// CHECK-DAG: krnl.global
// CHECK-DAG: memref.alloc
// CHECK-DAG: krnl.global
// CHECK-NEXT: zlow.add
// 
// No stick and unstick in between.
// CHECK-NOT: zlow.stick
// CHECK-NOT: zlow.unstick
// 
// CHECK-DAG: memref.alloc
// CHECK-DAG: krnl.global
// CHECK-NEXT: zlow.sigmoid
// 
// CHECK: memref.alloc
// CHECK-NEXT: zlow.unstick

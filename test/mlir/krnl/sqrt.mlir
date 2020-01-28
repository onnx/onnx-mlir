// RUN: onnf-opt --shape-inference --lower-all-llvm %s -split-input-file | FileCheck %s
module {
  func @test_sqrt_32(%arg0 : f32) -> f32 {
    %0 = "krnl.sqrt"(%arg0) : (f32) -> f32
    "std.return"(%0) : (f32) -> ()

    // CHECK: llvm.func @llvm.sqrt.f32(!llvm.float) -> !llvm.float
    // CHECK-NEXT: llvm.func @test_sqrt_32(%arg0: !llvm.float) -> !llvm.float {
    // CHECK-NEXT: [[RES:%.+]] = llvm.call @llvm.sqrt.f32(%arg0) : (!llvm.float) -> !llvm.float
    // CHECK-NEXT: llvm.return [[RES]] : !llvm.float
  }
}

module{
  func @test_sqrt_64(%arg0 : f64) -> f64 {
    %0 = "krnl.sqrt"(%arg0) : (f64) -> f64
    "std.return"(%0) : (f64) -> ()

    // CHECK: llvm.func @llvm.sqrt.f64(!llvm.double) -> !llvm.double
    // CHECK-NEXT: llvm.func @test_sqrt_64(%arg0: !llvm.double) -> !llvm.double {
    // CHECK-NEXT: [[RES:%.+]] = llvm.call @llvm.sqrt.f64(%arg0) : (!llvm.double) -> !llvm.double
    // CHECK-NEXT: llvm.return [[RES]] : !llvm.double
  }
}

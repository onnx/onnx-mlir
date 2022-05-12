// RUN: onnx-mlir-opt --zlow-rewrite --canonicalize %s -split-input-file | FileCheck %s

func @test_remove_unstick_view_stick(%arg0: memref<7x4x1x8x32x64xf16>) -> (memref<7x4x1x8x32x64xf16>){
    %0 = memref.alloc() {alignment = 16 : i64} : memref<7x1x256x200xf32>
    "zlow.unstick"(%arg0, %0) {layout = "4DS"} : (memref<7x4x1x8x32x64xf16>, memref<7x1x256x200xf32>) -> ()
    %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [7, 256, 200], strides: [51200, 200, 1] : memref<7x1x256x200xf32> to memref<7x256x200xf32>
    %2 = memref.alloc() {alignment = 4096 : i64} : memref<7x4x1x8x32x64xf16>
    "zlow.stick"(%1, %2) {layout = "3DS"} : (memref<7x256x200xf32>, memref<7x4x1x8x32x64xf16>) -> ()
    "func.return"(%2) : (memref<7x4x1x8x32x64xf16>) -> ()

    // CHECK-LABEL: test_remove_unstick_view_stick
    // CHECK-NEXT: return %arg0 : memref<7x4x1x8x32x64xf16>
    // CHECK-NOT: "zlow.unstick"
    // CHECK-NOT: "zlow.stick"
}

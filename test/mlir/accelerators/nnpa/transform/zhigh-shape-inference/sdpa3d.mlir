// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

// Q=[2,64,32], KT=[2,32,128], Mask=[2,64,128], V=[2,128,48] -> Out=[2,64,48]
func.func @test_sdpa3d_no_transpose(%Q : tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %KT : tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %Mask : tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %V : tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %0 = "zhigh.SDPA3D"(%Q, %KT, %Mask, %V) {transposeKT = 0 : si64} : (tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  "func.return"(%0) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @test_sdpa3d_no_transpose
// CHECK-SAME:   ([[Q_:%.+]]: tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[KT_:%.+]]: tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[MASK_:%.+]]: tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[V_:%.+]]: tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x48xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.SDPA3D"([[Q_]], [[KT_]], [[MASK_]], [[V_]]) <{transposeKT = 0 : si64}> : (tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x32x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x64x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// Q=[2,64,32], K=[2,128,32], Mask=[2,64,128], V=[2,128,48] -> Out=[2,64,48]
// transposeKT=1: K is [s,p,n], P=KDims[1]=128.
func.func @test_sdpa3d_transpose_kt(%Q : tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %K : tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %Mask : tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %V : tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %0 = "zhigh.SDPA3D"(%Q, %K, %Mask, %V) {transposeKT = 1 : si64} : (tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  "func.return"(%0) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @test_sdpa3d_transpose_kt
// CHECK-SAME:   ([[Q_:%.+]]: tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[K_:%.+]]: tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[MASK_:%.+]]: tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[V_:%.+]]: tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x48xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.SDPA3D"([[Q_]], [[K_]], [[MASK_]], [[V_]]) <{transposeKT = 1 : si64}> : (tensor<2x64x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x64x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x128x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x64x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x64x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// Unknown dimensions with transposeKT=0.
// Q=[?x?x?], KT=[?x?x?], Mask=[?x?x?], V=[?x?x48] -> Out=[?x?x48]
func.func @test_sdpa3d_unknown_dims(%Q : tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %KT : tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %Mask : tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %V : tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %0 = "zhigh.SDPA3D"(%Q, %KT, %Mask, %V) {transposeKT = 0 : si64} : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  "func.return"(%0) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @test_sdpa3d_unknown_dims
// CHECK-SAME:   ([[Q_:%.+]]: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[KT_:%.+]]: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[MASK_:%.+]]: tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[V_:%.+]]: tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.SDPA3D"([[Q_]], [[KT_]], [[MASK_]], [[V_]]) <{transposeKT = 0 : si64}> : (tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

// Unknown dimensions with transposeKT=1.
// Q=[2x?x?], K=[2x?x?], Mask=[2x?x?], V=[2x?x48] -> Out=[2x?x48]
func.func @test_sdpa3d_unknown_dims_transpose_kt(%Q : tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %K : tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %Mask : tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %V : tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %0 = "zhigh.SDPA3D"(%Q, %K, %Mask, %V) {transposeKT = 1 : si64} : (tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  "func.return"(%0) : (tensor<*xf16>) -> ()

// CHECK-LABEL:  func @test_sdpa3d_unknown_dims_transpose_kt
// CHECK-SAME:   ([[Q_:%.+]]: tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[K_:%.+]]: tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[MASK_:%.+]]: tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[V_:%.+]]: tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.SDPA3D"([[Q_]], [[K_]], [[MASK_]], [[V_]]) <{transposeKT = 1 : si64}> : (tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x?xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<2x?x48xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }
}

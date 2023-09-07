module {
}


// -----
module {
  func.func @test_dim_analysis_with_bert(%arg0: tensor<?x256xi64>, %arg1: tensor<?x256xi64>) -> (tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>) {
    "onnx.DimGroup"(%arg0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256xi64>) -> ()
    %0 = "onnx.Dim"(%arg1) <{axis = 0 : si64}> : (tensor<?x256xi64>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256xi64>) -> ()
    %1 = onnx.Constant dense<256> : tensor<1xi64>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = "onnx.Concat"(%0, %1, %2) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %4 = onnx.ConstantOfShape(%3) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x256x1xf32>
    "onnx.DimGroup"(%4) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256x1xf32>) -> ()
    %5 = onnx.Constant dense<1> : tensor<1xi64>
    %6 = onnx.Constant dense<256> : tensor<1xi64>
    %7 = "onnx.Concat"(%0, %5, %6) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %8 = "onnx.Reshape"(%arg0, %7) <{allowzero = 0 : si64}> : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<?x1x256xi64>
    "onnx.DimGroup"(%8) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256xi64>) -> ()
    %9 = "onnx.Cast"(%8) <{saturate = 1 : si64, to = f32}> : (tensor<?x1x256xi64>) -> tensor<?x1x256xf32>
    "onnx.DimGroup"(%9) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256xf32>) -> ()
    %10 = "onnx.Mul"(%4, %9) : (tensor<?x256x1xf32>, tensor<?x1x256xf32>) -> tensor<?x256x256xf32>
    "onnx.DimGroup"(%10) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256x256xf32>) -> ()
    %11 = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
    %12 = "onnx.Reshape"(%10, %11) <{allowzero = 0 : si64}> : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
    "onnx.DimGroup"(%12) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256x256xf32>) -> ()
    %13 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %14 = "onnx.Sub"(%13, %12) : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
    "onnx.DimGroup"(%14) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256x256xf32>) -> ()
    %15 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
    %16 = "onnx.Mul"(%14, %15) : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
    "onnx.DimGroup"(%16) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256x256xf32>) -> ()
    %17 = onnx.Constant dense<[-1, 1, 256, 256]> : tensor<4xi64>
    %18 = "onnx.Reshape"(%10, %17) <{allowzero = 0 : si64}> : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x1x256x256xf32>
    "onnx.DimGroup"(%18) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256x256xf32>) -> ()
    %19 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %20 = "onnx.Sub"(%19, %18) : (tensor<f32>, tensor<?x1x256x256xf32>) -> tensor<?x1x256x256xf32>
    "onnx.DimGroup"(%20) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256x256xf32>) -> ()
    %21 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
    %22 = "onnx.Mul"(%20, %21) : (tensor<?x1x256x256xf32>, tensor<f32>) -> tensor<?x1x256x256xf32>
    "onnx.DimGroup"(%22) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x1x256x256xf32>) -> ()
    onnx.Return %22, %20, %16 : tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>, tensor<?x1x256x256xf32>
  }
}


// -----
module {
  func.func @test_unary_elementwise(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
    "onnx.DimGroup"(%arg0) <{axis = 2 : si64, group_id = 1 : si64}> : (tensor<?x3x?xf32>) -> ()
    "onnx.DimGroup"(%arg0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x3x?xf32>) -> ()
    %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
    "onnx.DimGroup"(%0) <{axis = 2 : si64, group_id = 1 : si64}> : (tensor<?x3x?xf32>) -> ()
    "onnx.DimGroup"(%0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x3x?xf32>) -> ()
    onnx.Return %0 : tensor<?x3x?xf32>
  }
}


// -----
module {
  func.func @test_binary_elementwise(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
    "onnx.DimGroup"(%arg0) <{axis = 2 : si64, group_id = 1 : si64}> : (tensor<?x3x?xf32>) -> ()
    "onnx.DimGroup"(%arg0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x3x?xf32>) -> ()
    %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
    "onnx.DimGroup"(%0) <{axis = 2 : si64, group_id = 1 : si64}> : (tensor<?x3x?xf32>) -> ()
    "onnx.DimGroup"(%0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x3x?xf32>) -> ()
    %1 = "onnx.Add"(%0, %arg0) : (tensor<?x3x?xf32>, tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
    "onnx.DimGroup"(%1) <{axis = 2 : si64, group_id = 1 : si64}> : (tensor<?x3x?xf32>) -> ()
    "onnx.DimGroup"(%1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x3x?xf32>) -> ()
    onnx.Return %1 : tensor<?x3x?xf32>
  }
}


// -----
module {
  func.func @test_matmul_batchsize(%arg0: tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32> {
    "onnx.DimGroup"(%arg0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x8x16x16xf32>) -> ()
    %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
    "onnx.DimGroup"(%0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x8x16x16xf32>) -> ()
    %1 = "onnx.MatMul"(%0, %arg0) : (tensor<?x8x16x16xf32>, tensor<?x8x16x16xf32>) -> tensor<?x8x16x16xf32>
    "onnx.DimGroup"(%1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x8x16x16xf32>) -> ()
    onnx.Return %1 : tensor<?x8x16x16xf32>
  }
}


// -----
Warning: [Shape inference, dim 2] the inferred dim (128) is different from the existing dim (32). Use the existing dim instead.
Warning: [Shape inference, dim 2] the inferred dim (128) is different from the existing dim (32). Use the existing dim instead.
module {
  func.func @test_matmul_batchsize_diff_rank(%arg0: tensor<8x?x16x4xf32>) -> tensor<8x?x16x32xf32> {
    "onnx.DimGroup"(%arg0) <{axis = 1 : si64, group_id = 0 : si64}> : (tensor<8x?x16x4xf32>) -> ()
    %0 = onnx.Constant dense<[-1, 4, 128]> : tensor<3xi64>
    %1 = "onnx.Reshape"(%arg0, %0) <{allowzero = 0 : si64}> : (tensor<8x?x16x4xf32>, tensor<3xi64>) -> tensor<?x4x32xf32>
    "onnx.DimGroup"(%1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x4x32xf32>) -> ()
    %2 = "onnx.MatMul"(%arg0, %1) : (tensor<8x?x16x4xf32>, tensor<?x4x32xf32>) -> tensor<8x?x16x32xf32>
    "onnx.DimGroup"(%2) <{axis = 1 : si64, group_id = 0 : si64}> : (tensor<8x?x16x32xf32>) -> ()
    onnx.Return %2 : tensor<8x?x16x32xf32>
  }
}


// -----
Warning: [Shape inference, dim 2] the inferred dim (128) is different from the existing dim (32). Use the existing dim instead.
Warning: [Shape inference, dim 2] the inferred dim (128) is different from the existing dim (32). Use the existing dim instead.
module {
  func.func @test_reshape_single_dyn_dim(%arg0: tensor<8x?x16x4xf32>) -> tensor<?x4x32xf32> {
    "onnx.DimGroup"(%arg0) <{axis = 1 : si64, group_id = 0 : si64}> : (tensor<8x?x16x4xf32>) -> ()
    %0 = onnx.Constant dense<[-1, 4, 128]> : tensor<3xi64>
    %1 = "onnx.Reshape"(%arg0, %0) <{allowzero = 0 : si64}> : (tensor<8x?x16x4xf32>, tensor<3xi64>) -> tensor<?x4x32xf32>
    "onnx.DimGroup"(%1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x4x32xf32>) -> ()
    onnx.Return %1 : tensor<?x4x32xf32>
  }
}


// -----
module {
  func.func @test_expand_from_concat_dims(%arg0: tensor<1x256xi64>, %arg1: tensor<?x256xi64>) -> tensor<?x256xi64> {
    %0 = onnx.Constant dense<256> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg1) <{axis = 0 : si64}> : (tensor<?x256xi64>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256xi64>) -> ()
    %2 = "onnx.Concat"(%1, %0) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = "onnx.Expand"(%arg0, %2) {onnx_node_name = "Expand_30"} : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
    "onnx.DimGroup"(%3) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256xi64>) -> ()
    onnx.Return %3 : tensor<?x256xi64>
  }
}


// -----
module {
  func.func @test_reshape_rank_2(%arg0: tensor<?x?xi64>) -> tensor<?x?xi64> {
    "onnx.DimGroup"(%arg0) <{axis = 1 : si64, group_id = 0 : si64}> : (tensor<?x?xi64>) -> ()
    %0 = onnx.Constant dense<-1> : tensor<1xi64>
    "onnx.DimGroup"(%0) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<1xi64>) -> ()
    %1 = "onnx.Dim"(%arg0) <{axis = 1 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?xi64>) -> ()
    %2 = "onnx.Concat"(%1, %0) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = "onnx.Reshape"(%arg0, %2) <{allowzero = 0 : si64}> : (tensor<?x?xi64>, tensor<2xi64>) -> tensor<?x?xi64>
    "onnx.DimGroup"(%3) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?xi64>) -> ()
    "onnx.DimGroup"(%3) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?xi64>) -> ()
    onnx.Return %3 : tensor<?x?xi64>
  }
}


// -----
module {
  func.func @test_expand_from_concat_dims(%arg0: tensor<1x256xi64>, %arg1: tensor<?x256xi64>) -> tensor<?x256xi64> {
    %0 = onnx.Constant dense<256> : tensor<1xi64>
    %1 = "onnx.Dim"(%arg1) <{axis = 0 : si64}> : (tensor<?x256xi64>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256xi64>) -> ()
    %2 = "onnx.Concat"(%1, %0) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = "onnx.Expand"(%arg0, %2) : (tensor<1x256xi64>, tensor<2xi64>) -> tensor<?x256xi64>
    "onnx.DimGroup"(%3) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x256xi64>) -> ()
    return %3 : tensor<?x256xi64>
  }
}


// -----
module {
  func.func @test_tile_input_dim_1(%arg0: tensor<?x?xi64>, %arg1: tensor<1x1xi64>) -> tensor<?x?xi64> {
    %0 = "onnx.Dim"(%arg0) <{axis = 0 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg0) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?xi64>) -> ()
    %1 = "onnx.Dim"(%arg0) <{axis = 1 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg0) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?xi64>) -> ()
    %2 = "onnx.Concat"(%0, %1) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = "onnx.Tile"(%arg1, %2) : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<?x?xi64>
    "onnx.DimGroup"(%3) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?xi64>) -> ()
    "onnx.DimGroup"(%3) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?xi64>) -> ()
    onnx.Return %3 : tensor<?x?xi64>
  }
}


// -----
module {
  func.func @test_center_crop_pad_1(%arg0: tensor<?x?x8xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x8xf32> {
    "onnx.DimGroup"(%arg1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?xf32>) -> ()
    %0 = "onnx.Dim"(%arg1) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
    %1 = "onnx.Dim"(%arg1) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg1) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?xf32>) -> ()
    %2 = "onnx.Concat"(%0, %1) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = "onnx.CenterCropPad"(%arg0, %2) <{axes = [0, -2]}> : (tensor<?x?x8xf32>, tensor<2xi64>) -> tensor<?x?x8xf32>
    "onnx.DimGroup"(%3) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?x8xf32>) -> ()
    "onnx.DimGroup"(%3) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?x8xf32>) -> ()
    return %3 : tensor<?x?x8xf32>
  }
}


// -----
module {
  func.func @test_center_crop_pad_2(%arg0: tensor<?x8x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x8x?xf32> {
    "onnx.DimGroup"(%arg1) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?xf32>) -> ()
    %0 = "onnx.Dim"(%arg1) <{axis = 0 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg1) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?xf32>) -> ()
    %1 = "onnx.Dim"(%arg1) <{axis = 1 : si64}> : (tensor<?x?xf32>) -> tensor<1xi64>
    %2 = "onnx.Concat"(%0, %1) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %3 = "onnx.CenterCropPad"(%arg0, %2) <{axes = [-3, 2]}> : (tensor<?x8x?xf32>, tensor<2xi64>) -> tensor<?x8x?xf32>
    "onnx.DimGroup"(%3) <{axis = 2 : si64, group_id = 1 : si64}> : (tensor<?x8x?xf32>) -> ()
    "onnx.DimGroup"(%3) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x8x?xf32>) -> ()
    return %3 : tensor<?x8x?xf32>
  }
}


// -----
module {
  func.func @test_max_unpool(%arg0: tensor<1x1x2x2xf32>, %arg1: tensor<1x1x2x2xi64>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    "onnx.DimGroup"(%arg2) <{axis = 3 : si64, group_id = 3 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    "onnx.DimGroup"(%arg2) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    %0 = "onnx.Dim"(%arg2) <{axis = 0 : si64}> : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
    %1 = "onnx.Dim"(%arg2) <{axis = 1 : si64}> : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg2) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    %2 = "onnx.Dim"(%arg2) <{axis = 2 : si64}> : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
    "onnx.DimGroup"(%arg2) <{axis = 2 : si64, group_id = 2 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    %3 = "onnx.Dim"(%arg2) <{axis = 3 : si64}> : (tensor<?x?x?x?xf32>) -> tensor<1xi64>
    %4 = "onnx.Concat"(%0, %1, %2, %3) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %5 = "onnx.MaxUnpool"(%arg0, %arg1, %4) <{kernel_shape = [2, 2], strides = [2, 2]}> : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xi64>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    "onnx.DimGroup"(%5) <{axis = 3 : si64, group_id = 3 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    "onnx.DimGroup"(%5) <{axis = 1 : si64, group_id = 1 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    "onnx.DimGroup"(%5) <{axis = 2 : si64, group_id = 2 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    "onnx.DimGroup"(%5) <{axis = 0 : si64, group_id = 0 : si64}> : (tensor<?x?x?x?xf32>) -> ()
    return %5 : tensor<?x?x?x?xf32>
  }
}


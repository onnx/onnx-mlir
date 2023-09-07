module {
}


// -----
module {
  func.func @test_default_maxpoolsingleout(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3, 3], storage_order = 0 : si64}> : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
    onnx.Return %0 : tensor<5x5x30x30xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_defpad(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], storage_order = 0 : si64}> : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
    onnx.Return %0 : tensor<5x5x30x30xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_pad(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64}> : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
    onnx.Return %0 : tensor<5x5x32x32xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_pad_nonunif(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5, 3], pads = [2, 1, 1, 0], storage_order = 0 : si64}> : (tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32>
    onnx.Return %0 : tensor<5x5x31x31xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_strides(%arg0: tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
    onnx.Return %0 : tensor<5x5x16x16xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_strides_nonunifpad(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
    onnx.Return %0 : tensor<5x5x15x16xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
    onnx.Return %0 : tensor<5x5x16x16xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_strides_dilatation(%arg0: tensor<5x5x8x8xf32>) -> tensor<5x5x2x2xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, dilations = [2, 2], kernel_shape = [2, 2], storage_order = 0 : si64, strides = [3, 3]}> : (tensor<5x5x8x8xf32>) -> tensor<5x5x2x2xf32>
    onnx.Return %0 : tensor<5x5x2x2xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_upper(%arg0: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "SAME_UPPER", ceil_mode = 0 : si64, kernel_shape = [4, 4], storage_order = 0 : si64, strides = [4, 4]}> : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
    onnx.Return %0 : tensor<5x5x4x4xf32>
  }
}


// -----
module {
  func.func @test_default_maxpoolsingleout_lower(%arg0: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) <{auto_pad = "SAME_LOWER", ceil_mode = 0 : si64, kernel_shape = [4, 4], storage_order = 0 : si64, strides = [4, 4]}> : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
    onnx.Return %0 : tensor<5x5x4x4xf32>
  }
}


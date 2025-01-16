// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --convert-onnx-to-zhigh --nnpa-quant-dynamic --constprop-onnx --canonicalize --mlir-print-elementsattrs-with-hex-if-larger=-1 %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --convert-onnx-to-zhigh --nnpa-quant-dynamic=symActivation,symWeight --constprop-onnx --canonicalize --mlir-print-elementsattrs-with-hex-if-larger=-1 %s -split-input-file | FileCheck %s --check-prefix=SYMSYMI8

func.func @test_correctness_of_symmetric_quant_for_weight(%arg0: tensor<?x?x200xf32>) -> tensor<?x?x1xf32> {
  %0 = onnx.Constant dense<[[-0.00718058366], [5.253110e-01], [-0.0434652828], [-0.305256933], [0.193365857], [0.0105065238], [-0.143788248], [-0.0161222648], [0.0230324212], [-0.34107244], [-0.273072243], [-0.104352467], [0.0164068397], [-1.32305741], [-0.0345043093], [-0.232206389], [-0.150001124], [0.119475454], [0.730642438], [-0.407772154], [-0.0164191965], [-1.625590e-01], [-0.112515017], [0.158920377], [-0.0997497215], [0.0788274407], [1.1542908], [0.492949218], [-0.125796661], [0.0107790371], [0.141159713], [-0.0774109289], [-0.438130081], [-0.0888700857], [0.207725927], [-0.0913108587], [0.258232892], [0.0672571063], [-0.100412264], [1.68460846], [-0.289168775], [-0.686722457], [0.903651654], [0.110602334], [-0.0505490415], [1.31204939], [0.136107579], [0.26376456], [-0.508291602], [-0.0118971812], [-0.0373991691], [0.448705465], [0.00448446581], [-0.165114298], [0.156860754], [0.141124308], [-0.272756487], [-0.0834815949], [0.020905681], [-0.0877983123], [-1.0087887], [-0.353012145], [-0.0439243801], [-0.00592191564], [-0.0637216269], [0.175808683], [-0.193864927], [-0.0574007072], [0.390869558], [0.138100505], [0.429396927], [1.10117233], [-0.362377733], [0.116578773], [0.0540139228], [-5.85162896E-4], [-0.335441321], [-0.0902953073], [0.017575942], [-0.0359748788], [1.50025952], [-0.668821096], [0.0109066488], [9.907780e-01], [0.10227681], [-0.0582750589], [0.0172416102], [0.0429656394], [0.0465254933], [0.350135148], [-0.260139734], [0.199394852], [-0.136131078], [0.241424322], [0.855418264], [-0.160689577], [-0.825074911], [-0.124827594], [0.0153419804], [0.389386117], [0.153694436], [-0.897866904], [-0.292769879], [0.181667477], [-0.188009143], [-0.0245181341], [-2.17088842], [-0.0526076891], [-0.108600065], [0.187120304], [0.171495944], [0.310159177], [2.204240e+00], [0.0506350659], [-0.159419239], [-0.145082235], [-0.0991335287], [-0.0680764392], [-0.311415762], [-0.187137261], [-0.416945577], [0.0703471377], [0.498331547], [-0.41216433], [-0.427900195], [0.102105901], [0.130767033], [-0.440281332], [0.778514624], [-0.253678083], [0.395671815], [0.380029172], [-0.418493837], [-0.288157403], [0.0689846799], [1.269960e+00], [-0.0585722439], [-0.138125435], [-0.191710189], [0.0163070802], [0.159242466], [0.116627224], [0.289637923], [-0.299413532], [-0.0216965247], [0.271396786], [0.250576884], [-0.131420374], [0.137698188], [-0.0102280416], [0.234722644], [-0.0366179943], [-0.105632246], [-0.145528033], [-0.278210133], [-0.247100428], [0.217718393], [0.171669215], [0.0151556451], [0.961385667], [-0.0484847203], [0.434219301], [-0.00167646946], [-0.0308207348], [-0.102328695], [-0.127907664], [-0.185960412], [0.210866481], [0.140434876], [-0.233541235], [-0.123745643], [-0.0113738365], [1.30043447], [0.179708347], [-0.331716627], [0.0133318678], [-0.107284561], [-0.114116102], [-0.478514463], [0.0616452768], [-0.781869769], [-0.121830635], [-0.0684970543], [-6.584100e-02], [-0.131784603], [-0.619898796], [0.160366163], [-0.50115186], [0.0228514839], [0.581515431], [4.220270e-01], [1.944400e-01], [-1.07740963], [3.732520e-01], [0.725471556], [-0.117193311], [-0.105938725], [0.320118755], [-0.484032601], [-0.0467250831]]> : tensor<200x1xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<?x?x200xf32>, tensor<200x1xf32>) -> tensor<?x?x1xf32>
  return %1 : tensor<?x?x1xf32>

// CHECK-LABEL:  func.func @test_correctness_of_symmetric_quant_for_weight
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x200xf32>) -> tensor<?x?x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.750000e+02> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{.}}[0], [30], [-3], [-18], [11], [1], [-8], [-1], [1], [-20], [-16], [-6], [1], [-76], [-2], [-13], [-9], [7], [42], [-23], [-1], [-9], [-6], [9], [-6], [5], [67], [28], [-7], [1], [8], [-4], [-25], [-5], [12], [-5], [15], [4], [-6], [97], [-17], [-40], [52], [6], [-3], [76], [8], [15], [-29], [-1], [-2], [26], [0], [-10], [9], [8], [-16], [-5], [1], [-5], [-58], [-20], [-3], [0], [-4], [10], [-11], [-3], [23], [8], [25], [63], [-21], [7], [3], [0], [-19], [-5], [1], [-2], [86], [-39], [1], [57], [6], [-3], [1], [2], [3], [20], [-15], [11], [-8], [14], [49], [-9], [-48], [-7], [1], [22], [9], [-52], [-17], [10], [-11], [-1], [-125], [-3], [-6], [11], [10], [18], [127], [3], [-9], [-8], [-6], [-4], [-18], [-11], [-24], [4], [29], [-24], [-25], [6], [8], [-25], [45], [-15], [23], [22], [-24], [-17], [4], [73], [-3], [-8], [-11], [1], [9], [7], [17], [-17], [-1], [16], [14], [-8], [8], [-1], [14], [-2], [-6], [-8], [-16], [-14], [13], [10], [1], [55], [-3], [25], [0], [-2], [-6], [-7], [-11], [12], [8], [-13], [-7], [-1], [75], [10], [-19], [1], [-6], [-7], [-28], [4], [-45], [-7], [-4], [-4], [-8], [-36], [9], [-29], [1], [34], [24], [11], [-62], [22], [42], [-7], [-6], [18], [-28], [-3]{{.}}> : tensor<200x1xi8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<57.61623> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_3_]], [[VAR_3_]]) {layout = "3DS", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<?x?x200xf32>, none, none) -> (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_1_]], [[VAR_2_]], [[VAR_4_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<200x1xi8>, tensor<f32>, tensor<f32>) -> (tensor<200x1xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:         }
}

// -----

func.func @test_correctness_of_symmetric_quant_for_activation_and_weight(%arg0: tensor<?x?x200xf32>) -> tensor<?x?x1xf32> {
  %0 = onnx.Constant dense<[[-0.00718058366], [5.253110e-01], [-0.0434652828], [-0.305256933], [0.193365857], [0.0105065238], [-0.143788248], [-0.0161222648], [0.0230324212], [-0.34107244], [-0.273072243], [-0.104352467], [0.0164068397], [-1.32305741], [-0.0345043093], [-0.232206389], [-0.150001124], [0.119475454], [0.730642438], [-0.407772154], [-0.0164191965], [-1.625590e-01], [-0.112515017], [0.158920377], [-0.0997497215], [0.0788274407], [1.1542908], [0.492949218], [-0.125796661], [0.0107790371], [0.141159713], [-0.0774109289], [-0.438130081], [-0.0888700857], [0.207725927], [-0.0913108587], [0.258232892], [0.0672571063], [-0.100412264], [1.68460846], [-0.289168775], [-0.686722457], [0.903651654], [0.110602334], [-0.0505490415], [1.31204939], [0.136107579], [0.26376456], [-0.508291602], [-0.0118971812], [-0.0373991691], [0.448705465], [0.00448446581], [-0.165114298], [0.156860754], [0.141124308], [-0.272756487], [-0.0834815949], [0.020905681], [-0.0877983123], [-1.0087887], [-0.353012145], [-0.0439243801], [-0.00592191564], [-0.0637216269], [0.175808683], [-0.193864927], [-0.0574007072], [0.390869558], [0.138100505], [0.429396927], [1.10117233], [-0.362377733], [0.116578773], [0.0540139228], [-5.85162896E-4], [-0.335441321], [-0.0902953073], [0.017575942], [-0.0359748788], [1.50025952], [-0.668821096], [0.0109066488], [9.907780e-01], [0.10227681], [-0.0582750589], [0.0172416102], [0.0429656394], [0.0465254933], [0.350135148], [-0.260139734], [0.199394852], [-0.136131078], [0.241424322], [0.855418264], [-0.160689577], [-0.825074911], [-0.124827594], [0.0153419804], [0.389386117], [0.153694436], [-0.897866904], [-0.292769879], [0.181667477], [-0.188009143], [-0.0245181341], [-2.17088842], [-0.0526076891], [-0.108600065], [0.187120304], [0.171495944], [0.310159177], [2.204240e+00], [0.0506350659], [-0.159419239], [-0.145082235], [-0.0991335287], [-0.0680764392], [-0.311415762], [-0.187137261], [-0.416945577], [0.0703471377], [0.498331547], [-0.41216433], [-0.427900195], [0.102105901], [0.130767033], [-0.440281332], [0.778514624], [-0.253678083], [0.395671815], [0.380029172], [-0.418493837], [-0.288157403], [0.0689846799], [1.269960e+00], [-0.0585722439], [-0.138125435], [-0.191710189], [0.0163070802], [0.159242466], [0.116627224], [0.289637923], [-0.299413532], [-0.0216965247], [0.271396786], [0.250576884], [-0.131420374], [0.137698188], [-0.0102280416], [0.234722644], [-0.0366179943], [-0.105632246], [-0.145528033], [-0.278210133], [-0.247100428], [0.217718393], [0.171669215], [0.0151556451], [0.961385667], [-0.0484847203], [0.434219301], [-0.00167646946], [-0.0308207348], [-0.102328695], [-0.127907664], [-0.185960412], [0.210866481], [0.140434876], [-0.233541235], [-0.123745643], [-0.0113738365], [1.30043447], [0.179708347], [-0.331716627], [0.0133318678], [-0.107284561], [-0.114116102], [-0.478514463], [0.0616452768], [-0.781869769], [-0.121830635], [-0.0684970543], [-6.584100e-02], [-0.131784603], [-0.619898796], [0.160366163], [-0.50115186], [0.0228514839], [0.581515431], [4.220270e-01], [1.944400e-01], [-1.07740963], [3.732520e-01], [0.725471556], [-0.117193311], [-0.105938725], [0.320118755], [-0.484032601], [-0.0467250831]]> : tensor<200x1xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<?x?x200xf32>, tensor<200x1xf32>) -> tensor<?x?x1xf32>
  return %1 : tensor<?x?x1xf32>

// SYMSYMI8-LABEL:  func.func @test_correctness_of_symmetric_quant_for_activation_and_weight
// SYMSYMI8-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x200xf32>) -> tensor<?x?x1xf32> {
// SYMSYMI8-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[0], [30], [-3], [-18], [11], [1], [-8], [-1], [1], [-20], [-16], [-6], [1], [-76], [-2], [-13], [-9], [7], [42], [-23], [-1], [-9], [-6], [9], [-6], [5], [67], [28], [-7], [1], [8], [-4], [-25], [-5], [12], [-5], [15], [4], [-6], [97], [-17], [-40], [52], [6], [-3], [76], [8], [15], [-29], [-1], [-2], [26], [0], [-10], [9], [8], [-16], [-5], [1], [-5], [-58], [-20], [-3], [0], [-4], [10], [-11], [-3], [23], [8], [25], [63], [-21], [7], [3], [0], [-19], [-5], [1], [-2], [86], [-39], [1], [57], [6], [-3], [1], [2], [3], [20], [-15], [11], [-8], [14], [49], [-9], [-48], [-7], [1], [22], [9], [-52], [-17], [10], [-11], [-1], [-125], [-3], [-6], [11], [10], [18], [127], [3], [-9], [-8], [-6], [-4], [-18], [-11], [-24], [4], [29], [-24], [-25], [6], [8], [-25], [45], [-15], [23], [22], [-24], [-17], [4], [73], [-3], [-8], [-11], [1], [9], [7], [17], [-17], [-1], [16], [14], [-8], [8], [-1], [14], [-2], [-6], [-8], [-16], [-14], [13], [10], [1], [55], [-3], [25], [0], [-2], [-6], [-7], [-11], [12], [8], [-13], [-7], [-1], [75], [10], [-19], [1], [-6], [-7], [-28], [4], [-45], [-7], [-4], [-4], [-8], [-36], [9], [-29], [1], [34], [24], [11], [-62], [22], [42], [-7], [-6], [18], [-28], [-3]{{.}}> : tensor<200x1xi8>
// SYMSYMI8-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<57.61623> : tensor<f32>
// SYMSYMI8-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// SYMSYMI8-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// SYMSYMI8-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// SYMSYMI8:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_2_]], [[VAR_2_]]) {layout = "3DS", quantized_type = "DLFLOAT16", sym_mode = 1 : i64} : (tensor<?x?x200xf32>, none, none) -> (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// SYMSYMI8:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_0_]], [[VAR_1_]], [[VAR_3_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<200x1xi8>, tensor<f32>, tensor<f32>) -> (tensor<200x1xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// SYMSYMI8:           [[VAR_Out_3_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_1_]], [[VAR_3_]], [[VAR_2_]], [[VAR_4_]], [[VAR_3_]], [[VAR_4_]], [[VAR_3_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<200x1xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, none, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x1xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// SYMSYMI8:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_Out_3_]]) : (tensor<?x?x1xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x1xf32>
// SYMSYMI8:           return [[VAR_5_]] : tensor<?x?x1xf32>
// SYMSYMI8:         }
}

// -----

func.func @test_matmul(%arg0: tensor<?x?x200xf32>) -> tensor<?x?x1xf32> {
  %0 = onnx.Constant dense<-0.00718058366> : tensor<200x1xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<?x?x200xf32>, tensor<200x1xf32>) -> tensor<?x?x1xf32>
  return %1 : tensor<?x?x1xf32>

// CHECK-LABEL:  func.func @test_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x200xf32>) -> tensor<?x?x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-2.540000e+04> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-127> : tensor<200x1xi8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<17686.584> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_3_]], [[VAR_3_]]) {layout = "3DS", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<?x?x200xf32>, none, none) -> (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_1_]], [[VAR_2_]], [[VAR_4_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<200x1xi8>, tensor<f32>, tensor<f32>) -> (tensor<200x1xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_6_:%.+]] = "onnx.Div"([[VAR_5_]], [[VAR_RecScale_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Div"([[VAR_6_]], [[VAR_2_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Mul"([[VAR_7_]], [[VAR_Offset_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Sub"([[VAR_4_]], [[VAR_8_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Mul"([[VAR_9_]], [[VAR_0_]]) : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[VAR_10_]], [[VAR_5_]], [[VAR_4_]]) {layout = "1D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<1xf32>, tensor<f32>, tensor<f32>) -> (tensor<1xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_6_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_2_]], [[VAR_4_]], [[VAR_Out_]]_3, [[VAR_RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_5_]], [[VAR_4_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<200x1xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<1xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x1xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_Out_6_]]) : (tensor<?x?x1xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x1xf32>
// CHECK:           return [[VAR_11_]] : tensor<?x?x1xf32>
// CHECK:         }
}

// -----

func.func @test_matmul_add(%arg0: tensor<?x?x200xf32>) -> tensor<?x?x10xf32> {
  %0 = onnx.Constant dense<-0.00718058366> : tensor<200x10xf32>
  %1 = onnx.Constant dense<-0.00718058366> : tensor<10xf32>
  %2 = "onnx.MatMul"(%arg0, %0) : (tensor<?x?x200xf32>, tensor<200x10xf32>) -> tensor<?x?x10xf32>
  %3 = "onnx.Add"(%2, %1): (tensor<?x?x10xf32>, tensor<10xf32>) -> tensor<?x?x10xf32>
  return %3 : tensor<?x?x10xf32>

// CHECK-LABEL:  func.func @test_matmul_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x200xf32>) -> tensor<?x?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-2.540000e+04> : tensor<10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-127> : tensor<200x10xi8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<17686.584> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<-0.00718058366> : tensor<10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_4_]], [[VAR_4_]]) {layout = "3DS", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<?x?x200xf32>, none, none) -> (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_1_]], [[VAR_2_]], [[VAR_5_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<200x10xi8>, tensor<f32>, tensor<f32>) -> (tensor<200x10xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_7_:%.+]] = "onnx.Div"([[VAR_6_]], [[VAR_RecScale_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Div"([[VAR_7_]], [[VAR_2_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Mul"([[VAR_8_]], [[VAR_Offset_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Sub"([[VAR_5_]], [[VAR_9_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Mul"([[VAR_10_]], [[VAR_0_]]) : (tensor<f32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[VAR_11_]], [[VAR_6_]], [[VAR_5_]]) {layout = "1D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<10xf32>, tensor<f32>, tensor<f32>) -> (tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Stick"([[VAR_3_]]) {layout = "1D"} : (tensor<10xf32>) -> tensor<10xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_13_:%.+]] = "zhigh.Add"([[VAR_Out_3_]], [[VAR_12_]]) : (tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<10xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>
// CHECK:           [[VAR_Out_6_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_2_]], [[VAR_5_]], [[VAR_13_]], [[VAR_RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_6_]], [[VAR_5_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x?x200xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<200x10xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x?x10xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_14_:%.+]] = "zhigh.Unstick"([[VAR_Out_6_]]) : (tensor<?x?x10xf16, #zhigh.layout<{dataLayout = "3DS", quantizedType = "DLFLOAT16"}>>) -> tensor<?x?x10xf32>
// CHECK:           return [[VAR_14_]] : tensor<?x?x10xf32>
// CHECK:         }
}

// -----

func.func @test_gemm(%arg0: tensor<?x200xf32>) -> tensor<?x10xf32> {
  %0 = onnx.Constant dense<-0.00718058366> : tensor<200x10xf32>
  %1 = onnx.Constant dense<-0.00718058366> : tensor<10xf32>
  %2 = "onnx.Gemm"(%arg0, %0, %1) {transA = 0 : si64, transB = 0 : si64, alpha = 1.0 : f32, beta = 1.0 : f32} : (tensor<?x200xf32>, tensor<200x10xf32>, tensor<10xf32>) -> tensor<?x10xf32>
  return %2 : tensor<?x10xf32>

// CHECK-LABEL:  func.func @test_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x200xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-2.540000e+04> : tensor<10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-127> : tensor<200x10xi8>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<17686.584> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<-0.00718058366> : tensor<10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_Out_:%.+]], [[VAR_RecScale_:%.+]], [[VAR_Offset_:%.+]] = "zhigh.QuantizedStick"([[PARAM_0_]], [[VAR_4_]], [[VAR_4_]]) {layout = "2D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<?x200xf32>, none, none) -> (tensor<?x200xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_Out_0_:%.+]], [[VAR_RecScale_1_:%.+]], [[VAR_Offset_2_:%.+]] = "zhigh.QuantizedStick"([[VAR_1_]], [[VAR_2_]], [[VAR_5_]]) {layout = "2D", quantized_type = "WEIGHTS", sym_mode = 0 : i64} : (tensor<200x10xi8>, tensor<f32>, tensor<f32>) -> (tensor<200x10xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_7_:%.+]] = "onnx.Div"([[VAR_6_]], [[VAR_RecScale_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Div"([[VAR_7_]], [[VAR_2_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Mul"([[VAR_8_]], [[VAR_Offset_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Sub"([[VAR_5_]], [[VAR_9_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Mul"([[VAR_10_]], [[VAR_0_]]) : (tensor<f32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK:           [[VAR_Out_3_:%.+]], [[VAR_RecScale_4_:%.+]], [[VAR_Offset_5_:%.+]] = "zhigh.QuantizedStick"([[VAR_11_]], [[VAR_6_]], [[VAR_5_]]) {layout = "1D", quantized_type = "DLFLOAT16", sym_mode = 0 : i64} : (tensor<10xf32>, tensor<f32>, tensor<f32>) -> (tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_12_:%.+]] = "zhigh.Stick"([[VAR_3_]]) {layout = "1D"} : (tensor<10xf32>) -> tensor<10xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_13_:%.+]] = "zhigh.Add"([[VAR_Out_3_]], [[VAR_12_]]) : (tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<10xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>
// CHECK:           [[VAR_Out_6_:%.+]], [[VAR_OutRecScale_:%.+]], [[VAR_OutOffset_:%.+]] = "zhigh.QuantizedMatMul"([[VAR_Out_]], [[VAR_RecScale_]], [[VAR_Offset_]], [[VAR_Out_]]_0, [[VAR_2_]], [[VAR_5_]], [[VAR_13_]], [[VAR_RecScale_]]_4, [[VAR_Offset_]]_5, [[VAR_6_]], [[VAR_5_]]) {DequantizeOutput = 0 : si64, DisableClipping = -1 : si64, PreComputedBias = -1 : si64} : (tensor<?x200xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<200x10xi8, #zhigh.layout<{dataLayout = "2D", quantizedType = "WEIGHTS"}>>, tensor<f32>, tensor<f32>, tensor<10xf16, #zhigh.layout<{dataLayout = "1D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>, tensor<f32>, tensor<f32>)
// CHECK:           [[VAR_14_:%.+]] = "zhigh.Unstick"([[VAR_Out_6_]]) : (tensor<?x10xf16, #zhigh.layout<{dataLayout = "2D", quantizedType = "DLFLOAT16"}>>) -> tensor<?x10xf32>
// CHECK:           return [[VAR_14_]] : tensor<?x10xf32>
// CHECK:         }
}

// -----

// Do not quantize because B is not a constant.
func.func @test_matmul_not_quantized(%arg0: tensor<?x?x200xf32>, %arg1: tensor<200x1xf32>) -> tensor<?x?x1xf32> {
  %1 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x200xf32>, tensor<200x1xf32>) -> tensor<?x?x1xf32>
  return %1 : tensor<?x?x1xf32>

// CHECK-LABEL:  func.func @test_matmul_not_quantized
// CHECK:        "zhigh.MatMul"
// CHECK-NOT:    "zhigh.QuantizedMatMul"
}

// -----

// Do not quantize because C is not a constant.
func.func @test_matmul_add_not_quantized(%arg0: tensor<?x?x200xf32>, %arg1: tensor<10xf32>) -> tensor<?x?x10xf32> {
  %0 = onnx.Constant dense<-0.00718058366> : tensor<200x10xf32>
  %1 = "onnx.MatMul"(%arg0, %0) : (tensor<?x?x200xf32>, tensor<200x10xf32>) -> tensor<?x?x10xf32>
  %2 = "onnx.Add"(%1, %arg1): (tensor<?x?x10xf32>, tensor<10xf32>) -> tensor<?x?x10xf32>
  return %2 : tensor<?x?x10xf32>

// CHECK-LABEL:  func.func @test_matmul_add_not_quantized
// CHECK:        "zhigh.MatMul"
// CHECK-NOT:    "zhigh.QuantizedMatMul"
}

// -----

// Do not quantize because A is transposed.
func.func @test_gemm_not_quantized(%arg0: tensor<200x?xf32>) -> tensor<?x10xf32> {
  %0 = onnx.Constant dense<-0.00718058366> : tensor<200x10xf32>
  %1 = onnx.Constant dense<-0.00718058366> : tensor<10xf32>
  %2 = "onnx.Gemm"(%arg0, %0, %1) {transA = 1 : si64, transB = 0 : si64, alpha = 1.0 : f32, beta = 1.0 : f32} : (tensor<200x?xf32>, tensor<200x10xf32>, tensor<10xf32>) -> tensor<?x10xf32>
  return %2 : tensor<?x10xf32>

// CHECK-LABEL:  func.func @test_gemm_not_quantized
// CHECK:        "zhigh.MatMul"
// CHECK-NOT:    "zhigh.QuantizedMatMul"
}


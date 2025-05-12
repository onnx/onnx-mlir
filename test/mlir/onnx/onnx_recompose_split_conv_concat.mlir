// RUN: onnx-mlir-opt --recompose-onnx --remove-dead-values --constprop-onnx %s -split-input-file | FileCheck %s

func.func @simple_split_conv_concat(%arg0: tensor<1x6x512x512xf64> {onnx.name = "input"}) -> (tensor<1x6x512x512xf64> {onnx.name = "output"}) {
  %0 = onnx.Constant dense<[[[[-0.0017646604683250189, 0.12644097208976746, -0.19399359822273254], [-0.17346249520778656, -0.090781755745410919, 0.0632052943110466], [-0.0046700113452970982, 0.18688584864139557, -0.020917171612381935]], [[0.062369778752326965, -0.071232303977012634, -0.046330906450748444], [-0.22517779469490051, -0.15610139071941376, -0.097161918878555298], [0.008731253445148468, 0.093181401491165161, 0.14142672717571259]]], [[[-0.15979224443435669, -0.1026395708322525, 0.085611097514629364], [0.19572432339191437, -0.048507567495107651, 0.1763787716627121], [-0.037991281598806381, 0.024940622970461845, 0.21342279016971588]], [[-0.21865400671958923, -0.14838351309299469, -0.059671621769666672], [-0.09187673032283783, 0.2036469429731369, -0.15277740359306335], [-0.10850150138139725, -0.16467113792896271, -0.22074954211711884]]]]> : tensor<2x2x3x3xf64>
  %1 = onnx.Constant dense<[-0.13758894801139832, 0.20260919630527496]> : tensor<2xf64>
  %2 = onnx.Constant dense<[[[[0.10517467558383942, 0.11423841863870621, 0.01239595003426075], [-0.12084066122770309, 0.039877213537693024, -0.22007395327091217], [-0.17031049728393555, -0.12151158601045609, 0.14871349930763245]], [[0.13819724321365356, -0.10453278571367264, -0.0085046999156475067], [0.15074589848518372, 0.23431941866874695, 0.093546025454998016], [0.031841691583395004, 0.15803514420986176, -0.13878203928470612]]], [[[0.043921709060668945, -0.18274125456809998, -0.16336196660995483], [-0.12175991386175156, 0.10664892196655273, 0.09479011595249176], [-0.13961882889270782, 0.071207322180271149, 0.12939395010471344]], [[-0.029749717563390732, 0.0089994762092828751, 0.054613325744867325], [0.14622417092323303, 0.22631992399692535, -0.1816377192735672], [-0.086377747356891632, 0.09263332188129425, 0.19529096782207489]]]]> : tensor<2x2x3x3xf64>
  %3 = onnx.Constant dense<[0.20510983467102051, 0.20797348022460938]> : tensor<2xf64>
  %4 = onnx.Constant dense<[[[[0.046908177435398102, -0.2049625962972641, 0.021682839840650558], [-0.14745660126209259, -0.21966369450092316, 0.20941968262195587], [0.17921851575374603, -0.23511959612369537, 0.044116877019405365]], [[-0.039706405252218246, -0.038787435740232468, -0.10789433121681213], [0.090640760958194732, -0.13960728049278259, 0.086406409740447998], [0.11919654160737991, 0.16873255372047424, 0.088131703436374664]]], [[[-0.23328283429145813, -0.15289932489395142, 0.11768967658281326], [0.049332801252603531, -0.18386755883693695, -0.13572195172309875], [0.22173672914505005, 0.15882039070129395, -0.10277210921049118]], [[-0.059322673827409744, -0.22452951967716217, -0.0042365449480712414], [-0.17749768495559692, -0.18181051313877106, -0.012987101450562477], [0.035389527678489685, -0.096527211368083953, 0.13986043632030487]]]]> : tensor<2x2x3x3xf64>
  %5 = onnx.Constant dense<[-0.14343404769897461, 0.21386918425559998]> : tensor<2xf64>
  %6 = onnx.Constant dense<2> : tensor<3xi64>
  %7:3 = "onnx.Split"(%arg0, %6) {axis = 1 : si64} : (tensor<1x6x512x512xf64>, tensor<3xi64>) -> (tensor<1x2x512x512xf64>, tensor<1x2x512x512xf64>, tensor<1x2x512x512xf64>)
  %8 = "onnx.Conv"(%7#0, %0, %1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/conv1/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x2x512x512xf64>, tensor<2x2x3x3xf64>, tensor<2xf64>) -> tensor<1x2x512x512xf64>
  %9 = "onnx.Conv"(%7#1, %2, %3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/conv2/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x2x512x512xf64>, tensor<2x2x3x3xf64>, tensor<2xf64>) -> tensor<1x2x512x512xf64>
  %10 = "onnx.Conv"(%7#2, %4, %5) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/conv3/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x2x512x512xf64>, tensor<2x2x3x3xf64>, tensor<2xf64>) -> tensor<1x2x512x512xf64>
  %11 = "onnx.Concat"(%8, %9, %10) {axis = 1 : si64, onnx_node_name = "/Concat"} : (tensor<1x2x512x512xf64>, tensor<1x2x512x512xf64>, tensor<1x2x512x512xf64>) -> tensor<1x6x512x512xf64>
  onnx.Return %11 : tensor<1x6x512x512xf64>

  // CHECK: func.func @simple_split_conv_concat(%[[ARG0:.*]]: tensor<1x6x512x512xf64>{{.*}}) -> (tensor<1x6x512x512xf64>{{.*}})
  // CHECK: %[[Weights:.*]] = onnx.Constant dense<{{.*}}> : tensor<6x2x3x3xf64>
  // CHECK: %[[Bias:.*]] = onnx.Constant dense<{{.*}}> : tensor<6xf64>
  // CHECK: %[[CONV:.*]] = "onnx.Conv"(%[[ARG0]], %[[Weights]], %[[Bias]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 3 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x6x512x512xf64>, tensor<6x2x3x3xf64>, tensor<6xf64>) -> tensor<1x6x512x512xf64>
  // CHECK: onnx.Return %[[CONV]] : tensor<1x6x512x512xf64>
}

func.func @uneven_split(%arg0: tensor<1x3x256x256xf32> {onnx.name = "input"}) -> (tensor<1x2x256x256xf32> {onnx.name = "output"}) {
  %0 = onnx.Constant dense<[[[[-0.0439920649, 0.157494396, -0.218597859], [0.216857567, 0.0915632173, -0.0249686651], [-0.148716137, -0.113740437, -0.135975227]], [[0.0759392082, 0.211321741, 0.188139483], [0.0779103636, 0.11157462, -0.038455233], [-0.0563982166, 0.103472814, -0.2151196]]]]> : tensor<1x2x3x3xf32>
  %1 = onnx.Constant dense<-0.0471580811> : tensor<1xf32>
  %2 = onnx.Constant dense<[[[[0.211627096, -0.246834278, -0.0634299144], [-0.0321794376, -0.302116245, -0.283898681], [-1.724050e-01, 0.0552624874, -0.291402549]]]]> : tensor<1x1x3x3xf32>
  %3 = onnx.Constant dense<0.122131944> : tensor<1xf32>
  %4 = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %5:2 = "onnx.Split"(%arg0, %4) {axis = 1 : si64} : (tensor<1x3x256x256xf32>, tensor<2xi64>) -> (tensor<1x2x256x256xf32>, tensor<1x1x256x256xf32>)
  %6 = "onnx.Conv"(%5#0, %0, %1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/convs.0/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x2x256x256xf32>, tensor<1x2x3x3xf32>, tensor<1xf32>) -> tensor<1x1x256x256xf32>
  %7 = "onnx.Conv"(%5#1, %2, %3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/convs.1/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x256x256xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x256x256xf32>
  %8 = "onnx.Concat"(%6, %7) {axis = 1 : si64, onnx_node_name = "/Concat"} : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x2x256x256xf32>
  onnx.Return %8 : tensor<1x2x256x256xf32>

// CHECK: func.func @uneven_split(%[[ARG0:.*]]: tensor<1x3x256x256xf32> {{.*}}) -> (tensor<1x2x256x256xf32> {{.*}})
//Ensuring the pass is not applies as the weights are not concatenated
// CHECK: %[[CONST1:.*]] = onnx.Constant dense<{{.*}}> : tensor<1x2x3x3xf32>
// CHECK: %[[CONST2:.*]] = onnx.Constant dense<{{.*}}> : tensor<1xf32>
// CHECK: %[[CONST3:.*]] = onnx.Constant dense<{{.*}}> : tensor<1x1x3x3xf32>
// CHECK: %[[CONST4:.*]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK: %[[SPLIT_TENSOR:.*]]:2 = "onnx.Split"(%[[ARG0]], %[[CONST4]]) {axis = 1 : si64} : (tensor<1x3x256x256xf32>, tensor<2xi64>) -> (tensor<1x2x256x256xf32>, tensor<1x1x256x256xf32>)
}

/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Auto-Generated, do not change  ---------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================

bool isFasterOnNnpa_Add_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 58292
  // Regression for CPU with r2 = 0.9998796798458582
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.3014297439117151e-10 * complexityCpu + 2.0196878739035375e-07;
  // Regression for NNPA with r2 = 0.9946449269392905
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0267638124762705e-10 *  complexityNnpa + 1.8030585573544791e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Div_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 2483
  // Regression for CPU with r2 = 0.9999989217222092
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.4517483410281062e-09 * complexityCpu + 4.819629870926124e-07;
  // Regression for NNPA with r2 = 0.993964642126504
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0448506395503133e-10 *  complexityNnpa + 3.8276566890878624e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Exp_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = -1037
  // Regression for CPU with r2 = 0.9953451710472131
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 3.728416286592805e-09 * complexityCpu + 6.0588917718826046e-06;
  // Regression for NNPA with r2 = 0.9891659822712001
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 8.495396362536427e-11 *  complexityNnpa + 2.2771053257892673e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Log_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 214
  // Regression for CPU with r2 = 0.9999978790893227
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 8.437935291413468e-09 * complexityCpu + -5.198279741842633e-07;
  // Regression for NNPA with r2 = 0.9860021150687228
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 9.34925048506708e-11 *  complexityNnpa + 1.2737044319477467e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_MatMul_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 30626
  // Regression for CPU with r2 = 0.996620949480741
  double B=e3, N=e2, M=e1, K=e1;
  double complexityCpu = B * (N * M * K);
  double estimatedCpuTime = 8.612350895658484e-11 * complexityCpu + 1.718603370623734e-07;
  // Regression for NNPA with r2 = 0.915934452490499
  double complexityNnpa = B * (ceil(N/32.0)*32.0) * (ceil(M/64.0)*64.0) * (ceil(K/64.0)*64.0);
  double estimatedNnpaTime = 4.042257675757775e-12 *  complexityNnpa + 2.6857598605769233e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Max_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 28928
  // Regression for CPU with r2 = 0.999833820537654
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.2859513298352582e-10 * complexityCpu + 2.1445698817257752e-07;
  // Regression for NNPA with r2 = 0.9951863938046909
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0420243623790107e-10 *  complexityNnpa + 9.20104282137138e-07;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Min_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 99463
  // Regression for CPU with r2 = 0.9998504038595571
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.28567886153589e-10 * complexityCpu + 2.0245445263399963e-07;
  // Regression for NNPA with r2 = 0.9946372532381683
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0339073648992919e-10 *  complexityNnpa + 2.70667430925335e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Mul_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 55288
  // Regression for CPU with r2 = 0.9998538384140817
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.2961833783614705e-10 * complexityCpu + 2.2021597368460279e-07;
  // Regression for NNPA with r2 = 0.9944216212877205
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0281908200330892e-10 *  complexityNnpa + 1.7019104732413162e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Pow_2_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 56111
  // Regression for CPU with r2 = 0.9998705421933038
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.2964568661871788e-10 * complexityCpu + 2.0823276086876973e-07;
  // Regression for NNPA with r2 = 0.9945018708254405
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0317090324802998e-10 *  complexityNnpa + 1.693775697889311e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Pow_3_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 21706
  // Regression for CPU with r2 = 0.9997984857078572
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 3.7494455439876194e-10 * complexityCpu + -8.306238888444397e-07;
  // Regression for NNPA with r2 = 0.9411544973356172
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 2.085058530670984e-10 *  complexityNnpa + 2.7822286559924233e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Pow_4_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 41065
  // Regression for CPU with r2 = 0.9998522920538492
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 2.5405902189076053e-10 * complexityCpu + 1.143001005006861e-07;
  // Regression for NNPA with r2 = 0.9978757256268103
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.811269802457481e-10 *  complexityNnpa + 3.109286345912229e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Pow_8_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 38894
  // Regression for CPU with r2 = 0.9998492647347293
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 3.784270893083021e-10 * complexityCpu + 3.846063745069773e-08;
  // Regression for NNPA with r2 = 0.9984689810839804
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 2.5809839682053116e-10 *  complexityNnpa + 4.718642808461304e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Relu_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 12461
  // Regression for CPU with r2 = 0.9999336782083352
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.287932373847373e-10 * complexityCpu + 2.1978022850964433e-07;
  // Regression for NNPA with r2 = 0.9932963892258988
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 8.174802287383875e-11 *  complexityNnpa + 8.060373976318752e-07;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Sigmoid_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 605
  // Regression for CPU with r2 = 0.9999988107319201
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 5.386219204160899e-09 * complexityCpu + 5.634215435092746e-07;
  // Regression for NNPA with r2 = 0.9723030037974931
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0653822523387642e-10 *  complexityNnpa + 3.7620795110656856e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Softmax_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 3256
  // Regression for CPU with r2 = 0.9997073553561724
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 6.319129534911603e-09 * complexityCpu + 2.8581895675517657e-06;
  // Regression for NNPA with r2 = 0.7561317069552476
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.5370950144430126e-09 *  complexityNnpa + 1.8432846457914885e-05;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Sub_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 27788
  // Regression for CPU with r2 = 0.9998718450512123
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 1.2958900359490862e-10 * complexityCpu + 2.0747306461438247e-07;
  // Regression for NNPA with r2 = 0.9949061007697938
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 1.0456801418978144e-10 *  complexityNnpa + 9.027743246689109e-07;
  return estimatedNnpaTime < estimatedCpuTime;
}

bool isFasterOnNnpa_Tanh_3ds(double e3, double e2, double e1) {
  // Operation has cross over at complexity = 474
  // Regression for CPU with r2 = 0.999998963010189
  double complexityCpu = e3 * e2 * e1;
  double estimatedCpuTime = 5.609614699943962e-09 * complexityCpu + 1.8748266930146244e-07;
  // Regression for NNPA with r2 = 0.9897555127053886
  double complexityNnpa = e3 * (ceil(e2/32.0)*32.0) * (ceil(e1/64.0)*64.0);
  double estimatedNnpaTime = 8.058830271076489e-11 *  complexityNnpa + 2.8126039207664257e-06;
  return estimatedNnpaTime < estimatedCpuTime;
}

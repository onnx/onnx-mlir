/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Auto-Generated, do not change  ---------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================

inline static double ms_ceiling(double n, double m) { return ceil(n / m) * m; }

// Operation Add_3ds: estimated times with cross over at complexity = 58292.
double estimatedTimeForCPU_Add_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998796798458582
  return 1.3014297439117151e-10 * complexity + 2.0196878739035375e-07;
}
double estimatedTimeForNNPA_Add_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9946449269392905
  return 1.0267638124762705e-10 * complexity + 1.8030585573544791e-06;
}

// Operation Div_3ds: estimated times with cross over at complexity = 2483.
double estimatedTimeForCPU_Div_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999989217222092
  return 1.4517483410281062e-09 * complexity + 4.819629870926124e-07;
}
double estimatedTimeForNNPA_Div_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.993964642126504
  return 1.0448506395503133e-10 * complexity + 3.8276566890878624e-06;
}

// Operation Exp_3ds: estimated times with cross over at complexity = -1037.
double estimatedTimeForCPU_Exp_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9953451710472131
  return 3.728416286592805e-09 * complexity + 6.0588917718826046e-06;
}
double estimatedTimeForNNPA_Exp_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9891659822712001
  return 8.495396362536427e-11 * complexity + 2.2771053257892673e-06;
}

// Operation Log_3ds: estimated times with cross over at complexity = 214.
double estimatedTimeForCPU_Log_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999978790893227
  return 8.437935291413468e-09 * complexity + -5.198279741842633e-07;
}
double estimatedTimeForNNPA_Log_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9860021150687228
  return 9.34925048506708e-11 * complexity + 1.2737044319477467e-06;
}

// Operation MatMul_3ds: estimated times with cross over at complexity = 30745.
double estimatedTimeForCPU_MatMul_3ds(double B, double N, double M, double K) {
  double complexity = B * (N * M * K);
  // Regression for CPU with r2 = 0.9977563719224963
  return 8.516416082324592e-11 * complexity + 2.813872063426722e-07;
}
double estimatedTimeForNNPA_MatMul_3ds(double B, double N, double M, double K) {
  double complexity =
      B * ms_ceiling(N, 2.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  // Regression for NNPA with r2 = 0.6995070738236333
  return 6.1299740116770674e-12 * complexity + 2.7113063170731707e-06;
}

// Operation MatMul_all_3ds: estimated times with cross over at complexity =
// 32057.
double estimatedTimeForCPU_MatMul_all_3ds(
    double B, double N, double M, double K) {
  double complexity = B * (N * M * K);
  // Regression for CPU with r2 = 0.9907461032132769
  return 8.557471095127001e-11 * complexity + 2.651819741343577e-07;
}
double estimatedTimeForNNPA_MatMul_all_3ds(
    double B, double N, double M, double K) {
  double complexity =
      B * ms_ceiling(N, 2.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  // Regression for NNPA with r2 = 0.6007285904343204
  return 4.573189269510686e-12 * complexity + 2.8618706688323212e-06;
}

// Operation MatMul_e1_3ds: estimated times with cross over at complexity =
// 19563.
double estimatedTimeForCPU_MatMul_e1_3ds(
    double B, double N, double M, double K) {
  double complexity = B * (N * M * K);
  // Regression for CPU with r2 = 0.994413521405784
  return 8.705012847704432e-11 * complexity + 1.9122006955294004e-06;
}
double estimatedTimeForNNPA_MatMul_e1_3ds(
    double B, double N, double M, double K) {
  double complexity =
      B * ms_ceiling(N, 2.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  // Regression for NNPA with r2 = 0.8949993703987157
  return 1.9660671208761047e-12 * complexity + 3.5767057707073767e-06;
}

// Operation MatMul_e2_3ds: estimated times with cross over at complexity =
// 33816.
double estimatedTimeForCPU_MatMul_e2_3ds(
    double B, double N, double M, double K) {
  double complexity = B * (N * M * K);
  // Regression for CPU with r2 = 0.9994614139056304
  return 8.39578711725025e-11 * complexity + 2.739079754718904e-07;
}
double estimatedTimeForNNPA_MatMul_e2_3ds(
    double B, double N, double M, double K) {
  double complexity =
      B * ms_ceiling(N, 2.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  // Regression for NNPA with r2 = 0.929669464549952
  return 4.083643957216863e-12 * complexity + 2.9749801639764202e-06;
}

// Operation Max_3ds: estimated times with cross over at complexity = 28928.
double estimatedTimeForCPU_Max_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.999833820537654
  return 1.2859513298352582e-10 * complexity + 2.1445698817257752e-07;
}
double estimatedTimeForNNPA_Max_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9951863938046909
  return 1.0420243623790107e-10 * complexity + 9.20104282137138e-07;
}

// Operation Min_3ds: estimated times with cross over at complexity = 99463.
double estimatedTimeForCPU_Min_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998504038595571
  return 1.28567886153589e-10 * complexity + 2.0245445263399963e-07;
}
double estimatedTimeForNNPA_Min_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9946372532381683
  return 1.0339073648992919e-10 * complexity + 2.70667430925335e-06;
}

// Operation Mul_3ds: estimated times with cross over at complexity = 55288.
double estimatedTimeForCPU_Mul_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998538384140817
  return 1.2961833783614705e-10 * complexity + 2.2021597368460279e-07;
}
double estimatedTimeForNNPA_Mul_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9944216212877205
  return 1.0281908200330892e-10 * complexity + 1.7019104732413162e-06;
}

// Operation Pow_2_3ds: estimated times with cross over at complexity = 56111.
double estimatedTimeForCPU_Pow_2_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998705421933038
  return 1.2964568661871788e-10 * complexity + 2.0823276086876973e-07;
}
double estimatedTimeForNNPA_Pow_2_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9945018708254405
  return 1.0317090324802998e-10 * complexity + 1.693775697889311e-06;
}

// Operation Pow_3_3ds: estimated times with cross over at complexity = 21706.
double estimatedTimeForCPU_Pow_3_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9997984857078572
  return 3.7494455439876194e-10 * complexity + -8.306238888444397e-07;
}
double estimatedTimeForNNPA_Pow_3_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9411544973356172
  return 2.085058530670984e-10 * complexity + 2.7822286559924233e-06;
}

// Operation Pow_4_3ds: estimated times with cross over at complexity = 41065.
double estimatedTimeForCPU_Pow_4_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998522920538492
  return 2.5405902189076053e-10 * complexity + 1.143001005006861e-07;
}
double estimatedTimeForNNPA_Pow_4_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9978757256268103
  return 1.811269802457481e-10 * complexity + 3.109286345912229e-06;
}

// Operation Pow_8_3ds: estimated times with cross over at complexity = 38894.
double estimatedTimeForCPU_Pow_8_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998492647347293
  return 3.784270893083021e-10 * complexity + 3.846063745069773e-08;
}
double estimatedTimeForNNPA_Pow_8_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9984689810839804
  return 2.5809839682053116e-10 * complexity + 4.718642808461304e-06;
}

// Operation Relu_3ds: estimated times with cross over at complexity = 12461.
double estimatedTimeForCPU_Relu_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999336782083352
  return 1.287932373847373e-10 * complexity + 2.1978022850964433e-07;
}
double estimatedTimeForNNPA_Relu_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9932963892258988
  return 8.174802287383875e-11 * complexity + 8.060373976318752e-07;
}

// Operation Sigmoid_3ds: estimated times with cross over at complexity = 605.
double estimatedTimeForCPU_Sigmoid_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999988107319201
  return 5.386219204160899e-09 * complexity + 5.634215435092746e-07;
}
double estimatedTimeForNNPA_Sigmoid_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9723030037974931
  return 1.0653822523387642e-10 * complexity + 3.7620795110656856e-06;
}

// Operation Softmax_3ds: estimated times with cross over at complexity = 3256.
double estimatedTimeForCPU_Softmax_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9997073553561724
  return 6.319129534911603e-09 * complexity + 2.8581895675517657e-06;
}
double estimatedTimeForNNPA_Softmax_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.7561317069552476
  return 1.5370950144430126e-09 * complexity + 1.8432846457914885e-05;
}

// Operation Sub_3ds: estimated times with cross over at complexity = 27788.
double estimatedTimeForCPU_Sub_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998718450512123
  return 1.2958900359490862e-10 * complexity + 2.0747306461438247e-07;
}
double estimatedTimeForNNPA_Sub_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9949061007697938
  return 1.0456801418978144e-10 * complexity + 9.027743246689109e-07;
}

// Operation Tanh_3ds: estimated times with cross over at complexity = 474.
double estimatedTimeForCPU_Tanh_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.999998963010189
  return 5.609614699943962e-09 * complexity + 1.8748266930146244e-07;
}
double estimatedTimeForNNPA_Tanh_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9897555127053886
  return 8.058830271076489e-11 * complexity + 2.8126039207664257e-06;
}

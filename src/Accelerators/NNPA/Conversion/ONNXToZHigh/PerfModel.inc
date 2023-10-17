/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Auto-Generated, do not change  ---------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================

inline static double ms_ceiling(double n, double m) { return ceil(n / m) * m; }

// Operation Add_3ds: estimated times.
double estimatedTimeForCPU_Add_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9989711028792525
  return 3.9686846353007493e-07 + 1.1794164898251022e-10 * complexity;
}
// Operation Add_3ds: estimated times.
double estimatedTimeForNNPA_Add_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9994266956239162
  return 2.128070603450555e-06 + 3.884079345448728e-11 * complexity +
         3.8840793454487276e-11 * complexity2;
}

// Operation Div_3ds: estimated times.
double estimatedTimeForCPU_Div_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999973603706902
  return 6.024413187183776e-07 + 1.444210277092263e-09 * complexity;
}
// Operation Div_3ds: estimated times.
double estimatedTimeForNNPA_Div_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9991809091918582
  return 4.268037890056841e-06 + 3.958236721770986e-11 * complexity +
         3.9582367217709856e-11 * complexity2;
}

// Operation Exp_3ds: estimated times.
double estimatedTimeForCPU_Exp_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9964027710112378
  return -5.114482496042976e-06 + 4.191612771812482e-09 * complexity;
}
// Operation Exp_3ds: estimated times.
double estimatedTimeForNNPA_Exp_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9996489747804022
  return 2.940912571153368e-06 + 3.030831435560512e-11 * complexity +
         3.030831435560511e-11 * complexity2;
}

// Operation Log_3ds: estimated times.
double estimatedTimeForCPU_Log_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.98951908796714
  return -1.1673535780041665e-05 + 5.568744038404678e-09 * complexity;
}
// Operation Log_3ds: estimated times.
double estimatedTimeForNNPA_Log_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9994743517297515
  return 1.9298869463234537e-06 + 3.5198842979463965e-11 * complexity +
         3.519884297946396e-11 * complexity2;
}

// Operation MatMul_3ds: estimated times.
double estimatedTimeForCPU_MatMul_3ds(double B, double N, double M, double K) {
  double complexity = B * (N * M * K);
  // Regression for CPU with r2 = 0.9993063132437037
  return 1.2274778896376187e-06 + 8.277833300031912e-11 * complexity;
}
// Operation MatMul_3ds: estimated times.
double estimatedTimeForNNPA_MatMul_3ds(double B, double N, double M, double K) {
  double complexity =
      B * ms_ceiling(N, 2.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  double complexity2 =
      B * ms_ceiling(N, 32.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  // Regression for NNPA with r2 = 0.7300356886725241
  return 3.2376205400192875e-06 + 1.2476855786580124e-12 * complexity +
         1.2476855786580124e-12 * complexity2;
}

// Operation Max_3ds: estimated times.
double estimatedTimeForCPU_Max_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.999934744283212
  return 3.2005877071463053e-07 + 1.1769966151617187e-10 * complexity;
}
// Operation Max_3ds: estimated times.
double estimatedTimeForNNPA_Max_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.999738981118584
  return 1.4606129913618409e-06 + 3.931840035694782e-11 * complexity +
         3.93184003569478e-11 * complexity2;
}

// Operation Min_3ds: estimated times.
double estimatedTimeForCPU_Min_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999348493035218
  return 3.5344354604038237e-07 + 1.175917606999914e-10 * complexity;
}
// Operation Min_3ds: estimated times.
double estimatedTimeForNNPA_Min_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9994174686041679
  return 3.092302220284167e-06 + 3.897841215495711e-11 * complexity +
         3.8978412154957106e-11 * complexity2;
}

// Operation Mul_3ds: estimated times.
double estimatedTimeForCPU_Mul_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999212032519668
  return 3.506815960365203e-07 + 1.1755692215565626e-10 * complexity;
}
// Operation Mul_3ds: estimated times.
double estimatedTimeForNNPA_Mul_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9994566182095322
  return 2.0629649376500924e-06 + 3.899817129736847e-11 * complexity +
         3.8998171297368456e-11 * complexity2;
}

// Operation Pow2_3ds: estimated times.
double estimatedTimeForCPU_Pow2_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999320368494156
  return 4.0421584022966975e-07 + 1.1715722909330777e-10 * complexity;
}
// Operation Pow2_3ds: estimated times.
double estimatedTimeForNNPA_Pow2_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9994873079624843
  return 2.0966320666458707e-06 + 3.885726395091773e-11 * complexity +
         3.8857263950917726e-11 * complexity2;
}

// Operation Pow3_3ds: estimated times.
double estimatedTimeForCPU_Pow3_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998245341999067
  return 1.643157439725954e-06 + 2.5576590285464804e-10 * complexity;
}
// Operation Pow3_3ds: estimated times.
double estimatedTimeForNNPA_Pow3_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.3286847571457816
  return 5.54645111424955e-06 + 1.594942039674421e-10 * complexity +
         1.5949420396744207e-10 * complexity2;
}

// Operation Pow4_3ds: estimated times.
double estimatedTimeForCPU_Pow4_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999197459422748
  return 9.47474072842895e-07 + 2.4163095833040496e-10 * complexity;
}
// Operation Pow4_3ds: estimated times.
double estimatedTimeForNNPA_Pow4_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9994611814325557
  return 3.901310093115667e-06 + 7.801645620198453e-11 * complexity +
         7.801645620198452e-11 * complexity2;
}

// Operation Pow8_3ds: estimated times.
double estimatedTimeForCPU_Pow8_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9997872605697207
  return 8.784826358183354e-07 + 3.649713012323953e-10 * complexity;
}
// Operation Pow8_3ds: estimated times.
double estimatedTimeForNNPA_Pow8_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9995041673834962
  return 6.134676383478624e-06 + 1.1626159876565245e-10 * complexity +
         1.1626159876565244e-10 * complexity2;
}

// Operation ReduceMean_4d: estimated times.
double estimatedTimeForCPU_ReduceMean_4d(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9581972945355149
  return -1.561907564731308e-07 + 1.2218609072525066e-10 * complexity;
}
// Operation ReduceMean_4d: estimated times.
double estimatedTimeForNNPA_ReduceMean_4d(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.21677591777344662
  return 1.1936219405338953e-05 + 1.0676642952683933e-11 * complexity +
         1.0676642952683933e-11 * complexity2;
}

// Operation Relu_3ds: estimated times.
double estimatedTimeForCPU_Relu_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9997916471943519
  return 4.020992561015175e-07 + 1.1775068214689546e-10 * complexity;
}
// Operation Relu_3ds: estimated times.
double estimatedTimeForNNPA_Relu_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9996967196634997
  return 1.2244816818061312e-06 + 2.8344213155279377e-11 * complexity +
         2.8344213155279377e-11 * complexity2;
}

// Operation Sigmoid_3ds: estimated times.
double estimatedTimeForCPU_Sigmoid_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999935862933553
  return 2.4666188614796535e-07 + 5.3819773454779955e-09 * complexity;
}
// Operation Sigmoid_3ds: estimated times.
double estimatedTimeForNNPA_Sigmoid_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9997032611893206
  return 4.552265280283248e-06 + 4.268025052443249e-11 * complexity +
         4.2680250524432486e-11 * complexity2;
}

// Operation Softmax_3ds: estimated times.
double estimatedTimeForCPU_Softmax_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998169150056859
  return 3.850778825086575e-06 + 6.476546494036936e-09 * complexity;
}
// Operation Softmax_3ds: estimated times.
double estimatedTimeForNNPA_Softmax_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.6458689171873927
  return 3.823709210789688e-05 + 7.316577699975697e-10 * complexity +
         7.316577699975696e-10 * complexity2;
}

// Operation Stick_3ds: estimated times.
double estimatedTimeForNNPA_Stick_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9838446914891756
  return -1.1787349678206611e-07 + 9.738975985428137e-11 * complexity +
         9.738975985428137e-11 * complexity2;
}

// Operation Sub_3ds: estimated times.
double estimatedTimeForCPU_Sub_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9989967088496832
  return 4.6884880875538195e-07 + 1.178543625471088e-10 * complexity;
}
// Operation Sub_3ds: estimated times.
double estimatedTimeForNNPA_Sub_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9997163549514938
  return 1.3893829024566132e-06 + 3.958841159751218e-11 * complexity +
         3.95884115975122e-11 * complexity2;
}

// Operation Tanh_3ds: estimated times.
double estimatedTimeForCPU_Tanh_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9899146645413962
  return 4.591865418171123e-06 + 1.5243041278914726e-09 * complexity;
}
// Operation Tanh_3ds: estimated times.
double estimatedTimeForNNPA_Tanh_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9992996189301544
  return 3.1652117218733632e-06 + 2.7515670717117405e-11 * complexity +
         2.7515670717117402e-11 * complexity2;
}

// Operation Unstick_3ds: estimated times.
double estimatedTimeForNNPA_Unstick_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9804634203145826
  return -3.9924342483643434e-07 + 1.0477727134169295e-10 * complexity +
         1.0477727134169292e-10 * complexity2;
}
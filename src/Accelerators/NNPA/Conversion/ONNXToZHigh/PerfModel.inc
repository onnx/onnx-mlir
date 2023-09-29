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
  // Regression for CPU with r2 = 0.9998796798458582
  return 1.3014297439117151e-10 * complexity + 2.0196878739035375e-07;
}
double estimatedTimeForNNPA_Add_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9948030510499283
  return 5.325626680157495e-12 * complexity +
         9.784397460528751e-11 * complexity2 + 1.8134670050569665e-06;
}

// Operation Div_3ds: estimated times.
double estimatedTimeForCPU_Div_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999989217222092
  return 1.4517483410281062e-09 * complexity + 4.819629870926124e-07;
}
double estimatedTimeForNNPA_Div_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9943006915048658
  return 7.903245032273375e-12 * complexity +
         9.731375885167297e-11 * complexity2 + 3.843102854870541e-06;
}

// Operation Exp_3ds: estimated times.
double estimatedTimeForCPU_Exp_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9953451710472131
  return 3.728416286592805e-09 * complexity + 6.0588917718826046e-06;
}
double estimatedTimeForNNPA_Exp_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9926325392367861
  return 2.0688708686506677e-11 * complexity +
         6.618128966933956e-11 * complexity2 + 2.3175395047747865e-06;
}

// Operation Log_3ds: estimated times.
double estimatedTimeForCPU_Log_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999978790893227
  return 8.437935291413468e-09 * complexity + -5.198279741842633e-07;
}
double estimatedTimeForNNPA_Log_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9944656749382057
  return 3.563276809776611e-11 * complexity +
         6.11597794115873e-11 * complexity2 + 1.3433454010795865e-06;
}

// Operation MatMul_3ds: estimated times.
double estimatedTimeForCPU_MatMul_3ds(double B, double N, double M, double K) {
  double complexity = B * (N * M * K);
  // Regression for CPU with r2 = 0.9977563719224963
  return 8.516416082324592e-11 * complexity + 2.813872063426722e-07;
}
double estimatedTimeForNNPA_MatMul_3ds(double B, double N, double M, double K) {
  double complexity =
      B * ms_ceiling(N, 2.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  double complexity2 =
      B * ms_ceiling(N, 32.0) * ms_ceiling(M, 64.0) * ms_ceiling(K, 64.0);
  // Regression for NNPA with r2 = 0.9732532845259424
  return 1.4939220470182802e-12 * complexity +
         3.3230442404022774e-12 * complexity2 + 2.6050726452004855e-06;
}

// Operation Max_3ds: estimated times.
double estimatedTimeForCPU_Max_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.999833820537654
  return 1.2859513298352582e-10 * complexity + 2.1445698817257752e-07;
}
double estimatedTimeForNNPA_Max_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9954905947084662
  return 7.494472351978157e-12 * complexity +
         9.740204632144935e-11 * complexity2 + 9.347515392991801e-07;
}

// Operation Min_3ds: estimated times.
double estimatedTimeForCPU_Min_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998504038595571
  return 1.28567886153589e-10 * complexity + 2.0245445263399963e-07;
}
double estimatedTimeForNNPA_Min_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9947548027057314
  return 4.623754375472404e-12 * complexity +
         9.919519994587717e-11 * complexity2 + 2.7157110120672263e-06;
}

// Operation Mul_3ds: estimated times.
double estimatedTimeForCPU_Mul_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998538384140817
  return 1.2961833783614705e-10 * complexity + 2.2021597368460279e-07;
}
double estimatedTimeForNNPA_Mul_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9946309378034859
  return 6.136565090368117e-12 * complexity +
         9.725084005080186e-11 * complexity2 + 1.713903825508006e-06;
}

// Operation Pow_2_3ds: estimated times.
double estimatedTimeForCPU_Pow_2_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998705421933038
  return 1.2964568661871788e-10 * complexity + 2.0823276086876973e-07;
}
double estimatedTimeForNNPA_Pow_2_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9947176714834728
  return 6.251956774013825e-12 * complexity +
         9.749795633364532e-11 * complexity2 + 1.7059945725924484e-06;
}

// Operation Pow_3_3ds: estimated times.
double estimatedTimeForCPU_Pow_3_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9997984857078572
  return 3.7494455439876194e-10 * complexity + -8.306238888444397e-07;
}
double estimatedTimeForNNPA_Pow_3_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9744521673481049
  return -1.6133547280863913e-10 * complexity +
         3.54899631164666e-10 * complexity2 + 2.4669133077458067e-06;
}

// Operation Pow_4_3ds: estimated times.
double estimatedTimeForCPU_Pow_4_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998522920538492
  return 2.5405902189076053e-10 * complexity + 1.143001005006861e-07;
}
double estimatedTimeForNNPA_Pow_4_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9980332599676777
  return 9.361979621754725e-12 * complexity +
         1.7263203772974787e-10 * complexity2 + 3.1275834743324167e-06;
}

// Operation Pow_8_3ds: estimated times.
double estimatedTimeForCPU_Pow_8_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998492647347293
  return 3.784270893083021e-10 * complexity + 3.846063745069773e-08;
}
double estimatedTimeForNNPA_Pow_8_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9987430714971388
  return 1.7591387480214942e-11 * complexity +
         2.421361931564836e-10 * complexity2 + 4.753023557920182e-06;
}

// Operation Relu_3ds: estimated times.
double estimatedTimeForCPU_Relu_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999336782083352
  return 1.287932373847373e-10 * complexity + 2.1978022850964433e-07;
}
double estimatedTimeForNNPA_Relu_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9938000626848831
  return 7.572643578806345e-12 * complexity +
         7.487670136971643e-11 * complexity2 + 8.208374332696093e-07;
}

// Operation Sigmoid_3ds: estimated times.
double estimatedTimeForCPU_Sigmoid_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9999988107319201
  return 5.386219204160899e-09 * complexity + 5.634215435092746e-07;
}
double estimatedTimeForNNPA_Sigmoid_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9951347021911614
  return 6.715972471178672e-11 * complexity +
         4.559833603658543e-11 * complexity2 + 3.893337018277136e-06;
}

// Operation Softmax_3ds: estimated times.
double estimatedTimeForCPU_Softmax_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9997073553561724
  return 6.319129534911603e-09 * complexity + 2.8581895675517657e-06;
}
double estimatedTimeForNNPA_Softmax_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.7585429756849709
  return 3.570754152382795e-10 * complexity +
         1.2130892753921132e-09 * complexity2 + 1.9130717527098137e-05;
}

// Operation Sub_3ds: estimated times.
double estimatedTimeForCPU_Sub_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.9998718450512123
  return 1.2958900359490862e-10 * complexity + 2.0747306461438247e-07;
}
double estimatedTimeForNNPA_Sub_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9951200892937675
  return 6.3086757880993785e-12 * complexity +
         9.884360115565631e-11 * complexity2 + 9.151040514702722e-07;
}

// Operation Tanh_3ds: estimated times.
double estimatedTimeForCPU_Tanh_3ds(double e3, double e2, double e1) {
  double complexity = e3 * e2 * e1;
  // Regression for CPU with r2 = 0.999998963010189
  return 5.609614699943962e-09 * complexity + 1.8748266930146244e-07;
}
double estimatedTimeForNNPA_Tanh_3ds(double e3, double e2, double e1) {
  double complexity = e3 * ms_ceiling(e2, 2.0) * ms_ceiling(e1, 64.0);
  double complexity2 = e3 * ms_ceiling(e2, 32.0) * ms_ceiling(e1, 64.0);
  // Regression for NNPA with r2 = 0.9910103444430327
  return 1.1804183318811661e-11 * complexity +
         6.987733528795407e-11 * complexity2 + 2.835674111773654e-06;
}

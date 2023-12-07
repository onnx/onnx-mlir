/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ OMThreadAffinity.c --------------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ implementation of the OMThreadAffinity functions.
//
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "onnx-mlir/Runtime/OMTensor.h"

// todo: parse /proc/cpuinfo at runtime
static const int thread_affinity[] = {
    0, 1, 2, 3, 4, 5, 6, 7}; // id-to-zaiu mapping

static const int zaiu_cpuid_from[] = {
    0, 7, 14, 21, 27, 33, 39, 45}; // zaiu-to-cpuid mapping(from)
static const int zaiu_cpuid_to[] = {
    6, 13, 20, 26, 32, 38, 44, 50}; // zaiu-to-cpuid mapping(to)

void threadAffine(int64_t id) {
  // std::cout << "XXX threadAffine #" << id << "\n" << std::flush;
  // printf("XXX threadAffine start #%d\n", id);
  // fflush(stdout);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  const int zaiu_id = thread_affinity[id & 7];
  for (int i = zaiu_cpuid_from[zaiu_id]; i <= zaiu_cpuid_to[zaiu_id]; i++)
    CPU_SET(i, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  /*
  int cpu_base = (8 * id) % 16;
  for (int i = cpu_base; i < (cpu_base + 8); i++)
    CPU_SET(i, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  */
  // sleep(1);
  // printf("XXX threadAffine finish #%d\n", id);
  // fflush(stdout);
}

void dummyFuncForKeepParam(OMTensor *A, OMTensor *B) {
  // printf("XXXX dummYFuncForKeepParamt(%p, %p): called\n", A, B);
  // fflush(stdout);
}

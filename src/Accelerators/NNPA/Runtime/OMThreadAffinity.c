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

#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#undef TEST

#define MAX_CPU_NUMBER 1024
#define MAX_LINE_LENGTH 1024
#define LSCPU_COMMAND "lscpu -e"

static int OMThreadAffnity_cpu_number = -1;
static int OMThreadAffnity_cpu_to_socket_table[MAX_CPU_NUMBER];

void threadAffine(int64_t id) {
  // if OMThreadAffnity_cpu_number is not initialized, initialize it.
  if (OMThreadAffnity_cpu_number < 0) {
    FILE *fp = popen(LSCPU_COMMAND, "r");
    assert(fp != NULL && "cannot execute lscpu command");
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), fp) != NULL) {
      if (OMThreadAffnity_cpu_number < 0) { // skip the first line
        OMThreadAffnity_cpu_number = 0;
        continue;
      }
      int cpu, node, drawer, book, socket;
      sscanf(line, "%d %d %d %d %d", &cpu, &node, &drawer, &book, &socket);
      OMThreadAffnity_cpu_to_socket_table[cpu] = socket;
      OMThreadAffnity_cpu_number++;
    }
    assert(
        OMThreadAffnity_cpu_number < MAX_CPU_NUMBER && "too large cpu number");
  }
  // prepare cpuset for the specified id
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int i = 0; i < OMThreadAffnity_cpu_number; i++) {
    if (OMThreadAffnity_cpu_to_socket_table[i] == id) {
      CPU_SET(i, &cpuset);
#ifdef TEST
      printf("XXX CPU_SET threadId=%ld, cpuId=%d\n", id, i);
#endif
    }
  }
  // set thread affnity
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

#ifndef TEST
#include "onnx-mlir/Runtime/OMTensor.h"
void dummyFuncForKeepParam(OMTensor *A, OMTensor *B) {}
#endif

#ifdef TEST
#define NUM_ZAIU 8
int main() {
  // system("lscpu -e");
  for (int64_t id = 0; id < NUM_ZAIU; id++) {
    threadAffine(id);
  }
  return 0;
}
#endif

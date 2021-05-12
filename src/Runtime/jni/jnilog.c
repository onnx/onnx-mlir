/*
 * SPDX-License-Identifier: Apache-2.0
 */

/*===----------- jnilog.c - JNI wrapper simple logging routines -----------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementation of simple logging routines used by the JNI
// wrapper.
//
//===----------------------------------------------------------------------===*/

#include <errno.h>
#include <libgen.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "jnilog.h"

static __thread int log_inited = 0;
static __thread int log_level;
static __thread FILE *log_fp;

/* Must match enum in log.h */
static char *log_level_name[] = {
    "trace", "debug", "info", "warning", "error", "fatal"};

/* Generic log routine */
void log_printf(
    int level, char *file, const char *func, int line, char *fmt, ...) {

  if (level < log_level)
    return;

  time_t now;
  struct tm *tm;
  char buf[LOG_MAX_LEN];

  /* Get local time and format as 2020-07-03 05:17:42 -0400 */
  if (time(&now) == -1 || (tm = localtime(&now)) == NULL ||
      strftime(buf, sizeof(buf), "[%F %T %z]", tm) == 0)
    sprintf(buf, "[-]");

  /* Output thread ID, log level, file name, function number, and line number */
  snprintf(buf + strlen(buf), LOG_MAX_LEN - strlen(buf), "[%lx][%s]%s:%s:%d ",
      pthread_self(), log_level_name[level], basename(file), func, line);

  /* Output actual log data */
  va_list va_list;
  va_start(va_list, fmt);
  vsnprintf(buf + strlen(buf), LOG_MAX_LEN - strlen(buf), fmt, va_list);
  va_end(va_list);

  /* Add new line */
  snprintf(buf + strlen(buf), LOG_MAX_LEN - strlen(buf), "\n");

  /* Write out and flush the output buffer */
  fputs(buf, log_fp);
  fflush(log_fp);
}

/* Return numerical log level of give level name */
static int get_log_level_by_name(char *name) {
  int level = -1;
  for (int i = 0; i < sizeof(log_level_name) / sizeof(char *); i++) {
    if (!strcmp(name, log_level_name[i])) {
      level = i;
      break;
    }
  }
  return level;
}

/* Return FILE pointer of given file name */
static FILE *get_log_file_by_name(char *name) {
  FILE *fp = NULL;
  if (!strcmp(name, "stdout"))
    fp = stdout;
  else if (!strcmp(name, "stderr"))
    fp = stderr;
  else {
    char tname[strlen(name) + 32];
    snprintf(tname, strlen(name) + 32, "%s.%lx", name, pthread_self());
    fp = fopen(tname, "w");
  }
  return fp;
}

/* Initialize log system. Set log level and file from environment
 * variables ONNX_MLIR_JNI_LOG_LEVEL and ONNX_MLIR_JNI_LOG_FILE,
 * respectively if provided.
 *
 * When logging to stdout or stderr, output from multiple threads
 * will be interleaved. When logging to a file, output from multiple
 * threads go to separate files, suffixed with the thread ID.
 */
void log_init() {

  if (log_inited)
    return;

  log_level = LOG_INFO;
  char *strlevel = getenv("ONNX_MLIR_JNI_LOG_LEVEL");
  int level;
  if (strlevel && (level = get_log_level_by_name(strlevel)) != -1)
    log_level = level;

  log_fp = stderr;
  char *strfname = getenv("ONNX_MLIR_JNI_LOG_FILE");
  FILE *fp;
  if (strfname && (fp = get_log_file_by_name(strfname)))
    log_fp = fp;

  tzset();
  log_inited = 1;
}

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

#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "jnilog.h"

#define INLINE static __inline

#ifdef USE_PTHREAD_SPECIFIC
static pthread_once_t key_once = PTHREAD_ONCE_INIT;
static pthread_key_t log_inited;
static pthread_key_t log_level;
static pthread_key_t log_fp;

#define THREAD_LOCAL_INIT(key, func) pthread_once((key), (func))

INLINE void key_init() {
  pthread_key_create(&log_inited, NULL);
  pthread_key_create(&log_level, NULL);
  pthread_key_create(&log_fp, NULL);
}

INLINE int get_log_inited() {
#ifdef __MVS__
  void *inited;
  pthread_getspecific(log_inited, &inited);
  /* cast to long first to avoid compiler warning */
  return (int)(long)inited;
#else
  return (int)(long)pthread_getspecific(log_inited);
#endif
}

INLINE void set_log_inited(int inited) {
  pthread_setspecific(log_inited, (void *)(long)inited);
}

INLINE int get_log_level() {
#ifdef __MVS__
  void *level;
  pthread_getspecific(log_level, &level);
  return (int)(long)level;
#else
  return (int)(long)pthread_getspecific(log_level);
#endif
}

INLINE void set_log_level(int level) {
  pthread_setspecific(log_level, (void *)(long)level);
}

INLINE FILE *get_log_fp() {
#ifdef __MVS__
  void *fp;
  pthread_getspecific(log_fp, &fp);
  return (FILE *)fp;
#else
  return (FILE *)pthread_getspecific(log_fp);
#endif
}

INLINE void set_log_fp(FILE *fp) { pthread_setspecific(log_fp, (void *)fp); }

#else

#define THREAD_LOCAL_INIT(key, func)

#if defined(_MSC_VER)
#define THREAD_LOCAL_SPEC __declspec(thread)
#else
#define THREAD_LOCAL_SPEC __thread
#endif

static THREAD_LOCAL_SPEC int log_inited = 0;
static THREAD_LOCAL_SPEC int log_level;
static THREAD_LOCAL_SPEC FILE *log_fp;

INLINE int get_log_inited() { return log_inited; }

INLINE void set_log_inited(int inited) { log_inited = inited; }

INLINE int get_log_level() { return log_level; }

INLINE void set_log_level(int level) { log_level = level; }

INLINE FILE *get_log_fp() { return log_fp; }

INLINE void set_log_fp(FILE *fp) { log_fp = fp; }
#endif

/* Must match enum in log.h */
static const char *log_level_name[] = {
    "trace", "debug", "info", "warning", "error", "fatal"};

/* On z/OS, pthread_t is a struct that cannot be casted into unsigned long
 * so we must return pthread_t.
 */
#if defined(_MSC_VER)
typedef DWORD pthread_t;
#define THREAD_ID GetCurrentThreadId()
#else
#define THREAD_ID pthread_self()
#endif

pthread_t get_threadid() { return THREAD_ID; }

/* This is based on basename from lldb: lldb\source\Host\windows\Windows.cpp */
const char *get_filename(const char *path) {
#if defined(_MSC_VER)
  char *l1 = strrchr(path, '\\');
  char *l2 = strrchr(path, '/');
  if (l2 > l1)
    l1 = l2;
#else
  /* C library basename modifies path (except Gnu version) so we borrow
   * Windows code to implement our own basename.
   */
  char *l1 = strrchr(path, '/');
#endif
  if (!l1)
    return path; // no base name
  return &l1[1];
}

/* Generic log routine */
void log_printf(int level, const char *file, const char *func, int line,
    const char *fmt, ...) {

  if (level < get_log_level())
    return;

  time_t now;
  struct tm *tm;
  char buf[LOG_MAX_LEN];
  int num_chars_written = 0;

  /* Get local time and format as 2020-07-03 05:17:42 -0400 */
  if (time(&now) == -1 || (tm = localtime(&now)) == NULL ||
      strftime(buf, sizeof(buf), "[%F %T %z]", tm) == 0) {
    num_chars_written = sprintf(buf, "[-]");
    assert(num_chars_written >= 0 && "sprintf write error to buf");
  }

  /* Output thread ID, log level, file name, function number, and line number.
   * Note that pthread_t on most platforms is unsigned long but is a struct
   * of 8 bytes on z/OS.
   */
  pthread_t tid = get_threadid();
  assert(LOG_MAX_LEN >= strlen(buf) && "error in snprintf length");
  num_chars_written = snprintf(buf + strlen(buf), LOG_MAX_LEN - strlen(buf),
      "[%016lx][%s]%s:%s:%d ", *(unsigned long *)&tid, log_level_name[level],
      get_filename(file), func, line);
  assert(num_chars_written >= 0 && "snprintf write error to buf");

  /* Output actual log data */
  /* Definition of vsnprintf:
     int vsnprintf (char * s, size_t n, const char * format, va_list arg);

     Write formatted data from variable argument list to sized buffer
     Composes a string with the same text that would be printed if format was
     used on printf, but using the elements in the variable argument list
     identified by arg instead of additional function arguments and storing the
     resulting content as a C string in the buffer pointed by s (taking n as the
     maximum buffer capacity to fill).

    Analysis:
    As seen in the code, the buffer is statically declared with a size of
    `LOG_MAX_LEN`:

    char buf[LOG_MAX_LEN];

    And the vsnprintf will uses at most `LOG_MAX_LEN - strlen(buf)` bytes from
    the buffer which starts at `buff + strlen(buf)`.

    As an added security, we added an assert to make sure that quantity is
    positive.
  */
  assert(LOG_MAX_LEN >= strlen(buf) && "error in vsnprintf length");

  va_list log_data;
  va_start(log_data, fmt);
  num_chars_written =
      vsnprintf(buf + strlen(buf), LOG_MAX_LEN - strlen(buf), fmt, log_data);
  assert(num_chars_written >= 0 && "vsnprintf write error to buf");
  va_end(log_data);

  /* Add new line */
  num_chars_written =
      snprintf(buf + strlen(buf), LOG_MAX_LEN - strlen(buf), "\n");
  assert(num_chars_written >= 0 && "snprintf write error to buf");

  /* Write out and flush the output buffer */
  FILE *fp = get_log_fp();
  fputs(buf, fp);
  fflush(fp);
}

/* Return numerical log level of give level name */
static int get_log_level_by_name(char *name) {
  int level = -1;
  for (int i = 0; i < (int)(sizeof(log_level_name) / sizeof(char *)); i++) {
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
    char *tname = (char *)malloc(strlen(name) + 32);
    if (tname) {
      pthread_t tid = get_threadid();
      int num_chars_written = snprintf(
          tname, strlen(name) + 32, "%s.%016lx", name, *(unsigned long *)&tid);
      assert(num_chars_written >= 0 && "snprintf write error to tname");
      fp = fopen(tname, "w");
      assert(fp != NULL && "fopen error on tname");
      free(tname);
    }
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

  THREAD_LOCAL_INIT(&key_once, key_init);

  if (get_log_inited())
    return;

  set_log_level(LOG_INFO);
  char *str_level = getenv("ONNX_MLIR_JNI_LOG_LEVEL");
  int level;
  if (str_level && (level = get_log_level_by_name(str_level)) != -1)
    set_log_level(level);

  set_log_fp(stderr);
  char *str_fname = getenv("ONNX_MLIR_JNI_LOG_FILE");
  FILE *fp;
  if (str_fname && (fp = get_log_file_by_name(str_fname)))
    set_log_fp(fp);

  tzset();
  set_log_inited(1);
}

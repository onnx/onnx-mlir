/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- jnilog.h - JNI wrapper simple logging header- -----------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains macro definitions of simple logging routines used by the
// JNI wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef __JNILOG_H__
#define __JNILOG_H__

#include <stdio.h>

enum { LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_FATAL };

#define LOG_MAX_LEN 4096 /* max number of chars to output    */
#define LOG_MAX_NUM 128  /* max number of elements to output */

#define MIN(x, y) ((x) > (y) ? y : x)

/* Construct string of up to LOG_MAX_NUM elements of an array of C type */
#define LOG_BUF_C_TYPE(type, format, buf, data, n)                             \
  do {                                                                         \
    char *p = buf;                                                             \
    /* Reserve 5 char at the end for " ... \0". Note the first                 \
     * space will come from the '\0' of the previous string.                   \
     */                                                                        \
    int i = 0, j = sizeof(buf) - 5, k, l = MIN(n, LOG_MAX_NUM);                \
    /* j is the available number of chars including '\0'. k is the             \
     * number of chars printed without '\0'. So as long as k < j,              \
     * it means the output, with a trailing '\0', fits in the buffer.          \
     */                                                                        \
    while (i < l && (k = snprintf(p, j, format, ((type *)data)[i])) < j) {     \
      p += k;                                                                  \
      j -= k;                                                                  \
      i++;                                                                     \
    }                                                                          \
    /* If i==l, we finished all the elements, add " " at the end.              \
     * Otherwise, we ran out of buffer. If j==1, it means output               \
     * so far fits exactly in the buffer, and we are at the last               \
     * char of the buffer (which is '\0') available for output                 \
     * (you can't really output anything since you always need                 \
     * the '\0' at the end). So we add " ... " at the end to                   \
     * denote that the last element is complete. Otherwise, we                 \
     * add "... " at the end to denote that the last element is                \
     * truncated.                                                              \
     */                                                                        \
    snprintf(                                                                  \
        buf + strlen(buf), 6, (i == l) ? " " : (j == 1) ? " ... " : "... ");   \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of an array of ONNX type.
 * The type is specified at runtime so we have to use switch.
 */
#define LOG_BUF_ONNX_TYPE(type, buf, data, n, hex)                             \
  do {                                                                         \
    switch (type) {                                                            \
    case ONNX_TYPE_UINT8:                                                      \
    case ONNX_TYPE_INT8:                                                       \
      LOG_BUF_C_TYPE(char, hex ? " %02x" : "%c", buf, data, n);                \
      break;                                                                   \
    case ONNX_TYPE_UINT16:                                                     \
    case ONNX_TYPE_INT16:                                                      \
      LOG_BUF_C_TYPE(short, hex ? " %04x" : " %d", buf, data, n);              \
      break;                                                                   \
    case ONNX_TYPE_UINT32:                                                     \
    case ONNX_TYPE_INT32:                                                      \
      LOG_BUF_C_TYPE(int, hex ? " %08x" : " %d", buf, data, n);                \
      break;                                                                   \
    case ONNX_TYPE_UINT64:                                                     \
    case ONNX_TYPE_INT64:                                                      \
      LOG_BUF_C_TYPE(long, hex ? " %016x" : " %ld", buf, data, n);             \
      break;                                                                   \
    case ONNX_TYPE_FLOAT:                                                      \
      LOG_BUF_C_TYPE(float, hex ? " %08x" : " %f", buf, data, n);              \
      break;                                                                   \
    case ONNX_TYPE_DOUBLE:                                                     \
      LOG_BUF_C_TYPE(double, hex ? " %016x" : " %lf", buf, data, n);           \
      break;                                                                   \
    defaut:                                                                    \
      sprintf(buf, " unsupported data type %d ", type);                        \
    }                                                                          \
  } while (0)

#define LOG_BUF(type, buf, data, n) LOG_BUF_ONNX_TYPE(type, buf, data, n, 0)
#define LOG_XBUF(type, buf, data, n) LOG_BUF_ONNX_TYPE(type, buf, data, n, 1)

#define LOG_CHAR_BUF(buf, data, n) LOG_BUF_C_TYPE(char, "%c", buf, data, n)
#define LOG_CHAR_XBUF(buf, data, n) LOG_BUF_C_TYPE(char, " %02x", buf, data, n)
#define LOG_SHORT_BUF(buf, data, n) LOG_BUF_C_TYPE(short, " %d", buf, data, n)
#define LOG_SHORT_XBUF(buf, data, n)                                           \
  LOG_BUF_C_TYPE(short, " %04x", buf, data, n)
#define LOG_INT_BUF(buf, data, n) LOG_BUF_C_TYPE(int, " %d", buf, data, n)
#define LOG_INT_XBUF(buf, data, n) LOG_BUF_C_TYPE(int, " %08x", buf, data, n)
#define LOG_LONG_BUF(buf, data, n) LOG_BUF_C_TYPE(long, " %ld", buf, data, n)
#define LOG_LONG_XBUF(buf, data, n) LOG_BUF_C_TYPE(long, " %016x", buf, data, n)
#define LOG_FLAT_BUF(buf, data, n) LOG_BUF_C_TYPE(float, " %f", buf, data, n)
#define LOG_FLOAT_XBUF(buf, data, n)                                           \
  LOG_BUF_C_TYPE(float, " %08x", buf, data, n)
#define LOG_DOUBLE_BUF(buf, data, n)                                           \
  LOG_BUF_C_TYPE(double, " %lf", buf, data, n)
#define LOG_DOUBLE_XBUF(buf, data, n)                                          \
  LOG_BUF_C_TYPE(double, " %016x", buf, data, n)

/* Main macro for log output */
#define LOG_PRINTF(level, ...)                                                 \
  log_printf(level, (char *)__FILE__, (char *)__FUNCTION__, __LINE__,          \
      (char *)__VA_ARGS__)

/* Generic log routine */
extern void log_init(void);
extern void log_printf(
    int level, char *file, const char *func, int line, char *fmt, ...);

#endif

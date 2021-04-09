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

#define LOG_MAX_NUM 16 /* max number of elements to output */

#define MIN(x, y) ((x) > (y) ? y : x)

/* Construct string of up to LOG_MAX_NUM elements of a char array */
#define LOG_CHAR_BUF(buf, data, n)                                             \
  do {                                                                         \
    buf[0] = '\0';                                                             \
    for (int i = 0; i < MIN(n, LOG_MAX_NUM); i++)                              \
      sprintf(buf + strlen(buf), " %02x", ((char *)data)[i]);                  \
    sprintf(buf + strlen(buf), n > LOG_MAX_NUM ? " ... " : " ");               \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of a short array */
#define LOG_SHORT_BUF(buf, data, n)                                            \
  do {                                                                         \
    buf[0] = '\0';                                                             \
    for (int i = 0; i < MIN(n, LOG_MAX_NUM); i++)                              \
      sprintf(buf + strlen(buf), " %d", ((short *)data)[i]);                   \
    sprintf(buf + strlen(buf), n > LOG_MAX_NUM ? " ... " : " ");               \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of a int array */
#define LOG_INT_BUF(buf, data, n)                                              \
  do {                                                                         \
    buf[0] = '\0';                                                             \
    for (int i = 0; i < MIN(n, LOG_MAX_NUM); i++)                              \
      sprintf(buf + strlen(buf), " %d", ((int *)data)[i]);                     \
    sprintf(buf + strlen(buf), n > LOG_MAX_NUM ? " ... " : " ");               \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of a long array */
#define LOG_LONG_BUF(buf, data, n)                                             \
  do {                                                                         \
    buf[0] = '\0';                                                             \
    for (int i = 0; i < MIN(n, LOG_MAX_NUM); i++)                              \
      sprintf(buf + strlen(buf), " %ld", ((long *)data)[i]);                   \
    sprintf(buf + strlen(buf), n > LOG_MAX_NUM ? " ... " : " ");               \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of a float array */
#define LOG_FLOAT_BUF(buf, data, n)                                            \
  do {                                                                         \
    buf[0] = '\0';                                                             \
    for (int i = 0; i < MIN(n, LOG_MAX_NUM); i++)                              \
      sprintf(buf + strlen(buf), " %f", ((float *)data)[i]);                   \
    sprintf(buf + strlen(buf), n > LOG_MAX_NUM ? " ... " : " ");               \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of a double array */
#define LOG_DOUBLE_BUF(buf, data, n)                                           \
  do {                                                                         \
    buf[0] = '\0';                                                             \
    for (int i = 0; i < MIN(n, LOG_MAX_NUM); i++)                              \
      sprintf(buf + strlen(buf), " %f", ((double *)data)[i]);                  \
    sprintf(buf + strlen(buf), n > LOG_MAX_NUM ? " ... " : " ");               \
  } while (0)

/* Construct string of up to LOG_MAX_NUM elements of a "type" array */
#define LOG_TYPE_BUF(type, buf, data, n)                                       \
  do {                                                                         \
    switch (type) {                                                            \
    case ONNX_TYPE_UINT8:                                                      \
    case ONNX_TYPE_INT8:                                                       \
      LOG_CHAR_BUF(buf, data, n);                                              \
      break;                                                                   \
    case ONNX_TYPE_UINT16:                                                     \
    case ONNX_TYPE_INT16:                                                      \
      LOG_SHORT_BUF(buf, data, n);                                             \
      break;                                                                   \
    case ONNX_TYPE_UINT32:                                                     \
    case ONNX_TYPE_INT32:                                                      \
      LOG_INT_BUF(buf, data, n);                                               \
      break;                                                                   \
    case ONNX_TYPE_UINT64:                                                     \
    case ONNX_TYPE_INT64:                                                      \
      LOG_LONG_BUF(buf, data, n);                                              \
      break;                                                                   \
    case ONNX_TYPE_FLOAT:                                                      \
      LOG_FLOAT_BUF(buf, data, n);                                             \
      break;                                                                   \
    case ONNX_TYPE_DOUBLE:                                                     \
      LOG_DOUBLE_BUF(buf, data, n);                                            \
      break;                                                                   \
    defaut:                                                                    \
      sprintf(buf, " unsupported data type %d ", type);                        \
    }                                                                          \
  } while (0)

/* Main macro for log output */
#define LOG_PRINTF(level, ...)                                                 \
  log_printf(level, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

/* Generic log routine */
void log_printf(
    int level, char *file, const char *func, int line, char *fmt, ...);

#endif

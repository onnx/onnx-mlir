#include <errno.h>
#include <libgen.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "jnilog.h"

static int log_initd = 0;
static int log_level;
static FILE *log_fp;

/* Must match enum in log.h */
static char *log_level_name[] = {
    "trace", "debug", "info", "warning", "error", "fatal"};

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
  else
    fp = fopen(name, "w");
  return fp;
}

/* Initialize log system. Set default log level and file or use environment
 * variables ONNX_MLIR_JNI_LOG_LEVEL and ONNX_MLIR_JNI_LOG_FILE, respectively.
 */
static void log_init() {
  if (log_initd)
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
  log_initd = 1;
}

/* Generic log routine */
void log_printf(
    int level, char *file, const char *func, int line, char *fmt, ...) {
  if (!log_initd)
    log_init();
  if (level < log_level)
    return;

  time_t now;
  struct tm *tm;
  char buf[80];

  /* Get local time and format as 2020-07-03 05:17:42 -0400 */
  if (time(&now) == -1 || (tm = localtime(&now)) == NULL ||
      strftime(buf, sizeof(buf), "%F %T %z", tm) == 0)
    sprintf(buf, "-");

  /* Output log prefix */
  fprintf(log_fp, "[%s][%s]%s:%s:%d ", buf, log_level_name[level],
      basename(file), func, line);

  /* Output actually log data */
  va_list va_list;
  va_start(va_list, fmt);
  vfprintf(log_fp, fmt, va_list);
  va_end(va_list);

  fprintf(log_fp, "\n");
}

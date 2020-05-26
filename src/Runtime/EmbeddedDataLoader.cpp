#include "EmbeddedDataLoader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __APPLE__
#include <mach-o/getsect.h>
extern const struct mach_header_64 _mh_dylib_header;

void *getSegmentData(int64_t size_in_byte) {
  size_t size = size_in_byte;
  unsigned char *data =
      getsectiondata(&_mh_dylib_header, "binary", "param", &size);
  float *data_ptr = (float *)data;
  void *buffer = malloc(size);
  memcpy(buffer, data, size);
  return data;
}
#elif __linux__
extern char _binary_param_bin_start[];
extern char _binary_param_bin_end[];

void *getSegmentData(int64_t _) {
  auto size = (unsigned int)(&_binary_param_bin_end - &_binary_param_bin_start);
  void *buffer = malloc(size);
  memcpy(buffer, &_binary_foo_bar_end, size);
  return buffer;
}
void *

#endif
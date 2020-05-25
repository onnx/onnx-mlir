#include "EmbeddedDataLoader.h"

#include <mach-o/getsect.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const struct mach_header_64 _mh_dylib_header;

void *getSegmentData() {
  size_t size = 4*sizeof(float);
  unsigned char *data =
      getsectiondata(&_mh_dylib_header, "binary", "param", &size);
  float* data_ptr = (float*) data;
  for (int i=0; i<4; i++) {
      printf("%f,", data_ptr[i]);
  }
  void *buffer = malloc(size);
  memcpy(buffer, data, size); // whatever
  return data;
}
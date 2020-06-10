#include "GetEmbeddedConstPool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Adapted from:
// https://developer.ibm.com/technologies/systems/articles/au-endianc/
const int i = 1;
#define IS_SYSTEM_LE() (!((*(char *)&i) == 0))

#define XOR(a, b) (!(a) != !(b))

extern const char constPackIsLE;

void checkEndianness() {
  if (XOR(IS_SYSTEM_LE(), constPackIsLE)) {
    fprintf(stderr, "Constant pack is stored in a byte order that is not "
                    "native to this current system.");
    exit(1);
  }
}

#if __APPLE__
#include <mach-o/getsect.h>
extern const struct mach_header_64 _mh_dylib_header;

void *getEmbeddedConstPool(int64_t size_in_byte) {
  checkEndianness();
  size_t size = size_in_byte;
  unsigned char *data =
      getsectiondata(&_mh_dylib_header, "binary", "param", &size);
  float *data_ptr = (float *)data;
  void *buffer = malloc(size);
  memcpy(buffer, data, size);
  return data;
}

#elif __linux__
extern char _binary_param_bin_start;
extern char _binary_param_bin_end;

void *getEmbeddedConstPool(int64_t _) {
  checkEndianness();
  auto size = (unsigned int)(&_binary_param_bin_end - &_binary_param_bin_start);
  void *buffer = malloc(size);
  memcpy(buffer, &_binary_param_bin_start, size);
  return buffer;
}

#else

extern char constPackFileName[];
extern int64_t constPackFileNameStrLen;

void *getEmbeddedConstPool(int64_t _) {
  checkEndianness();
  char *fname = (char *)calloc(1, constPackFileNameStrLen + 1);
  memcpy(fname, constPackFileName, constPackFileNameStrLen);

  // Adapted from https://stackoverflow.com/a/22059317 .
  FILE *fileptr;
  char *buffer;
  long filelen;

  fileptr = fopen(fname, "rb"); // Open the file in binary mode
  fseek(fileptr, 0, SEEK_END);  // Jump to the end of the file
  filelen = ftell(fileptr);     // Get the current byte offset in the file
  rewind(fileptr);              // Jump back to the beginning of the file

  buffer = (char *)malloc(filelen * sizeof(char)); // Enough memory for the file
  fread(buffer, filelen, 1, fileptr);              // Read in the entire file
  fclose(fileptr);                                 // Close the file

  return (void *)buffer;
}
#endif
#include "GetEmbeddedConstPool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __APPLE__
#include <mach-o/getsect.h>
extern const struct mach_header_64 _mh_dylib_header;

void *getEmbeddedConstPool(int64_t size_in_byte) {
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
  auto size = (unsigned int)(&_binary_param_bin_end - &_binary_param_bin_start);
  void *buffer = malloc(size);
  memcpy(buffer, &_binary_param_bin_start, size);
  return buffer;
}

#else

extern char constPackFilePath[];
extern int64_t filePathStrLen;

void *getEmbeddedConstPool(int64_t _) {
  char *fname = (char *)calloc(1, filePathStrLen + 1);
  memcpy(fname, constPackFilePath, filePathStrLen);

  printf("Getting packed constants from %s\n", fname);

  FILE *fileptr;
  char *buffer;
  long filelen;

  //  printf("%s\n", &constPackFilePath[0]);
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
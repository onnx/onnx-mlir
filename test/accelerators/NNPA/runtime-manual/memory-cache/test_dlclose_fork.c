/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Manual test of OMMemoryCache.c (see README.md): fork() after the library
// that contains the cache was unloaded.
//
// On Linux the cache registers pthread_atfork handlers. glibc must remove them
// when the library is dlclose'd; otherwise the next fork() calls into unmapped
// code and crashes. Usage: test_dlclose_fork <path to libomcachetest.so>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// Fork; the child exits at once. Return 1 if the child exited normally.
static int forkOnce(void) {
  pid_t pid = fork();
  if (pid < 0)
    return 0;
  if (pid == 0)
    _exit(0);
  int status;
  if (waitpid(pid, &status, 0) != pid)
    return 0;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <libomcachetest.so>\n", argv[0]);
    return 2;
  }
  for (int round = 0; round < 3; round++) {
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      fprintf(stderr, "dlopen failed: %s\n", dlerror());
      return 1;
    }
    int (*use)(void) = (int (*)(void))dlsym(lib, "dsoUseCache");
    if (!use || use() != 0) {
      fprintf(stderr, "dsoUseCache failed\n");
      return 1;
    }
    // While loaded: the handlers (if any) run and must work.
    if (!forkOnce()) {
      fprintf(stderr, "fork while loaded failed\n");
      return 1;
    }
    if (dlclose(lib) != 0) {
      fprintf(stderr, "dlclose failed: %s\n", dlerror());
      return 1;
    }
    // After unload: a handler left behind would crash here (in the parent,
    // before the child exists, for the prepare handler).
    if (!forkOnce()) {
      fprintf(stderr, "fork after dlclose failed\n");
      return 1;
    }
    printf("round %d: dlopen, use, fork, dlclose, fork: ok\n", round);
  }
  printf("test_dlclose_fork: ALL OK\n");
  return 0;
}

/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- ErrorMessage.hpp --------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//
// Return error message for given error number for opening file at given path
// If error number is not given, investigate it by opening it.
//
#define PATH_SIZE 4096
static string getErrorMessageforFileOpeningErrors(
    const string &path, int msgnum, int flags, int mode) {
  // If errno not given investigate the error by opening the path
  if (msgnum < 0) {
    flags = (flags > 0) ? flags : (O_CREAT | O_WRONLY);
    mode = (mode >= 0) ? mode : (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int fd = open(path.c_str(), flags, mode);
    msgnum = errno;
    close(fd);
  }
  char dir[PATH_SIZE];
  getcwd(dir, PATH_SIZE);
  string msg = string(strerror(msgnum)) + "(" + std::to_string(msgnum) +
               ") for " + path + " at " + dir;
  return msg;
}

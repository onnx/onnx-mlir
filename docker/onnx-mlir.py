#!/usr/bin/env python3

import os
import re
import shutil
import stat
import subprocess
import sys

DOCKER_SOCKET   = '/var/run/docker.sock'
ONNX_MLIR_IMAGE = 'onnxmlirczar/onnx-mlir'
WORK_DIR        = '/workdir'
OUTPUT_DIR      = '/output'
EMIT_IR_OPTS    = [ '--EmitONNXBasic',
                    '--EmitONNXIR',
                    '--EmitMLIR',
                    '--EmitLLVMIR' ]
EMIT_BIN_OPTS   = [ '--EmitLib',
                    '--EmitObj',
                    '--EmitJNI' ]

# When running onnx-mlir inside a docker container, the directory
# containing the input ONNX model file must be mounted into the
# container for onnx-mlir to read the input and generate the output.
#
# This convenient script will do that automatically to make it
# as if you are running onnx-mlir directly on the host.
def main():
    # Make sure docker client is installed
    if not shutil.which('docker'):
        print('docker client not found')
        return

    # Make sure docker daemon is running
    if not stat.S_ISSOCK(os.stat(DOCKER_SOCKET).st_mode):
        print('docker daemon not running')
        return

    # Pull the latest onnxmlirczar/onnx-mlir image, if image
    # is already up-to-date, pull will do nothing.
    args = [ 'docker', 'pull', ONNX_MLIR_IMAGE ]
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    # Only print messages related to pulling a layer or error.
    for l in proc.stdout:
        line = l.decode('utf-8')
        print(line if re.match('^([0-9a-f]{12})|Error', line) else '',
              end='', flush=True)
    proc.wait()
    if (proc.returncode != 0):
        print("docker pull failed")
        return

    # Go through the command line options and locate the
    # input ONNX model file.
    argi  = 0
    ionnx = None
    argo  = 0
    obase = None
    argv  = sys.argv
    argc  = len(sys.argv)

    for i, arg in enumerate(argv):
        # File specified on the first argument, defaults to --EmitLib
        if i == 1 and not argv[i].startswith('-'):
            argi = i
            ionnx = argv[i]
        # If a file is not specified on the first argument, it must be
        # specified after a valid --EmitXXX option.
        elif (arg in EMIT_IR_OPTS+EMIT_BIN_OPTS and i < argc-1 and
              not argv[i+1].startswith('-')):
            # File specified more than once, treat as not specified
            if ionnx:
                sys.exit("Too many --EmitXXX options")
            if (arg in EMIT_BIN_OPTS and sys.platform != 'linux'):
                print(('Warning: host {} is not linux, ' +
                       'output not directly usable').format(sys.platform))
            argi = i + 1
            ionnx = argv[argi]
        elif (arg == "-o" and i < argc-1 and
              not argv[i+1].startswith('-')):
            if obase:
                sys.exit("Too many -o options")
            argo = i + 1
            obase = argv[argo]

    # Prepare the arguments for docker run
    args = [ 'docker', 'run', '--rm', '-ti' ]

    # Construct the mount option if an input ONNX model file is found
    if ionnx:
        p = os.path.abspath(ionnx)
        d = os.path.dirname(p)
        f = os.path.basename(p)

        # Add the mount option, directory containing the input
        # ONNX model file will be mounted under /workdir inside
        # the container. If /workdir doesn't exist, it will be
        # created.
        args.append('-v')
        args.append(d + ':' + WORK_DIR + d)

        # Change directory into /workdir
        #args.append('-w')
        #args.append(WORK_DIR)

        # Rewrite the original input ONNX model file, which will
        # reside under /workdir inside the container.
        argv[argi] = WORK_DIR + p

    # Construct the mount option if -o is specified
    if obase:
        # Check invalid -o values such as ".", "..", "/.", "./", etc.
        if re.match('(.*/)*\.*$', obase):
            sys.exit("Invalid value for -o option")

        p = os.path.abspath(obase)
        d = os.path.dirname(p)
        f = os.path.basename(p)

        # Add the mount option, directory containing the output
        # files will be mounted under /output inside the container.
        # If /output/... doesn't exist, it will be
        # created.
        args.append('-v')
        args.append(d + ':' + OUTPUT_DIR + d)

        # Rewrite the original output basename, which will
        # reside under /output inside the container.
        argv[argo] = OUTPUT_DIR + p

    # Add effective uid and gid
    args.append('-u')
    args.append(str(os.geteuid()) + ':' + str(os.getegid()))

    # Add image name
    args.append(ONNX_MLIR_IMAGE)

    # Pass in all the original arguments
    argv.remove(argv[0])
    args.extend(argv)
    # print(args) #debug only

    # Run onnx-mlir in the container
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in proc.stdout:
        # Remove first occurrence of /workdir or /output before printing
        print(re.sub(WORK_DIR + '|' + OUTPUT_DIR, '',
                     line.decode('utf-8'), 1),
              end='', flush=True)
    proc.wait()
    if (proc.returncode != 0):
        print(os.strerror(proc.returncode))

if __name__ == "__main__":
    main()

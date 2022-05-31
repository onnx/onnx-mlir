#!/usr/bin/env python3

import logging
import math
import os
import subprocess
import sys

logging.basicConfig(
    level = logging.INFO, format = '[%(asctime)s] %(levelname)s: %(message)s')

# Set parallel jobs based on both CPU count and memory size.
# Because using CPU count alone can result in out of memory
# and get Jenkins killed. For example, we may have 64 CPUs
# (128 threads) and only 32GB memory. So spawning off 128
# cc/c++ processes is going to quickly exhaust the memory.
#
# Algorithm: NPROC = min(2, # of CPUs) if memory < 8GB, otherwise
#            NPROC = min(memory / 4, # of CPUs)
MEMORY_IN_GB               = (os.sysconf('SC_PAGE_SIZE') *
                              os.sysconf('SC_PHYS_PAGES') / (1024.**3))
NPROC                      = str(math.ceil(min(max(2, MEMORY_IN_GB/4), os.cpu_count())))

JENKINS_HOME               = os.getenv('JENKINS_HOME')
JOB_NAME                   = os.getenv('JOB_NAME')
WORKSPACE                  = os.getenv('WORKSPACE')
MODELZOO_WORKDIR           = os.getenv('MODELZOO_WORKDIR')
MODELZOO_HTML              = os.getenv('MODELZOO_HTML')
MODELZOO_STDOUT            = os.getenv('MODELZOO_STDOUT')

DOCKER_DEV_IMAGE_WORKDIR   = '/workdir'
RUN_ONNX_MODEL_PY          = 'RunONNXModel.py'
RUN_ONNX_MODELZOO_PY       = 'RunONNXModelZoo.py'

WORKSPACE_MODELZOO         = os.path.join(JENKINS_HOME, 'workspace',
                                          JOB_NAME+'@'+MODELZOO_WORKDIR)

github_repo_name           = os.getenv('GITHUB_REPO_NAME')
github_repo_name2          = os.getenv('GITHUB_REPO_NAME').replace('-', '_')
github_pr_baseref          = os.getenv('GITHUB_PR_BASEREF')
github_pr_baseref2         = os.getenv('GITHUB_PR_BASEREF').lower()
docker_dev_image_name      = (github_repo_name + '-dev' +
                              ('.' + github_pr_baseref2
                               if github_pr_baseref != 'main' else ''))

def main():
    cmd = [ 'docker', 'run', '--rm', '-ti',
            '-e', ('ONNX_MLIR_HOME=' +
                   os.path.join(WORKSPACE, 'build', 'Debug')),
            '-v', (WORKSPACE_MODELZOO + ':' +
                   os.path.join(DOCKER_DEV_IMAGE_WORKDIR, MODELZOO_WORKDIR)),
            '-v', (os.path.join(WORKSPACE, 'utils', RUN_ONNX_MODEL_PY) + ':' +
                   os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODEL_PY)),
            '-v', (os.path.join(WORKSPACE, 'utils', RUN_ONNX_MODELZOO_PY) + ':' +
                   os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODELZOO_PY)),
            docker_dev_image_name,
            os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODELZOO_PY),
            '-j', NPROC,
            '-w', MODELZOO_WORKDIR,
            '-H', MODELZOO_HTML,
            '-l', 'debug',
            '-k' ]

    logging.info(' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # print messages from RunONNXModelZoo.py and RunONNXModel.py
    for line in proc.stderr:
        print(line.decode('utf-8'), file=sys.stderr)

    proc.wait()

    if proc.returncode:
        sys.exit(proc.returncode)

    # write summary line to file for Jenkinsfile to pickup
    with open(os.path.join(WORKSPACE_MODELZOO, MODELZOO_STDOUT), 'w') as f:
        f.write(proc.stdout.decode('utf-8'))
    sys.exit(0)


if __name__ == "__main__":
    main()

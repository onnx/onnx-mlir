#!/usr/bin/env python3

from jenkins_common import *

docker_usr_image_workdir = "/workdir"
docker_usr_image_full = (
    (docker_registry_host_name + "/" if docker_registry_host_name else "")
    + (docker_registry_user_name + "/" if docker_registry_user_name else "")
    + docker_usr_image_name
    + ":"
    + pr_image_tag
)

BUILD_DIR = "build-ompyrt"

workspace_workdir = os.path.join(jenkins_workspace_dir, BUILD_DIR)
test_dir = os.path.join(workspace_workdir, "src/Runtime/python/om_pyrt/tests")


def main():
    os.makedirs(workspace_workdir)

    cmd_configure = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DONNX_MLIR_TARGET_TO_BUILD=OMPyRt",
        "..",
    ]

    cmd_build = [
        "cmake",
        "--build",
        ".",
        "--target",
        "OMCreateOMPyRtPackage",
    ]

    # pip install
    cmd_pip = [
        "pip3",
        "install",
        "src/Runtime/python/om_pyrt",
        "--prefix=/usr",
        "--no-build-isolation",
    ]

    for cmd in [cmd_configure, cmd_build, cmd_pip]:
        logging.info(" ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            print(line.decode("utf-8"), end="", flush=True)
        proc.wait()
        if proc.returncode != 0:
            sys.exit(proc.returncode)

    """
    # commands in test_dir
    cmd_use_local = ["python", "use_local_compiler.py", "--image", docker_usr_image_full]

    logging.info(" ".join(cmd_use_local))
    proc = subprocess.Popen(cmd_use_local, cwd=test_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode("utf-8"), end="", flush=True)
    proc.wait()
    """
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()

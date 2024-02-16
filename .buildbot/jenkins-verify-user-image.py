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

MNIST_ONNX = "mnist.onnx"
MNIST_DIR = "mnist"

workspace_workdir = os.path.join(jenkins_workspace_dir, MNIST_DIR)
container_workdir = os.path.join(docker_usr_image_workdir, MNIST_DIR)


def main():
    os.makedirs(workspace_workdir)

    # Copy docs/mnist_example/mnist.onnx to workspace_workdir
    shutil.copy(
        os.path.join(jenkins_workspace_dir, "docs", "mnist_example", MNIST_ONNX),
        os.path.join(workspace_workdir, MNIST_ONNX),
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        str(os.geteuid()) + ":" + str(os.getegid()),
        "-v",
        workspace_workdir + ":" + container_workdir,
        docker_usr_image_full,
        "--EmitLib",
        os.path.join(container_workdir, MNIST_ONNX),
    ]

    logging.info(" ".join(cmd))

    # Run the user image to compile the model
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in proc.stdout:
        print(line.decode("utf-8"), end="", flush=True)

    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

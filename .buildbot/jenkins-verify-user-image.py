#!/usr/bin/env python3

import datetime
import git
import logging
import math
import os
import requests
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DOCKER_USR_IMAGE_WORKDIR = "/workdir"

MNIST8_URL = (
    "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/"
)
MNIST8_ONNX = "mnist-8.onnx"
MNIST8_DIR = "mnist-8"

docker_daemon_socket = os.getenv("DOCKER_DAEMON_SOCKET")
docker_registry_host_name = os.getenv("DOCKER_REGISTRY_HOST_NAME")
docker_registry_user_name = os.getenv("DOCKER_REGISTRY_USER_NAME")
github_repo_name = os.getenv("GITHUB_REPO_NAME")
github_repo_name2 = os.getenv("GITHUB_REPO_NAME").replace("-", "_")
github_pr_baseref = os.getenv("GITHUB_PR_BASEREF")
github_pr_baseref2 = os.getenv("GITHUB_PR_BASEREF").lower()
github_pr_number = os.getenv("GITHUB_PR_NUMBER")
jenkins_home = os.getenv("JENKINS_HOME")
job_name = os.getenv("JOB_NAME")
workspace_dir = os.getenv("WORKSPACE")

docker_usr_image_name = github_repo_name + (
    "." + github_pr_baseref2 if github_pr_baseref != "main" else ""
)
docker_usr_image_tag = github_pr_number.lower()
docker_usr_image_full = (
    (docker_registry_host_name + "/" if docker_registry_host_name else "")
    + (docker_registry_user_name + "/" if docker_registry_user_name else "")
    + docker_usr_image_name
    + ":"
    + docker_usr_image_tag
)

workspace_workdir = os.path.join(workspace_dir, MNIST8_DIR)
container_workdir = os.path.join(DOCKER_USR_IMAGE_WORKDIR, MNIST8_DIR)


def urlretrieve(remote_url, local_file):
    req = requests.get(remote_url)
    with open(local_file, "wb") as f:
        f.write(req.content)


def main():
    os.makedirs(workspace_workdir)

    # Download mnist-8.onnx
    urlretrieve(MNIST8_URL + MNIST8_ONNX, os.path.join(workspace_workdir, MNIST8_ONNX))

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
        os.path.join(container_workdir, MNIST8_ONNX),
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

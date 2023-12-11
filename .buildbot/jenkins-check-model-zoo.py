#!/usr/bin/env python3

from jenkins_common import *

ONNX_MLIR_SOURCE = "/workdir/onnx-mlir"
ONNX_MLIR_HOME = "/workdir/onnx-mlir/build/Debug"
RUN_ONNX_MODEL_PY = "RunONNXModel.py"
RUN_ONNX_MODELZOO_PY = "RunONNXModelZoo.py"

RENDERJSON_URL = "https://raw.githubusercontent.com/caldwell/renderjson/master/"
RENDERJSON_JS = "renderjson.js"

modelzoo_reportdir = os.getenv("MODELZOO_REPORTDIR")
modelzoo_workdir = os.getenv("MODELZOO_WORKDIR")
modelzoo_html = os.getenv("MODELZOO_HTML")
modelzoo_stdout = os.getenv("MODELZOO_STDOUT")

docker_dev_image_full = (
    (docker_registry_host_name + "/" if docker_registry_host_name else "")
    + (docker_registry_user_name + "/" if docker_registry_user_name else "")
    + docker_dev_image_name
    + ":"
    + pr_image_tag
)

# History directory is just the job directory
workspace_historydir = os.path.join(jenkins_home, "jobs", jenkins_job_name)
container_historydir = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, jenkins_job_name)

workspace_reportdir = os.path.join(jenkins_workspace_dir, modelzoo_reportdir)
container_reportdir = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, modelzoo_reportdir)

workspace_workdir = os.path.join(jenkins_workspace_dir, modelzoo_workdir)
container_workdir = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, modelzoo_workdir)

workspace_model_py = os.path.join(jenkins_workspace_dir, "utils", RUN_ONNX_MODEL_PY)
container_model_py = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODEL_PY)

workspace_modelzoo_py = os.path.join(
    jenkins_workspace_dir, "utils", RUN_ONNX_MODELZOO_PY
)
container_modelzoo_py = os.path.join(DOCKER_DEV_IMAGE_WORKDIR, RUN_ONNX_MODELZOO_PY)


def main():
    repo = git.Repo(".")
    head_commit_message = repo.head.commit.message.split("\n", 1)[0]
    head_commit_author = "{} <{}>".format(
        repo.head.commit.author.name, repo.head.commit.author.email
    )
    head_commit_hash = repo.head.commit.hexsha
    head_commit_date = (
        datetime.datetime.utcfromtimestamp(repo.head.commit.committed_date).isoformat()
        + "Z"
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        str(os.geteuid()) + ":" + str(os.getegid()),
        "-e",
        "ONNX_MLIR_HOME=" + ONNX_MLIR_HOME,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_MESSAGE=" + head_commit_message,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_AUTHOR=" + head_commit_author,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_HASH=" + head_commit_hash,
        "-e",
        "ONNX_MLIR_HEAD_COMMIT_DATE=" + head_commit_date,
        "-v",
        workspace_historydir + ":" + container_historydir,
        "-v",
        workspace_reportdir + ":" + container_reportdir,
        "-v",
        workspace_workdir + ":" + container_workdir,
        "-v",
        workspace_model_py + ":" + container_model_py,
        "-v",
        workspace_modelzoo_py + ":" + container_modelzoo_py,
        docker_dev_image_full,
        container_modelzoo_py,
        "-H",
        modelzoo_html,
        "-j",
        NPROC,
        "-l",
        "info",
        "-q",
        container_historydir,
        "-r",
        container_reportdir,
        "-w",
        container_workdir,
    ]

    # Write summary line to file for Jenkinsfile to pickup
    logging.info(" ".join(cmd))
    os.makedirs(workspace_workdir)
    os.makedirs(workspace_reportdir)
    with open(os.path.join(workspace_reportdir, modelzoo_stdout), "w") as f:
        try:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)

            # print messages from RunONNXModelZoo.py and RunONNXModel.py
            for line in proc.stderr:
                print(line.decode("utf-8"), file=sys.stderr, end="", flush=True)

            proc.wait()
        except:
            f.write("failed")

    # Download renderjson.js
    urlretrieve(
        RENDERJSON_URL + RENDERJSON_JS, os.path.join(workspace_reportdir, RENDERJSON_JS)
    )


if __name__ == "__main__":
    main()

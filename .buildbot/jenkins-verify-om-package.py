#!/usr/bin/env python3

from jenkins_common import *

OM_PYRT_LIGHT_DOCKERFILE = "docker/Dockerfile.om-pyrt-light"


def main():
    image_name = github_repo_name + "-om-pyrt-light"
    image_tag = pr_image_tag
    image_full = image_name + ":" + image_tag

    layer_sha256 = ""
    for line in docker_api.build(
        path=".",
        dockerfile=OM_PYRT_LIGHT_DOCKERFILE,
        tag=image_full,
        decode=True,
        rm=True,
        buildargs={
            "NPROC": NPROC,
            GITHUB_REPO_NAME2 + "_PR_NUMBER": github_pr_number,
            GITHUB_REPO_NAME2 + "_PR_NUMBER2": github_pr_number2,
        },
    ):
        if "stream" in line:
            m = re.match(r"^\s*---> ([0-9a-f]+)$", line["stream"])
            if m:
                layer_sha256 = m.group(1)
            print(line["stream"], end="", flush=True)

        if "error" in line:
            if layer_sha256:
                image_layer = "sha256:" + layer_sha256
                remove_dependent_containers(image_layer)
                logging.info("tagging %s -> %s for debugging", image_layer, image_full)
                docker_api.tag(image_layer, image_name, image_tag, force=True)
            else:
                logging.info("no successful image layer for tagging")
            raise Exception(line["error"])

    id = docker_api.images(name=image_full, all=False, quiet=True)
    logging.info("image %s (%s) built", image_full, id[0][0:19])

    # Run test inside the built image
    docker_usr_image_full = (
        (docker_registry_host_name + "/" if docker_registry_host_name else "")
        + (docker_registry_user_name + "/" if docker_registry_user_name else "")
        + docker_usr_image_name
        + ":"
        + pr_image_tag
    )

    test_script = os.path.join(
        jenkins_workspace_dir,
        "src/Runtime/python/om_pyrt/tests/use_container_compiler.py",
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-v",
        jenkins_workspace_dir + ":" + jenkins_workspace_dir,
        image_full,
        "python3",
        test_script,
        "--image",
        docker_usr_image_full,
    ]

    logging.info(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode("utf-8"), end="", flush=True)
    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

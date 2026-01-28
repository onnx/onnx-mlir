#!/usr/bin/env python3

from jenkins_common import *

LLVM_PROJECT_IMAGE = {"dev": docker_static_image_name, "usr": docker_shared_image_name}
ONNX_MLIR_IMAGE = {"dev": docker_dev_image_name, "usr": docker_usr_image_name}
ONNX_MLIR_DOCKERFILE = {
    "dev": "docker/Dockerfile." + github_repo_name + "-dev",
    "usr": "docker/Dockerfile." + github_repo_name,
}
ONNX_MLIR_LABELS = [
    github_repo_name2 + "_sha1",
    github_repo_name2 + "_sha1_date",
    github_repo_name2 + "_dockerfile_sha1",
]


def get_onnx_mlir_info(image_type, local_repo):
    """Get project repo commit sha1 and date we are expecting to build
    from the local pull request repo."""

    repo = git.Repo(local_repo)
    exp_onnx_mlir_sha1 = repo.head.commit.hexsha
    exp_onnx_mlir_sha1_date = (
        datetime.datetime.utcfromtimestamp(repo.head.commit.committed_date).isoformat()
        + "Z"
    )

    exp_onnx_mlir_dockerfile_sha1 = compute_file_sha1(ONNX_MLIR_DOCKERFILE[image_type])

    # Labels used to filter local images
    exp_onnx_mlir_filter = {
        "label": [
            github_repo_name2 + "_sha1=" + exp_onnx_mlir_sha1,
            github_repo_name2 + "_dockerfile_sha1=" + exp_onnx_mlir_dockerfile_sha1,
            github_repo_name2 + "_successfully_built=yes",
        ]
    }

    logging.info("%s expected", ONNX_MLIR_IMAGE[image_type])
    logging.info("commit sha1:     %s", exp_onnx_mlir_sha1)
    logging.info("commit date:     %s", exp_onnx_mlir_sha1_date)
    logging.info("dockerfile sha1: %s", exp_onnx_mlir_dockerfile_sha1)
    logging.info("image filter:    %s", exp_onnx_mlir_filter)

    return {
        github_repo_name2 + "_sha1": exp_onnx_mlir_sha1,
        github_repo_name2 + "_sha1_date": exp_onnx_mlir_sha1_date,
        github_repo_name2 + "_dockerfile_sha1": exp_onnx_mlir_dockerfile_sha1,
        github_repo_name2 + "_filter": exp_onnx_mlir_filter,
    }


def build_per_pr_onnx_mlir(image_type, exp):
    """Build onnx-mlir dev and user images."""

    host_name = docker_registry_host_name
    user_name = docker_registry_user_name
    login_name = docker_registry_login_name
    login_token = docker_registry_login_token
    base_image_name = LLVM_PROJECT_IMAGE[image_type]
    base_image_repo = (
        (host_name + "/" if host_name else "")
        + (user_name + "/" if user_name else "")
        + base_image_name
    )
    base_image_tag = pr_image_tag
    image_name = ONNX_MLIR_IMAGE[image_type]
    image_repo = (
        (host_name + "/" if host_name else "")
        + (user_name + "/" if user_name else "")
        + image_name
    )
    image_tag = pr_image_tag
    image_full = image_repo + ":" + image_tag
    image_arch = image_repo + ":" + cpu_arch
    image_filter = exp[github_repo_name2 + "_filter"]
    image_labels = ONNX_MLIR_LABELS

    # First look for a local project image for the pull request that
    # was built by a previous build job. We can use it if it has the
    # expected project repo sha1, which means that the repo hasn't changed.
    # This is useful for situations where we trigger the build by the
    # "{test|publish} this please" comment phrase for various testing
    # purposes without actually changing the repo itself, e.g.,
    # testing different Jenkins job configurations.
    #
    # Note that, unlike the case with llvm-project images, we don't need
    # to check the dockerfile sha1 used to built the onnx-mlir images
    # because the dockerfile is part of onnx-mlir. If we changed it, then
    # onnx-mlir commit sha1 would have changed.
    id = docker_api.images(name=image_full, filters=image_filter, all=False, quiet=True)

    # If a local useable project image was not found, see if we can
    # pull one from the registry.
    if not id:
        # Acquire read lock to pull the arch image. This is to serialize
        # against other PR merges trying to push (writers) the arch image.
        # PR builds trying to pull (readers) the arch image can go concurrently.
        logging.info("acquiring read lock for pulling %s", image_arch)
        docker_rwlock.acquire_read_lock()
        try:
            labels = get_remote_image_labels(
                host_name,
                user_name,
                image_name,
                cpu_arch,
                image_labels,
                login_name,
                login_token,
            )

            # Image in registry has expected onnx-mlir commit sha1, pull and
            # tag it with pull request number for our private use.
            if (
                labels
                and labels[github_repo_name2 + "_sha1"]
                == exp[github_repo_name2 + "_sha1"]
            ):
                for line in docker_api.pull(
                    image_repo, tag=cpu_arch, stream=True, decode=True
                ):
                    print(
                        (
                            line["id"] + ": "
                            if "id" in line and "progress" not in line
                            else ""
                        )
                        + (
                            line["status"] + "\n"
                            if "status" in line and "progress" not in line
                            else ""
                        ),
                        end="",
                        flush=True,
                    )

                # Tag pulled arch image with pull request number then remove
                # the arch image
                docker_api.tag(image_arch, image_repo, image_tag, force=True)
                docker_api.remove_image(image_arch, force=True)

                # For logging purpose only
                id = docker_api.images(name=image_full, all=False, quiet=True)
                logging.info("image %s (%s) tagged", image_full, id[0][0:19])
                return
        except Exception as e:
            logging.exception(e)
        # Remove arch image and release lock regardless of exception or not
        finally:
            docker_rwlock.release_read_lock()
            logging.info("released read lock for pulling %s", image_arch)

        # Build project locally if one of the following is true
        #
        # - image in registry does not exist
        # - pull image failed
        # - image in registry has a project repo commit sha1 different
        #   from what we expect
        #
        layer_sha256 = ""
        for line in docker_api.build(
            path=".",
            dockerfile=ONNX_MLIR_DOCKERFILE[image_type],
            tag=image_repo + ":" + image_tag,
            platform="linux/" + cpu_arch,
            decode=True,
            rm=True,
            buildargs={
                "BASE_IMAGE": base_image_repo + ":" + base_image_tag,
                "NPROC": NPROC,
                GITHUB_REPO_NAME2 + "_SHA1": exp[github_repo_name2 + "_sha1"],
                GITHUB_REPO_NAME2 + "_SHA1_DATE": exp[github_repo_name2 + "_sha1_date"],
                GITHUB_REPO_NAME2
                + "_DOCKERFILE_SHA1": exp[github_repo_name2 + "_dockerfile_sha1"],
                GITHUB_REPO_NAME2 + "_PR_NUMBER": github_pr_number,
                GITHUB_REPO_NAME2 + "_PR_NUMBER2": github_pr_number2,
            },
        ):
            if "stream" in line:
                # Keep track of the latest successful image layer
                m = re.match(r"^\s*---> ([0-9a-f]+)$", line["stream"])
                if m:
                    layer_sha256 = m.group(1)
                print(line["stream"], end="", flush=True)

            if "error" in line:
                # Tag the latest successful image layer for easier debugging.
                #
                # It's OK to tag the broken image since it will not have the
                # onnx_mlir_successfully_built=yes label so it will not be
                # incorrectly reused.
                if layer_sha256:
                    image_layer = "sha256:" + layer_sha256
                    remove_dependent_containers(image_layer)
                    logging.info(
                        "tagging %s -> %s for debugging", image_layer, image_full
                    )
                    docker_api.tag(image_layer, image_repo, image_tag, force=True)
                else:
                    logging.info("no successful image layer for tagging")
                raise Exception(line["error"])

        id = docker_api.images(name=image_full, all=False, quiet=True)
        logging.info("image %s (%s) built", image_full, id[0][0:19])

    # Found useable local image
    else:
        logging.info("image %s (%s) found", image_full, id[0][0:19])


def main():
    # Ensure cpu_arch matches the Docker daemon/host to avoid pulling the wrong arch
    resolve_and_override_cpu_arch_from_docker()

    build_per_pr_onnx_mlir("dev", get_onnx_mlir_info("dev", "."))
    build_per_pr_onnx_mlir("usr", get_onnx_mlir_info("usr", "."))


if __name__ == "__main__":
    main()

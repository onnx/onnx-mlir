#!/usr/bin/env python3

from jenkins_common import *

LLVM_PROJECT_SHA1_FILE = "utils/clone-mlir.sh"
LLVM_PROJECT_SHA1_REGEX = "git checkout ([0-9a-f]+)"
LLVM_PROJECT_DOCKERFILE = "docker/Dockerfile.llvm-project"
LLVM_PROJECT_GITHUB_URL = "https://api.github.com/repos/llvm/llvm-project"
LLVM_PROJECT_BASE_IMAGE = {
    "static": "ghcr.io/onnxmlir/ubuntu:noble-",  # Will append cpu_arch
    "shared": "registry.access.redhat.com/ubi9-minimal:latest",  # No arch suffix needed
}
LLVM_PROJECT_BASE_IMAGE_NEEDS_ARCH = {
    "static": True,
    "shared": False,  # UBI9 uses manifest lists, no arch-specific tags
}
LLVM_PROJECT_IMAGE = {
    "static": docker_static_image_name,
    "shared": docker_shared_image_name,
}
LLVM_PROJECT_LABELS = [
    "llvm_project_sha1",
    "llvm_project_sha1_date",
    "llvm_project_dockerfile_sha1",
]
LLVM_PROJECT_BUILD_SHARED_LIBS = {"static": "off", "shared": "on"}


def extract_llvm_project_info():
    """From the pull request source, extract expected llvm-project sha1, sha1 date,
    and dockerfile sha1."""

    exp_llvm_project_sha1 = extract_pattern_from_file(
        LLVM_PROJECT_SHA1_FILE, LLVM_PROJECT_SHA1_REGEX
    )
    exp_llvm_project_sha1_date = get_repo_sha1_date(
        LLVM_PROJECT_GITHUB_URL, exp_llvm_project_sha1, github_repo_access_token
    )
    exp_llvm_project_dockerfile_sha1 = compute_file_sha1(LLVM_PROJECT_DOCKERFILE)

    # Labels used to filter local images
    exp_llvm_project_filter = {
        "label": [
            "llvm_project_sha1=" + exp_llvm_project_sha1,
            "llvm_project_dockerfile_sha1=" + exp_llvm_project_dockerfile_sha1,
            "llvm_project_successfully_built=yes",
        ]
    }

    logging.info("llvm-project expected")
    logging.info("commit sha1:     %s", exp_llvm_project_sha1)
    logging.info("commit date:     %s", exp_llvm_project_sha1_date)
    logging.info("dockerfile sha1: %s", exp_llvm_project_dockerfile_sha1)
    logging.info("image filter:    %s", exp_llvm_project_filter)

    return {
        "llvm_project_sha1": exp_llvm_project_sha1,
        "llvm_project_sha1_date": exp_llvm_project_sha1_date,
        "llvm_project_dockerfile_sha1": exp_llvm_project_dockerfile_sha1,
        "llvm_project_filter": exp_llvm_project_filter,
    }


def setup_per_pr_llvm_project(image_type, exp):
    """Pull or build llvm-project images, which is required for building our
    onnx-mlir dev and user images. Each pull request will be using its own
    "private" llvm-project images, which have the pull request number as
    the image tag."""

    host_name = docker_registry_host_name
    user_name = docker_registry_user_name
    login_name = docker_registry_login_name
    login_token = docker_registry_login_token
    image_name = LLVM_PROJECT_IMAGE[image_type]
    image_tag = pr_image_tag
    image_repo = (
        (host_name + "/" if host_name else "")
        + (user_name + "/" if user_name else "")
        + image_name
    )
    image_full = image_repo + ":" + image_tag
    image_arch = image_repo + ":" + cpu_arch
    image_filter = exp["llvm_project_filter"]
    image_labels = LLVM_PROJECT_LABELS

    # First look for a local llvm-project image for the pull request that
    # was built by a previous build job. We can use it if it has both the
    # expected llvm-project sha1 and Dockerfile.llvm-project sha1 (i.e.,
    # the pull request did not modify the Dockerfile.llvm-project that was
    # used to build the llvm-project image.
    id = docker_api.images(name=image_full, filters=image_filter, all=False, quiet=True)

    # If a local useable llvm-project image was not found, see if we can
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

            # Image in registry has expected llvm-project commit sha1 and
            # Dockerfile.llvm-project sha1, pull and tag it with pull request
            # number for our private use.
            if (
                labels
                and labels["llvm_project_sha1"] == exp["llvm_project_sha1"]
                and labels["llvm_project_dockerfile_sha1"]
                == exp["llvm_project_dockerfile_sha1"]
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
        except:
            labels["llvm_project_sha1_date"] = ""
        # Remove arch image and release lock regardless of exception or not
        finally:
            docker_rwlock.release_read_lock()
            logging.info("released read lock for pulling %s", image_arch)

        # Build llvm-project locally if one of the following is true
        #
        # - image in registry does not exist
        # - pull image failed
        # - image in registry has an invalid llvm-project commit sha1 date
        #   (should never happen)
        # - expected llvm-project commit sha1 date is invalid (fetch sha1
        #   date failed)
        # - image in registry has an llvm-project commit sha1 date earlier
        #   than what we expect (registry image out of date)
        #
        # Note that if pull failed labels['llvm_project_sha1_date'] will
        # be cleared to make valid_sha1_date false.
        if (
            not labels
            or not valid_sha1_date(labels["llvm_project_sha1_date"])
            or not valid_sha1_date(exp["llvm_project_sha1_date"])
            or labels["llvm_project_sha1_date"] <= exp["llvm_project_sha1_date"]
        ):
            layer_sha256 = ""
            # Conditionally append cpu_arch to base image
            base_image = LLVM_PROJECT_BASE_IMAGE[image_type]
            if LLVM_PROJECT_BASE_IMAGE_NEEDS_ARCH[image_type]:
                base_image += cpu_arch

            for line in docker_api.build(
                path=".",
                dockerfile=LLVM_PROJECT_DOCKERFILE,
                tag=image_full,
                platform=cpu_arch,
                decode=True,
                rm=True,
                buildargs={
                    "BASE_IMAGE": base_image,
                    "NPROC": NPROC,
                    "BUILD_SHARED_LIBS": LLVM_PROJECT_BUILD_SHARED_LIBS[image_type],
                    "LLVM_PROJECT_SHA1": exp["llvm_project_sha1"],
                    "LLVM_PROJECT_SHA1_DATE": exp["llvm_project_sha1_date"],
                    "LLVM_PROJECT_DOCKERFILE_SHA1": exp["llvm_project_dockerfile_sha1"],
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
                    # llvm_project_successfully_built=yes label so it will not be
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

        # Registry image has an llvm-project commit sha1 date later than what
        # we expect, the build source is out of date. Exit to fail the build,
        # regardless of Dockerfile.llvm-project sha1 being expected or not.
        else:
            raise Exception("PR source out of date, rebase then rebuild")

    # Found useable local image
    else:
        logging.info("image %s (%s) found", image_full, id[0][0:19])


def main():
    exp = extract_llvm_project_info()

    # Ensure cpu_arch matches the Docker daemon/host to avoid pulling the wrong arch
    resolve_and_override_cpu_arch_from_docker()

    setup_per_pr_llvm_project("static", exp)
    setup_per_pr_llvm_project("shared", exp)


if __name__ == "__main__":
    main()

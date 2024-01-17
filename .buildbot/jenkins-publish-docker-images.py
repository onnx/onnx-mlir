#!/usr/bin/env python3

from jenkins_common import *

# dot cannot be used in python dict key so we use dash
python_static_image_name = docker_static_image_name.replace(".", "-")
python_shared_image_name = docker_shared_image_name.replace(".", "-")
python_dev_image_name = docker_dev_image_name.replace(".", "-")
python_usr_image_name = docker_usr_image_name.replace(".", "-")

DOCKER_IMAGE_NAME = {
    "static": docker_static_image_name,
    "shared": docker_shared_image_name,
    "dev": docker_dev_image_name,
    "usr": docker_usr_image_name,
}
PYTHON_IMAGE_NAME = {
    "static": python_static_image_name,
    "shared": python_shared_image_name,
    "dev": python_dev_image_name,
    "usr": python_usr_image_name,
}
IMAGE_LABELS = {
    python_static_image_name: LLVM_PROJECT_LABELS,
    python_shared_image_name: LLVM_PROJECT_LABELS,
    python_dev_image_name: ONNX_MLIR_LABELS,
    python_usr_image_name: ONNX_MLIR_LABELS,
}
IMAGE_ARCHS = {"s390x", "amd64", "ppc64le"}
commit_sha1_date_label = {
    python_static_image_name: "llvm_project_sha1_date",
    python_shared_image_name: "llvm_project_sha1_date",
    python_dev_image_name: github_repo_name2 + "_sha1_date",
    python_usr_image_name: github_repo_name2 + "_sha1_date",
}
dockerfile_sha1_label = {
    python_static_image_name: "llvm_project_dockerfile_sha1",
    python_shared_image_name: "llvm_project_dockerfile_sha1",
    python_dev_image_name: github_repo_name2 + "_dockerfile_sha1",
    python_usr_image_name: github_repo_name2 + "_dockerfile_sha1",
}
pr_mergeable_state = {
    "behind": {"mergeable": False, "desc": "the head ref is out of date"},
    # see comments in image_publishable
    "blocked": {"mergeable": True, "desc": "the merge is blocked"},
    "clean": {"mergeable": True, "desc": "mergeable and passing commit status"},
    "dirty": {"mergeable": False, "desc": "the merge commit cannot be cleanly created"},
    "draft": {
        "mergeable": False,
        "desc": "the merge is blocked due to the pull request being a draft",
    },
    "has_hooks": {
        "mergeable": True,
        "desc": "mergeable with passing commit status and pre-receive hooks",
    },
    "unknown": {"mergeable": True, "desc": "the state cannot currently be determined"},
    "unstable": {"mergeable": True, "desc": "mergeable with non-passing commit status"},
}


def put_image_manifest(
    host_name,
    user_name,
    image_name,
    image_tag,
    manifest_list,
    login_name,
    login_token,
    access_token,
):
    """Make REST call to put multiarch manfiest list of an image in
    public docker registry."""

    # Put manifest
    url = (
        "https://"
        + (host_name if host_name else "registry-1.docker.io")
        + "/v2/"
        + (user_name + "/" if user_name else "")
        + image_name
        + "/manifests/"
        + image_tag
    )
    json = {
        "schemaVersion": 2,
        "mediaType": DOCKER_DIST_MANIFEST_LIST,
        "manifests": manifest_list,
    }
    headers = {}
    headers["Content-Type"] = DOCKER_DIST_MANIFEST_LIST
    if access_token:
        headers["Authorization"] = "Bearer " + access_token
        auth = None
    else:
        auth = (login_name, login_token)

    resp = requests.put(url=url, json=json, headers=headers, auth=auth)
    resp.raise_for_status()
    return resp


def get_pr_mergeable_state(url, token):
    """Get pull request source mergeable state."""

    try:
        resp = requests.get(
            url=url,
            headers={"Accept": "application/json", "Authorization": "token " + token},
        )
        resp.raise_for_status()
        return resp.json()["mergeable_state"]
    except Exception as e:
        logging.exception(e)
        return "unknown"


def image_publishable(
    host_name,
    user_name,
    docker_image_name,
    python_image_name,
    image_tag,
    image_labels,
    login_name,
    login_token,
):
    """Decide whether we should publish the local images or not."""

    # If local image is missing or has invalid labels, exception
    # will be raised to fail the build.
    local_labels = get_local_image_labels(
        host_name, user_name, docker_image_name, image_tag, image_labels
    )
    remote_labels = get_remote_image_labels(
        host_name,
        user_name,
        docker_image_name,
        cpu_arch,
        image_labels,
        login_name,
        login_token,
    )

    # If url is 'none', it's a push event from merging so skip
    # mergeable state check.
    #
    # Note that when our (and/or some other) build is marked as required,
    # while the build(s) are ongoing, the mergeable state will be "blocked".
    # So for publish triggered by "publish this please" phrase, we have a
    # catch 22 problem. But if we can come to this point, we know that at
    # least our build successfully built the docker images. So we allow
    # the blocked mergeable state to publish our images.
    if github_pr_request_url != "none":
        state = get_pr_mergeable_state(github_pr_request_url, github_repo_access_token)
        logging.info(
            "pull request url: %s, mergeable state: %s, %s",
            github_pr_request_url,
            state,
            pr_mergeable_state[state]["desc"],
        )
        if not pr_mergeable_state[state]["mergeable"]:
            raise Exception("publish aborted due to unmergeable state")
    if not remote_labels:
        logging.info("publish due to invalid remote labels")
        return True
    if github_pr_phrase == "publish":
        logging.info("publish forced by trigger phrase")
        return True
    if (
        local_labels[commit_sha1_date_label[python_image_name]]
        > remote_labels[commit_sha1_date_label[python_image_name]]
    ):
        logging.info("publish due to newer local sha1 date")
        return True
    # Commits can only be merged one at a time so it's guaranteed
    # that the same commit sha1 date will have the same commit sha1,
    # and vise versa.
    #
    # For llvm-project images, if commit sha1 are the same but the
    # dockerfile for building them changed, they will be published.
    # For onnx-mlir images, if commit sha1 are the same, it's
    # guaranteed the dockerfile for building them are the same, so
    # they will not be published.
    if (
        local_labels[commit_sha1_date_label[python_image_name]]
        == remote_labels[commit_sha1_date_label[python_image_name]]
        and local_labels[dockerfile_sha1_label[python_image_name]]
        != remote_labels[dockerfile_sha1_label[python_image_name]]
    ):
        logging.info("publish due to different dockerfile sha1")
        return True

    logging.info("publish skipped due to older or identical local image")
    return False


def publish_arch_image(
    host_name, user_name, image_name, image_tag, login_name, login_token
):
    """Publish an arch specific image."""

    image_repo = (
        (host_name + "/" if host_name else "")
        + (user_name + "/" if user_name else "")
        + image_name
    )
    image_pr = image_repo + ":" + image_tag
    image_arch = image_repo + ":" + cpu_arch

    # Acquire write lock to prepare for tagging to the arch image and
    # pushing it. This is to serialize against other PR merges trying
    # to push (writers) and/or other PR builds trying to pull (readers)
    # the arch image.
    logging.info("acquiring write lock for tagging and pushing %s", image_arch)
    docker_rwlock.acquire_write_lock()
    try:
        # Tag the image with arch
        logging.info("tagging %s -> %s", image_pr, image_arch)
        docker_api.tag(image_pr, image_repo, cpu_arch)

        # Push the image tagged with arch then remove it, regardless of
        # whether the push worked or not.
        for i in range(0, RETRY_LIMIT):
            try:
                logging.info("pushing %s [%s/%s]", image_arch, i, RETRY_LIMIT)
                for line in docker_api.push(
                    repository=image_repo,
                    tag=cpu_arch,
                    auth_config={"username": login_name, "password": login_token},
                    stream=True,
                    decode=True,
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
                logging.info("pushed %s [%s/%s]", image_arch, i, RETRY_LIMIT)
                break
            except Exception as e:
                logging.warning(
                    "pushing %s [%s/%s] failed: %s", image_arch, i, RETRY_LIMIT, e
                )
                continue
    # Remove arch image and release lock regardless of exception or not
    finally:
        docker_api.remove_image(image_arch, force=True)
        docker_rwlock.release_write_lock()
        logging.info("released write lock for tagging and pushing %s", image_arch)


def publish_multiarch_manifest(
    host_name, user_name, image_name, manifest_tag, login_name, login_token
):
    """Publish multiarch manifest for an image."""

    try:
        access_token = (
            get_access_token(
                host_name, user_name, image_name, login_name, login_token, "pull,push"
            )
            if strtobool(docker_registry_token_access)
            else None
        )

        # For each arch, construct the manifest element needed for the manifest
        # list by extracting fields from v2 image manifest and config.
        manifest_list = []
        for image_tag in IMAGE_ARCHS:
            m = {}
            manifest, config = get_image_manifest_config(
                host_name,
                user_name,
                image_name,
                image_tag,
                login_name,
                login_token,
                access_token,
            )
            m["mediaType"] = manifest.headers["Content-Type"]
            m["size"] = len(manifest.text)
            m["digest"] = manifest.headers["Docker-Content-Digest"]

            m["platform"] = {}
            m["platform"]["architecture"] = config.json()["architecture"]
            m["platform"]["os"] = config.json()["os"]

            manifest_list.append(m)
        logging.info("manifests: %s", manifest_list)

        # Make the REST call to PUT the multiarch manifest list.
        resp = put_image_manifest(
            host_name,
            user_name,
            image_name,
            manifest_tag,
            manifest_list,
            login_name,
            login_token,
            access_token,
        )

        logging.info("publish %s/%s:%s", user_name, image_name, manifest_tag)
        logging.info("        %s", resp.headers["Docker-Content-Digest"])
    except Exception as e:
        logging.exception(e)


def publish_image(image_type):
    """Publish an image if it should be published and publish multiarch manifest
    for developer and user images if necessary."""

    host_name = docker_registry_host_name
    user_name = docker_registry_user_name
    login_name = docker_registry_login_name
    login_token = docker_registry_login_token

    docker_image_name = DOCKER_IMAGE_NAME[image_type]
    python_image_name = PYTHON_IMAGE_NAME[image_type]
    image_tag = pr_image_tag
    image_labels = IMAGE_LABELS[python_image_name]

    # Decide if the image should be published or not
    if not image_publishable(
        host_name,
        user_name,
        docker_image_name,
        python_image_name,
        image_tag,
        image_labels,
        login_name,
        login_token,
    ):
        return

    # Publish the arch specific image
    publish_arch_image(
        host_name, user_name, docker_image_name, image_tag, login_name, login_token
    )

    # For developer and user images, we publish a multiarch manifest so we can
    # pull the images without having to explicitly specify the arch tag.
    if image_type == "dev" or image_type == "usr":
        publish_multiarch_manifest(
            host_name, user_name, docker_image_name, "latest", login_name, login_token
        )


def main():
    for image_type in ["static", "shared", "dev", "usr"]:
        publish_image(image_type)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import docker
import glob
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

IMAGE_NAMES          = [ 'onnx-mlir-llvm-static',
                         'onnx-mlir-llvm-shared',
                         'onnx-mlir-dev',
                         'onnx-mlir' ]

docker_daemon_socket = os.getenv('DOCKER_DAEMON_SOCKET')
dockerhub_user_name  = os.getenv('DOCKERHUB_USER_NAME')
jenkins_build_result = os.getenv('JENKINS_BUILD_RESULT')

onnx_mlir_pr_number  = os.getenv('ONNX_MLIR_PR_NUMBER')
onnx_mlir_pr_action  = os.getenv('ONNX_MLIR_PR_ACTION')
onnx_mlir_pr_merged  = os.getenv('ONNX_MLIR_PR_MERGED')

docker_api           = docker.APIClient(base_url=docker_daemon_socket)

# Cleanup docker images and containers associated with a pull request number.
# For action open/reopen/synchronize, only dangling images and containers are
# removed. For action close, non-dangling images and containers are removed.
def cleanup_docker_images(pr_number, dangling):
    # First find all the dangling docker images associated with the
    # pull request number
    filters = { 'dangling': True,
                'label': [ 'onnx_mlir_pr_number=' + pr_number ] }
    images = docker_api.images(filters = filters, quiet = True)

    # When a build is aborted the cleanup may try to remove an intermediate
    # image or container that the docker build process itself is already doing,
    # resulting a conflict. So we catch the exception and ignore it.

    # For each dangling image found, find and remove all the dependant
    # containers
    for image in images:
        containers = docker_api.containers(
            filters = { 'ancestor': image }, all = True, quiet = True)
        for container in containers:
            try:
                container_info = docker_api.inspect_container(container['Id'])
                logging.info('Removing     Id:%s', container['Id'])
                logging.info('   Image %s', container_info['Image'])
                logging.info('     Cmd %s', str(container_info['Config']['Cmd']))
                logging.info('  Labels %s', str(container_info['Config']['Labels']))
                docker_api.remove_container(container['Id'], v = True, force = True)
            except:
                logging.info(sys.exc_info()[1])

    # If we are doing final cleanup, i.e., dangling = False, add non-dangling
    # images to the list of images to be removed. The non-dangling images
    # shouldn't have running containers depending on them.
    #
    # Note the llvm-project images is built by a previous pull request until
    # we bump its commit sha1. So the filter will not catch them. And since
    # they can be shared by multiple pull requests (by tagging the s390x/amd64
    # images with a private pull request number), we can't remove them by using
    # the sha256 unless we force the removal. But we don't want to do that since
    # forceful removing a sha256 will remove all the shared repo:tag images. So
    # they are cleaned by untagging the image. Untagging is done by simply
    # passing the full image name instead of the image sha256 to remove_image.
    if not dangling:
        for image_name in IMAGE_NAMES:
            image_full = dockerhub_user_name + '/' + image_name + ':' + pr_number
            images.append(image_full)

    for image in images:
        # Remove the docker images associated with the pull request number
        try:
            image_info = docker_api.inspect_image(image)
            logging.info('Removing %s', image)
            logging.info('RepoTags %s', str(image_info['RepoTags']))
            logging.info('     Cmd %s', str(image_info['Config']['Cmd']))
            logging.info('  Labels %s', str(image_info['Config']['Labels']))
            docker_api.remove_image(image, force = True)
        except:
            logging.info(sys.exc_info()[1])

def main():
    # Don't cleanup in case of failure for debugging purpose.
    if jenkins_build_result == 'FAILURE':
        return

    # Only cleanup dangling if we are starting up (build result UNKNOWN)
    #
    # Only cleanup dangling if the pull request is closed by merging
    # since a push event will be coming so we want the build for the
    # push event to be able to reuse cached docker image layers. The
    # push event will do full cleanup after publish.
    #
    # On further testing, it appears that when one pull request is based
    # on another pull request, merging the first pull request can result
    # in the second pull request being merged automatically. And for the
    # second pull request, only a pull_request event with close action
    # and mergeable state setting to true will be sent, no push event.
    # So for a pull request event with close action, regardless of the
    # mergeable state, we will do full cleanup. Push event will also do
    # full cleanup.

    dangling = False if (jenkins_build_result != 'UNKNOWN' and
                         (onnx_mlir_pr_action == 'closed' or
                          onnx_mlir_pr_action == 'push')) else True

    logging.info('Docker cleanup for pull request: #%s, ' +
                 'build result: %s, action: %s, merged: %s, dangling: %s',
                 onnx_mlir_pr_number,
                 jenkins_build_result,
                 onnx_mlir_pr_action,
                 onnx_mlir_pr_merged,
                 dangling)

    cleanup_docker_images(onnx_mlir_pr_number, dangling)

if __name__ == "__main__":
    main()

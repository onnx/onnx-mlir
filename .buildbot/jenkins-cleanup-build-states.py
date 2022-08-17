#!/usr/bin/env python3

import docker
import glob
import logging
import os
import sys

logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S')

docker_daemon_socket      = os.getenv('DOCKER_DAEMON_SOCKET')
docker_registry_host_name = os.getenv('DOCKER_REGISTRY_HOST_NAME')
docker_registry_user_name = os.getenv('DOCKER_REGISTRY_USER_NAME')
github_repo_name          = os.getenv('GITHUB_REPO_NAME')
github_repo_name2         = os.getenv('GITHUB_REPO_NAME').replace('-', '_')
github_pr_baseref         = os.getenv('GITHUB_PR_BASEREF')
github_pr_baseref2        = os.getenv('GITHUB_PR_BASEREF').lower()
github_pr_number          = os.getenv('GITHUB_PR_NUMBER')
github_pr_number2         = os.getenv('GITHUB_PR_NUMBER2')
github_pr_action          = os.getenv('GITHUB_PR_ACTION')
github_pr_merged          = os.getenv('GITHUB_PR_MERGED')
jenkins_build_result      = os.getenv('JENKINS_BUILD_RESULT')

IMAGE_NAMES               = [ github_repo_name + '-llvm-static',
                              github_repo_name + '-llvm-shared',
                              github_repo_name + '-dev',
                              github_repo_name ]

LOG_PULL_PUSH             = ('pull request: #'
                             if github_pr_number == github_pr_number2 else
                             'merge branch: ')

docker_api                = docker.APIClient(base_url=docker_daemon_socket)

# Remove all the containers depending on an (dangling) image.
def remove_dependent_containers(image):
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
            logging.info('errors ignored while removing dependent containers')

# Cleanup docker images and containers associated with a pull request number.
# For action open/reopen/synchronize, only dangling images and containers are
# removed. For action close, non-dangling images and containers are removed.
def cleanup_docker_images(host_name,
                          user_name,
                          repo_name,
                          repo_name2,
                          pr_baseref,
                          pr_baseref2,
                          pr_number,
                          pr_number2,
                          dangling):
    # First find all the dangling docker images associated with the
    # pull request number
    filters = { 'dangling': True,
                'label': [ repo_name2 + '_pr_number=' + pr_number ] }
    images = docker_api.images(filters = filters, quiet = True)

    # When a build is aborted the cleanup may try to remove an intermediate
    # image or container that the docker build process itself is already doing,
    # resulting a conflict. So we catch the exception and ignore it.

    # For each dangling image found, find and remove all the dependant
    # containers
    for image in images:
        remove_dependent_containers(image)

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
        image_repo1 = ((host_name + '/' if host_name else '') +
                       (user_name + '/' if user_name else ''))
        image_tag = pr_number.lower()
        for image_name in IMAGE_NAMES:
            image_repo2 = (image_name +
                           ('.' + pr_baseref2 if pr_baseref != 'main' else ''))
            image_full = image_repo1 + image_repo2 + ':' + image_tag
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
                         (github_pr_action == 'closed' or
                          github_pr_action == 'push')) else True

    logging.info('Docker cleanup for ' + LOG_PULL_PUSH +
                 '%s, build result: %s, action: %s, merged: %s, dangling: %s',
                 github_pr_number,
                 jenkins_build_result,
                 github_pr_action,
                 github_pr_merged,
                 dangling)

    cleanup_docker_images(docker_registry_host_name,
                          docker_registry_user_name,
                          github_repo_name,   # original docker repo name
                          github_repo_name2,  # docker repo name with '-' replaced by '_'
                                              # for use in image label
                          github_pr_baseref,  # original github base branch name
                          github_pr_baseref2, # lower case base branch name
                                              # for use in image name
                          github_pr_number,   # github pull request number or merge branch
                                              # which can be alphanumeric (e.g. main)
                          github_pr_number2,  # original github pull request number
                                              # which is numeric only
                          dangling)

if __name__ == "__main__":
    main()

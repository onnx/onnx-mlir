#!/usr/bin/env python3

import docker
import fasteners
import json
import logging
import os
import requests
import sys

logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S')

cpu_arch                    = os.getenv('CPU_ARCH')
docker_pushpull_rwlock      = os.getenv('DOCKER_PUSHPULL_RWLOCK')
docker_daemon_socket        = os.getenv('DOCKER_DAEMON_SOCKET')
docker_registry_host_name   = os.getenv('DOCKER_REGISTRY_HOST_NAME')
docker_registry_user_name   = os.getenv('DOCKER_REGISTRY_USER_NAME')
docker_registry_login_name  = os.getenv('DOCKER_REGISTRY_LOGIN_NAME')
docker_registry_login_token = os.getenv('DOCKER_REGISTRY_LOGIN_TOKEN')
github_repo_access_token    = os.getenv('GITHUB_REPO_ACCESS_TOKEN')
github_repo_name            = os.getenv('GITHUB_REPO_NAME')
github_repo_name2           = os.getenv('GITHUB_REPO_NAME').replace('-', '_')
github_pr_baseref           = os.getenv('GITHUB_PR_BASEREF')
github_pr_baseref2          = os.getenv('GITHUB_PR_BASEREF').lower()
github_pr_number            = os.getenv('GITHUB_PR_NUMBER')
github_pr_phrase            = os.getenv('GITHUB_PR_PHRASE')
github_pr_request_url       = os.getenv('GITHUB_PR_REQUEST_URL')

# dot can be used in docker image name
docker_static_image_name    = (github_repo_name + '-llvm-static' +
                               ('.' + github_pr_baseref2
                                if github_pr_baseref != 'main' else ''))
docker_shared_image_name    = (github_repo_name + '-llvm-shared' +
                               ('.' + github_pr_baseref2
                                if github_pr_baseref != 'main' else ''))
docker_dev_image_name       = (github_repo_name + '-dev' +
                               ('.' + github_pr_baseref2
                                if github_pr_baseref != 'main' else ''))
docker_usr_image_name       = (github_repo_name +
                               ('.' + github_pr_baseref2
                                if github_pr_baseref != 'main' else ''))

# dot cannot be used in python dict key so we use dash
python_static_image_name    = docker_static_image_name.replace('.', '-')
python_shared_image_name    = docker_shared_image_name.replace('.', '-')
python_dev_image_name       = docker_dev_image_name.replace('.', '-')
python_usr_image_name       = docker_usr_image_name.replace('.', '-')

LLVM_PROJECT_LABELS         = [ 'llvm_project_sha1',
                                'llvm_project_sha1_date',
                                'llvm_project_dockerfile_sha1']
PROJECT_LABELS              = [ github_repo_name2 + '_sha1',
                                github_repo_name2 + '_sha1_date',
                                github_repo_name2 + '_dockerfile_sha1' ]
DOCKER_IMAGE_NAME           = { 'static': docker_static_image_name,
                                'shared': docker_shared_image_name,
                                'dev':    docker_dev_image_name,
                                'usr':    docker_usr_image_name }
PYTHON_IMAGE_NAME           = { 'static': python_static_image_name,
                                'shared': python_shared_image_name,
                                'dev':    python_dev_image_name,
                                'usr':    python_usr_image_name }
IMAGE_TAG                   = github_pr_number.lower()
IMAGE_LABELS                = { python_static_image_name: LLVM_PROJECT_LABELS,
                                python_shared_image_name: LLVM_PROJECT_LABELS,
                                python_dev_image_name:    PROJECT_LABELS,
                                python_usr_image_name:    PROJECT_LABELS }
IMAGE_ARCHS                 = { 's390x', 'amd64', 'ppc64le' }
commit_sha1_date_label      = { python_static_image_name: 'llvm_project_sha1_date',
                                python_shared_image_name: 'llvm_project_sha1_date',
                                python_dev_image_name:    github_repo_name2 + '_sha1_date',
                                python_usr_image_name:    github_repo_name2 + '_sha1_date' }
dockerfile_sha1_label       = { python_static_image_name: 'llvm_project_dockerfile_sha1',
                                python_shared_image_name: 'llvm_project_dockerfile_sha1',
                                python_dev_image_name:    github_repo_name2 + '_dockerfile_sha1',
                                python_usr_image_name:    github_repo_name2 + '_dockerfile_sha1' }
pr_mergeable_state          = {
    'behind':    { 'mergeable': False,
                   'desc': 'the head ref is out of date' },
    # see comments in image_publishable
    'blocked':   { 'mergeable': True,
                   'desc': 'the merge is blocked' },
    'clean':     { 'mergeable': True,
                   'desc': 'mergeable and passing commit status' },
    'dirty':     { 'mergeable': False,
                   'desc': 'the merge commit cannot be cleanly created' },
    'draft':     { 'mergeable': False,
                   'desc': 'the merge is blocked due to the pull request being a draft' },
    'has_hooks': { 'mergeable': True,
                   'desc': 'mergeable with passing commit status and pre-receive hooks' },
    'unknown':   { 'mergeable': True,
                   'desc': 'the state cannot currently be determined' },
    'unstable':  { 'mergeable': True,
                   'desc': 'mergeable with non-passing commit status' } }

DOCKER_DIST_MANIFESTS       = {
    'v1': 'application/vnd.docker.distribution.manifest.v1+json',
    'v2': 'application/vnd.docker.distribution.manifest.v2+json' }
DOCKER_DIST_MANIFEST_LIST   = 'application/vnd.docker.distribution.manifest.list.v2+json'

docker_rwlock               = fasteners.InterProcessReaderWriterLock(docker_pushpull_rwlock)
docker_api                  = docker.APIClient(base_url=docker_daemon_socket)

# Get the labels of a local docker image, raise exception
# if image doesn't exist or has invalid labels.
def get_local_image_labels(host_name, user_name, image_name, image_tag, image_labels):
    image_full = ((host_name + '/' if host_name else '') +
                  (user_name + '/' if user_name else '') +
                  image_name + ':' + image_tag)
    info = docker_api.inspect_image(image_full)
    logging.info('local image %s labels: %s', image_full, info['Config']['Labels'])
    labels = info['Config']['Labels']
    if (labels):
        labels_ok = True
        for label in image_labels:
            if not labels[label]:
                labels_ok = False
                break
        if labels_ok:
            return labels
    raise Exception('local image ' + image_full +
                    ' does not exist or has invalid labels')

# Make REST call to get the v1 or v2 manifest of an image from
# private docker registry
def get_image_manifest_private(host_name, user_name, image_name, image_tag,
                               schema_version, login_name, login_token):
    resp = requests.get(
        url = ('https://' + host_name + '/v2/' +
               (user_name + '/' if user_name else '') +
               image_name + '/manifests/' + image_tag),
        headers = { 'Accept': DOCKER_DIST_MANIFESTS[schema_version] },
        auth = (login_name, login_token))
    resp.raise_for_status()
    return resp

# Make REST call to put multiarch manfiest list of an image in
# private docker registry
def put_image_manifest_private(host_name, user_name, image_name, image_tag,
                               manifest_list, login_name, login_token):
    resp = requests.put(
        url = ('https://' + host_name + '/v2/' +
               (user_name + '/' if user_name else '') +
               image_name + '/manifests/' + image_tag),
        headers = { 'Content-Type': DOCKER_DIST_MANIFEST_LIST },
        auth = (login_name, login_token),
        json = { 'schemaVersion': 2,
                 'mediaType': DOCKER_DIST_MANIFEST_LIST,
                 'manifests': manifest_list })
    resp.raise_for_status()
    return resp

# Make REST call to get the access token to operate on an image in
# public docker registry
def get_access_token(user_name, image_name, action, login_name, login_token):
    resp = requests.get(
        url = ('https://auth.docker.io/token' +
               '?service=registry.docker.io' +
               '&scope=repository:' +
               (user_name + '/' if user_name else '') + image_name + ':'+ action),
        auth = (login_name, login_token))
    resp.raise_for_status()
    return resp.json()['token']

# Make REST call to get the v1 or v2 manifest of an image from
# public docker registry
def get_image_manifest_public(user_name, image_name, image_tag,
                              schema_version, login_name, login_token, access_token = None):
    # Get access token if not passed in
    if not access_token:
        access_token = get_access_token(
            user_name, image_name, 'pull', login_name, login_token)
    # Get manifest
    resp = requests.get(
        url = ('https://registry-1.docker.io/v2/' +
               (user_name + '/' if user_name else '') +
               image_name + '/manifests/' + image_tag),
        headers={ 'Accept': DOCKER_DIST_MANIFESTS[schema_version],
                  'Authorization': 'Bearer ' + access_token })
    resp.raise_for_status()
    return resp

# Make REST call to put multiarch manfiest list of an image in
# private docker registry
def put_image_manifest_public(user_name, image_name, image_tag,
                              manifest_list, login_name, login_token, access_token = None):
    # Get access token if not passed in
    if not access_token:
        access_token = get_access_token(
            user_name, image_name, 'push', login_name, login_token)
    # Put manifest
    resp = requests.put(
        url = ('https://registry-1.docker.io/v2/' +
               (user_name + '/' if user_name else '') +
               image_name + '/manifests/' + image_tag),
        headers = { 'Content-Type': DOCKER_DIST_MANIFEST_LIST,
                    'Authorization': 'Bearer ' + access_token },
        json = { 'schemaVersion': 2,
                 'mediaType': DOCKER_DIST_MANIFEST_LIST,
                 'manifests': manifest_list })
    resp.raise_for_status()
    return resp

# Get the labels of a docker image in the docker registry.
# python docker SDK does not support this so we have to make
# our own REST calls.
def get_remote_image_labels(host_name, user_name, image_name, image_tag,
                            image_labels, login_name, login_token):
    try:
        # Get manifest, only v1 schema has labels so accept v1 only
        resp = (
            # private docker registry
            get_image_manifest_private(host_name, user_name, image_name, image_tag,
                                       'v1', login_name, login_token)
            if host_name else
            # public docker registry
            get_image_manifest_public(user_name, image_name, image_tag,
                                      'v1', login_name, login_token))

        image_full = ((host_name + '/' if host_name else '') +
                      (user_name + '/' if user_name else '') +
                      image_name + ':' + image_tag)

        # v1Compatibility is a quoted JSON string, not a JSON object
        manifest = json.loads(resp.json()['history'][0]['v1Compatibility'])
        logging.info('remote image %s labels: %s',
                     image_full, manifest['config']['Labels'])
        labels = manifest['config']['Labels']
        if (labels):
            labels_ok = True
            for label in image_labels:
                if not labels[label]:
                    labels_ok = False
                    break
            if labels_ok:
                return labels
        raise Exception('remote image ' + image_full +
                        ' does not exist or has invalid labels')
    except:
        logging.info(sys.exc_info()[1])
        return ''

# Post a comment on the pull request issue page when the pull request
# source is outdated and publish is rejected.
def post_pr_comment(url, msg, token):
    try:
        resp = requests.post(
            url = url,
            headers = { 'Accept': 'application/json',
                        'Authorization': 'token ' + token },
            data = { 'body': msg })
        resp.raise_for_status()
        logging.info(
            '{ "url": "%s", "created_at": "%s", ' + '"updated_at": "%s", "body": "%s" }',
            resp.json()['url'],
            resp.json()['created_at'],
            resp.json()['updated_at'],
            resp.json()['body'])
    except:
        logging.info(sys.exc_info()[1])

# Get pull request source mergeable state
def get_pr_mergeable_state(url, token):
    try:
        resp = requests.get(
            url = url,
            headers = { 'Accept': 'application/json',
                        'Authorization': 'token ' + token })
        resp.raise_for_status()
        return resp.json()['mergeable_state']
    except:
        logging.info(sys.exc_info()[1])
        return 'unknown'

# Decide whether we should publish the local images or not
def image_publishable(host_name, user_name,
                      docker_image_name, python_image_name, image_tag,
                      image_labels, login_name, login_token):

    # If local image is missing or has invalid labels, exception
    # will be raised to fail the build.
    local_labels = get_local_image_labels(
        host_name, user_name, docker_image_name, image_tag, image_labels)
    remote_labels = get_remote_image_labels(
        host_name, user_name, docker_image_name, cpu_arch, image_labels, login_name, login_token)

    # If url is 'none', it's a push event from merging so skip
    # mergeable state check.
    #
    # Note that when our (and/or some other) build is marked as required,
    # while the build(s) are ongoing, the mergeable state will be "blocked".
    # So for publish triggered by "publish this please" phrase, we have a
    # catch 22 problem. But if we can come to this point, we know that at
    # least our build successfully built the docker images. So we allow
    # the blocked mergeable state to publish our images.
    if github_pr_request_url != 'none':
        state = get_pr_mergeable_state(github_pr_request_url, github_repo_access_token)
        logging.info('pull request url: %s, mergeable state: %s, %s',
                     github_pr_request_url, state, pr_mergeable_state[state]['desc'])
        if not pr_mergeable_state[state]['mergeable']:
            raise Exception('publish aborted due to unmergeable state')
    if not remote_labels:
        logging.info('publish due to invalid remote labels')
        return True
    if github_pr_phrase == 'publish':
        logging.info('publish forced by trigger phrase')
        return True
    if (local_labels[commit_sha1_date_label[python_image_name]] >
        remote_labels[commit_sha1_date_label[python_image_name]]):
        logging.info('publish due to newer local sha1 date')
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
    if (local_labels[commit_sha1_date_label[python_image_name]] ==
        remote_labels[commit_sha1_date_label[python_image_name]] and
        local_labels[dockerfile_sha1_label[python_image_name]] !=
        remote_labels[dockerfile_sha1_label[python_image_name]]):
        logging.info('publish due to different dockerfile sha1')
        return True

    logging.info('publish skipped due to older or identical local image')
    return False

def publish_arch_image(host_name, user_name, image_name, image_tag,
                       login_name, login_token):
    image_repo  = ((host_name + '/' if host_name else '') +
                   (user_name + '/' if user_name else '') + image_name)
    image_pr    = image_repo + ':' + image_tag
    image_arch  = image_repo + ':' + cpu_arch

    # Acquire write lock to prepare for tagging to the arch image and
    # pushing it. This is to serialize against other PR merges trying
    # to push (writers) and/or other PR builds trying to pull (readers)
    # the arch image.
    logging.info('acquiring write lock for tagging and pushing %s', image_arch)
    docker_rwlock.acquire_write_lock()
    try:
        # Tag the image with arch
        logging.info('tagging %s -> %s', image_pr, image_arch)
        docker_api.tag(image_pr, image_repo, cpu_arch)

        # Push the image tagged with arch then remove it, regardless of
        # whether the push worked or not.
        logging.info('pushing %s', image_arch)
        for line in docker_api.push(repository = image_repo,
                                    tag = cpu_arch,
                                    auth_config = { 'username': login_name,
                                                    'password': login_token },
                                    stream = True, decode = True):
            print((line['id']+': '
                   if 'id' in line and 'progress' not in line else '') +
                  (line['status'] + '\n'
                   if 'status' in line and 'progress' not in line else ''),
                  end = '', flush = True)
    # Remove arch image and release lock regardless of exception or not
    finally:
        docker_api.remove_image(image_arch, force = True)
        docker_rwlock.release_write_lock()
        logging.info('released write lock for tagging and pushing %s', image_arch)

# Publish multiarch manifest for an image
def publish_multiarch_manifest(host_name, user_name, image_name, manifest_tag,
                               login_name, login_token):
    try:
        if not host_name:
            access_token = get_access_token(user_name, image_name,
                                            'pull,push', login_name, login_token)

        # For each arch, construct the manifest element needed for the
        # manifest list by extracting fields from v1 and v2 image manifests.
        # We get platform from v1 image manifest, and mediaType, size, and
        # digest from v2 image manifest.
        manifest_list = []
        for image_tag in IMAGE_ARCHS:
            m = {}
            resp = (
                get_image_manifest_private(host_name, user_name, image_name, image_tag,
                                           'v2', login_name, login_token)
                if host_name else
                get_image_manifest_public(user_name, image_name, image_tag,
                                          'v2', login_name, login_token, access_token))
            m['mediaType'] = resp.headers['Content-Type']
            m['size'] = len(resp.text)
            m['digest'] = resp.headers['Docker-Content-Digest']

            resp = (
                get_image_manifest_private(host_name, user_name, image_name, image_tag,
                                           'v1', login_name, login_token)
                if host_name else
                get_image_manifest_public(user_name, image_name, image_tag,
                                          'v1', login_name, login_token, access_token))
            m['platform'] = {}
            v1Compatibility = json.loads(resp.json()['history'][0]['v1Compatibility'])
            m['platform']['architecture'] = v1Compatibility['architecture']
            m['platform']['os'] = v1Compatibility['os']

            manifest_list.append(m)

        # Make the REST call to PUT the multiarch manifest list.
        resp = (
            put_image_manifest_private(
                host_name, user_name, image_name, manifest_tag,
                manifest_list, login_name, login_token)
            if host_name else
            put_image_manifest_public(
                user_name, image_name, manifest_tag,
                manifest_list, login_name, login_token, access_token))

        logging.info('publish %s/%s:%s', user_name, image_name, manifest_tag)
        logging.info('        %s', resp.headers['Docker-Content-Digest'])
    except:
        logging.info(sys.exc_info()[1])

# Publish an image if it should be published and publish multiarch manifest
# for developer and user images if necessary.
def publish_image(image_type):

    host_name         = docker_registry_host_name
    user_name         = docker_registry_user_name
    login_name        = docker_registry_login_name
    login_token       = docker_registry_login_token

    docker_image_name = DOCKER_IMAGE_NAME[image_type]
    python_image_name = PYTHON_IMAGE_NAME[image_type]
    image_tag         = IMAGE_TAG
    image_labels      = IMAGE_LABELS[python_image_name]

    # Decide if the image should be published or not
    if not image_publishable(host_name, user_name,
                             docker_image_name, python_image_name, image_tag,
                             image_labels, login_name, login_token):
        return

    # Publish the arch specific image
    publish_arch_image(host_name, user_name, docker_image_name, image_tag,
                       login_name, login_token)

    # For developer and user images, we publish a multiarch manifest so we can
    # pull the images without having to explicitly specify the arch tag.
    if image_type == 'dev' or image_type == 'usr':
        publish_multiarch_manifest(host_name, user_name, docker_image_name, 'latest',
                                   login_name, login_token)

def main():
    for image_type in [ 'static', 'shared', 'dev', 'usr' ]:
        publish_image(image_type)

if __name__ == "__main__":
    main()

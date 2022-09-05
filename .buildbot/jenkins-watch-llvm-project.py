#!/usr/bin/env python3

import copy
import docker
import git
import jenkins
import json
import logging
import math
import os
import platform
import re
import requests
import shutil
import sys
import traceback

from datetime import datetime
from longest_increasing_subsequence import longest_decreasing_subsequence

logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S')

# Set parallel jobs based on both CPU count and memory size.
# Because using CPU count alone can result in out of memory
# and get Jenkins killed. For example, we may have 64 CPUs
# (128 threads) and only 32GB memory. So spawning off 128
# cc/c++ processes is going to quickly exhaust the memory.
#
# Algorithm: NPROC = min(2, # of CPUs) if memory < 8GB, otherwise
#            NPROC = min(memory / 4, # of CPUs)
MEMORY_IN_GB                = (os.sysconf('SC_PAGE_SIZE') *
                               os.sysconf('SC_PHYS_PAGES') / (1024.**3))
NPROC                       = str(math.ceil(min(max(2, MEMORY_IN_GB/4), os.cpu_count())))

CPU_ARCH                    = platform.uname().machine.replace('x86_64', 'amd64')
EPOCH0                      = datetime.utcfromtimestamp(0).isoformat() + 'Z'

DOCKER_DAEMON_SOCKET        = 'unix://var/run/docker.sock'

JSON_DUMPS_INDENT           = 2

docker_api                  = docker.APIClient(base_url=DOCKER_DAEMON_SOCKET)

jenkins_home                = os.getenv('JENKINS_HOME')
job_name                    = os.getenv('JOB_NAME')
build_number                = os.getenv('BUILD_NUMBER')
workspace_dir               = os.getenv('WORKSPACE')
job_dir                     = os.path.join(jenkins_home, 'jobs', job_name)
publish_dir                 = os.path.join(job_dir, 'htmlreports', 'LLVM_20Watch_20Report')
report_dir                  = os.path.join(workspace_dir, 'llvm_watch_report')

jenkins_rest_api_user       = os.getenv('JENKINS_USER', 'jenkins')
jenkins_rest_api_url        = { 's390x':   'http://localhost:8080/jenkins',
                                'amd64':   'http://localhost:8080/jenkinx',
                                'ppc64le': 'http://localhost:8080/jenkinp' }[CPU_ARCH]

github_repo_access_token    = os.getenv('GITHUB_REPO_ACCESS_TOKEN')
jenkins_rest_api_token      = os.getenv('JENKINS_REST_API_TOKEN')

LLVM_PROJECT_BASE_IMAGE     = 'ubuntu:focal'
LLVM_PROJECT_WATCH_IMAGE    = 'llvm-project-watch'
LLVM_PROJECT_GITHUB_URL     = 'https://api.github.com/repos/llvm/llvm-project'
LLVM_PROJECT_SHA1_REGEX     = 'git checkout ([0-9a-f]+)'
LLVM_PROJECT_DOCKERFILE     = os.path.join(workspace_dir,
                                           'docker', 'Dockerfile.llvm-project')
LLVM_PROJECT_SHA1_FILE      = os.path.join(workspace_dir, 'utils', 'clone-mlir.sh')

LLVM_PROJECT_WATCH_STATE    = 'llvm-watch-state.json'
LLVM_PROJECT_WATCH_LOGDATA  = 'llvm-watch-logdata.js'
LLVM_PROJECT_WATCH_HTML     = 'llvm-watch.html'

ONNX_MLIR_BASE_IMAGE        = LLVM_PROJECT_WATCH_IMAGE + ':' + build_number
ONNX_MLIR_WATCH_IMAGE       = 'onnx-mlir-watch'
ONNX_MLIR_DOCKERFILE        = os.path.join(workspace_dir,
                                           'docker', 'Dockerfile.onnx-mlir-dev')

ONNX_MLIR_JOB_NAME          = 'ONNX-MLIR-Pipeline-Docker-Build'
LLVM_WATCH_JOB_NAME         = 'LLVM-Watch-Docker-Build'

AMCHARTS_URL                = 'https://cdn.amcharts.com/lib/5/'
AMCHARTS_THEMES_URL         = AMCHARTS_URL + 'themes/'
AMCHARTS_INDEX_JS           = 'index.js'
AMCHARTS_XY_JS              = 'xy.js'
AMCHARTS_ANIMATED_JS        = 'Animated.js'
AMCHARTS_PREFIX             = 'amcharts-'

INIT_WATCH_STATE            = {
    'converged': True,
    'recent':    {
        'failed':    [ { 'sha1': '', 'date': EPOCH0, 'stat': {}, 'mesg': '' }, '' ],
        'succeeded': [ { 'sha1': '', 'date': EPOCH0, 'stat': {}, 'mesg': '' }, '' ] },
    'build_history': [ { 'head':   { 'sha1': '', 'date': EPOCH0, 'stat': {}, 'mesg': '' },
                         'middle': { 'sha1': '', 'date': EPOCH0, 'stat': {}, 'mesg': '' },
                         'tail':   { 'sha1': '', 'date': EPOCH0, 'stat': {}, 'mesg': '' },
                         'size':   0,
                         'status': False,
                         'build':  [] } ],
    'llvm_history_github': { 'index': {}, 'history': [] },
    'llvm_history':        { 'index': {}, 'history': [] },
    'commits_dropped':     [] }

# Download remote URL and save to local file
def urlretrieve(remote_url, local_file):
    req = requests.get(remote_url)
    with open(local_file, 'wb') as f:
        f.write(req.content)

# Extract text matching regex from file
def extract_pattern_from_file(file_name, regex_pattern):
    try:
        for line in open(file_name):
            matched = re.search(re.compile(regex_pattern), line)
            if matched:
                return matched.group(1)
    except:
        return ''

# Get the author commit date of a commit sha
def get_repo_sha1_date(github_repo, access_token, commit_sha1):
    try:
        resp = requests.get(
            url = github_repo + '/commits/' + commit_sha1,
            headers = { 'Accept': 'application/json',
                        'Authorization': 'token ' + access_token
            })
        resp.raise_for_status()
        return resp.json()['commit']['committer']['date']
    except:
        logging.info(sys.exc_info()[1])
        return ''

# Retrieve llvm-project main branch commit history from the latest
# until specified commit_sha1.
def get_remote_repo_sha1_history(github_repo, access_token,
                                 commit_sha1_oldest, commit_sha1_newest=None):

    logging.info('Fetch LLVM commit history until ' + commit_sha1_oldest)
    sha1_history = []
    date_history = []
    stat_history = []
    mesg_history = []
    hist = { 'index': {}, 'history': [] }
    try:
        # Keep retrieving if we don't see commit_sha1 in sha1_history
        while not (commit_sha1_oldest in sha1_history):
            # Retrieve 100 commits at a time and keep going backwards
            # from the last commit we retrieved the last time.
            resp = requests.get(
                url = (github_repo + '/commits?per_page=100' +
                       ('&sha=' + sha1_history[-1] if sha1_history else '')),
                headers = { 'Accept': 'application/json',
                            'Authorization': 'token ' + access_token })
            resp.raise_for_status()

            # Go through the returned json array and add commit sha
            # to sha1_history. The first returned commit will be the
            # last commit in sha1_history we started from, so skip it.
            for commit in resp.json():
                if sha1_history and commit['sha'] == sha1_history[-1]:
                    continue
                sha1_history += [ commit['sha'] ]
                date_history += [ commit['commit']['committer']['date'] ]

                resp = requests.get(
                    url = github_repo + '/commits/' + commit['sha'],
                    headers = { 'Accept': 'application/json',
                            'Authorization': 'token ' + access_token })
                resp.raise_for_status()
                stat_history += [ resp.json()['stats'] ]
                mesg_history += [ resp.json()['commit']['message'] ]

        # Return the sublist from the beginning until commit_sha1_oldest
        # (not including commit_sha1 since we know onnx-mlir builds
        # against commit_sha1 fine).
        #
        # start index inclusive, end index exclusive
        index_newest = (sha1_history.index(commit_sha1_newest) if commit_sha1_newest else 0)
        index_oldest = sha1_history.index(commit_sha1_oldest)

        # Construct an array of
        #   [ { 'sha1': ..., 'date': ..., 'stat': ..., 'mesg': ... }, ... ]
        # And because we need to know the index of an item given the sha1,
        # we add sha1->index mapping to the dict.
        for (i, (sha1, date, stat, mesg)) in enumerate(
                zip(sha1_history[index_newest:index_oldest],
                    date_history[index_newest:index_oldest],
                    stat_history[index_newest:index_oldest],
                    mesg_history[index_newest:index_oldest])):
            hist['index'][sha1] = i
            hist['history'] += [ {
                'sha1': sha1, 'date': date, 'stat': stat, 'mesg': mesg } ]

        logging.info('{} new commits retrieved'.format(len(hist['history'])))
    except:
        logging.info(sys.exc_info()[1])
    finally:
        return hist

# Get the latest commit sha1 and date of a local git repo
def get_local_repo_sha1_date(local_repo):
    repo = git.Repo(local_repo)
    repo_sha1 = repo.head.commit.hexsha
    repo_sha1_date = datetime.utcfromtimestamp(
        repo.head.commit.committed_date).isoformat() + 'Z'
    return { 'sha1': repo_sha1, 'date': repo_sha1_date }

# Remove all the containers depending on an (dangling) image.
def remove_dependent_containers(image):
    containers = docker_api.containers(
        filters = { 'ancestor': image }, all=True, quiet=True)
    for container in containers:
        try:
            container_info = docker_api.inspect_container(container['Id'])
            logging.info('Removing     Id:%s', container['Id'])
            logging.info('   Image %s', container_info['Image'])
            logging.info('     Cmd %s', str(container_info['Config']['Cmd']))
            logging.info('  Labels %s', str(container_info['Config']['Labels']))
            docker_api.remove_container(container['Id'], v=True, force=True)
        except:
            logging.info(sys.exc_info()[1])
            logging.info('errors ignored while removing dependent containers')

# Remove all the images in the list
def remove_docker_images(images):
    for image in images:
        try:
            image_info = docker_api.inspect_image(image)
            logging.info('Removing %s', image)
            logging.info('RepoTags %s', str(image_info['RepoTags']))
            logging.info('     Cmd %s', str(image_info['Config']['Cmd']))
            logging.info('  Labels %s', str(image_info['Config']['Labels']))
            docker_api.remove_image(image, force = True)
        except:
            logging.info(sys.exc_info()[1])
    
# Remove all dangling images associated with a build
def remove_dangling_images(build):
    # Find all the danglig images associated with the build
    filters = { 'dangling': True,
                'label': [ 'onnx_mlir_pr_number=' + build ] }
    images = docker_api.images(filters = filters, quiet = True)

    # For each dangling image found, find and remove all the dependant
    # containers
    for image in images:
        remove_dependent_containers(image)

    # Remove all the images
    remove_docker_images(images)

# Remove recent failed or succeeded docker image, and if requested also
# reset the corresponding new watch_state key.
def remove_recent_image(curr_state, watch_state, key, reset=False):
    build = curr_state['recent'][key][1]
    if build:
        remove_docker_images([ LLVM_PROJECT_WATCH_IMAGE + ':' + build,
                               ONNX_MLIR_WATCH_IMAGE + ':' + build ])
    if reset:
        watch_state['recent'][key] = [
            { 'sha1': '', 'date': EPOCH0, 'stat': {}, 'mesg': '' }, '' ]

# Build watch image
def build_watch_image(repo, commit, dockerfile, base_image, image_repo, image_tag):
    image_full = image_repo + ':' + image_tag

    layer_sha256 = ''
    for line in docker_api.build(
            path = '.',
            dockerfile = dockerfile,
            tag = image_full,
            decode = True,
            rm = True,
            buildargs = {
                'BASE_IMAGE': base_image,
                'NPROC': NPROC,
                repo + '_SHA1': commit['sha1'],
                repo + '_SHA1_DATE': commit['date'],
                # set both the same to avoid git fetch --unshallow
                'ONNX_MLIR_PR_NUMBER': image_tag,
                'ONNX_MLIR_PR_NUMBER2': image_tag,
            }):

        if 'stream' in line:
            # Keep track of the latest successful image layer
            m = re.match('^\s*---> ([0-9a-f]+)$', line['stream'])
            if m:
                layer_sha256 = m.group(1)
            print(line['stream'], end='', flush=True)

        if 'error' in line:
            # Tag the latest successful image layer for easier debugging.
            if layer_sha256:
                image_layer = 'sha256:' + layer_sha256
                remove_dependent_containers(image_layer)
                logging.info('tagging %s -> %s for debugging', image_layer, image_full)
                docker_api.tag(image_layer, image_repo, image_tag, force=True)
            else:
                logging.info('no successful image layer for tagging')
            return False

    id = docker_api.images(name=image_full, all=False, quiet=True)
    logging.info('image %s (%s) built', image_full, id[0][0:19])

    return True

# Check if an active ONNX-MLIR build or a previous LLVM watch build is running
def check_running_job():
    jenkins_server = jenkins.Jenkins(url = jenkins_rest_api_url,
                                     username = jenkins_rest_api_user,
                                     password = jenkins_rest_api_token)
    running_builds = jenkins_server.get_running_builds()
    for build in running_builds:
        # An active build is found in ONNX-MLIR-Pipeline-Docker-Build, or
        # a previous build in LLVM-Watch-Docker-Build is still running.
        if (build['name'] == ONNX_MLIR_JOB_NAME or
            (build['name'] == LLVM_WATCH_JOB_NAME and
             build['number'] < int(build_number))):
            logging.info('Active build(s) find in {}, skip this run'.format(
                build['name']))
            return True

    return False

# Create index for commit history using sha1 as the index key
def index_commit_history(commit_history):
    indexed_commit_history = { 'index': {}, 'history': [] }
    for (i, hist) in enumerate(commit_history):
        indexed_commit_history['index'][hist['sha1']] = i
        indexed_commit_history['history'] += [ hist ]

    return indexed_commit_history

# Set new range to search
#
# head:        new head
# head_adjust: +1/-1 to head depending on situation
# tail:        new tail
# tail_adjust: +1/-1 to tail depending on situation
# history:     commit history between [head_index, tail_index]
#
def set_range(head, head_adjust, tail, tail_adjust, history):
    history_head = history['history'][0]
    history_tail = history['history'][-1]

    head_index = history['index'][head['sha1']] + head_adjust
    tail_index = history['index'][tail['sha1']] + tail_adjust

    # no commit left in the list
    if head_index > tail_index:
        return {}, {}, { 'index': {}, 'history': [] }

    next_head = history['history'][head_index]
    next_tail = history['history'][tail_index]
    next_history = index_commit_history(history['history'][head_index:tail_index+1])

    return next_head, next_tail, next_history

# Write watch state and log data, and also generate all the files
# necessary for the HTML report.
def write_watch_files(curr_state, watch_state, next_history):
    # Write global watch state
    with open(os.path.join(job_dir, LLVM_PROJECT_WATCH_STATE), 'w') as state:
        json.dump(watch_state, state)

    # Write per build watch log data
    with open(os.path.join(report_dir, LLVM_PROJECT_WATCH_LOGDATA), 'w') as logdata:
        logdata.write('var logdata = ' +
                      json.dumps({ 'curr_state':   curr_state,
                                   'next_state':   watch_state,
                                   'next_history': next_history },
                                 indent=JSON_DUMPS_INDENT))

    # Copy llvm-watch.html
    shutil.copy(os.path.join(workspace_dir, '.buildbot', LLVM_PROJECT_WATCH_HTML),
                os.path.join(report_dir, LLVM_PROJECT_WATCH_HTML))

    # Download amcharts
    urlretrieve(AMCHARTS_URL + AMCHARTS_INDEX_JS,
                os.path.join(report_dir, AMCHARTS_PREFIX + AMCHARTS_INDEX_JS))
    urlretrieve(AMCHARTS_URL + AMCHARTS_XY_JS,
                os.path.join(report_dir, AMCHARTS_PREFIX + AMCHARTS_XY_JS))
    urlretrieve(AMCHARTS_THEMES_URL + AMCHARTS_ANIMATED_JS,
                os.path.join(report_dir, AMCHARTS_PREFIX + AMCHARTS_ANIMATED_JS))

# Workhorse function to compute and build the next llvm-project commit
def compute_range_build_next():
    # Load previous build state json to decide how we should proceed
    try:
        with open(os.path.join(job_dir, LLVM_PROJECT_WATCH_STATE), 'r') as f:
            watch_state = json.load(f)
    except:
        watch_state  = INIT_WATCH_STATE

    # Copy watch_state as our current states before the next build
    curr_state = copy.deepcopy(watch_state)

    # logging.info('watch state:\n{}\nllvm range:\n{}'.format(
    #     json.dumps(watch_state, indent=JSON_DUMPS_INDENT),
    #     json.dumps([ { 'index': llvm_head_index, 'head': llvm_head },
    #                  { 'index': llvm_tail_index, 'tail': llvm_tail } ],
    #                indent=JSON_DUMPS_INDENT)))

    # Get current llvm-project commit onnx-mlir is built against.
    # This is the oldest commit known to be good.
    #
    # llvm_history does NOT include llvm_onnx_mlir.
    #
    #   llvm_history = [llvm_head, ..., llvm_onnx_mlir-1]
    #
    llvm_onnx_mlir = extract_pattern_from_file(LLVM_PROJECT_SHA1_FILE,
                                               LLVM_PROJECT_SHA1_REGEX)
    logging.info('ONNX-MLIR currently using LLVM commit {}'.format(
        llvm_onnx_mlir))

    # Instead of always retrieve all the way back until llvm_onnx_mlir,
    # we only need to retrieve back until the head of the previous
    # llvm_history. This saves us a lot of github calls since for each
    # sha1 we need to call github to get its stat and that typically
    # makes us exceed the the github REST API call limit (5000/hour).
    retrieve_until_sha1 = (watch_state['llvm_history_github']['history'][0]['sha1']
                           if watch_state['llvm_history_github']['history']
                           else llvm_onnx_mlir)

    new_llvm_history = get_remote_repo_sha1_history(LLVM_PROJECT_GITHUB_URL,
                                                    github_repo_access_token,
                                                    retrieve_until_sha1)

    # Merge newly retrieved llvm_history with watch_state['llvm_history_github']
    # then advance the tail of the history to the sha1 that just passes
    # llvm_onnx_mlir.
    #
    # Search for llvm_onnx_mlir in new_llvm_history['history'] to find its index.
    # If not found, set llvm_onnx_mlir_index to length of new_llvm_history['history'].
    new_llvm_history['history'] += watch_state['llvm_history_github']['history']
    new_llvm_history_len = len(new_llvm_history['history'])
    llvm_onnx_mlir_index = next((index for (index, hist) in
                                 enumerate(new_llvm_history['history'])
                                 if hist['sha1'] == llvm_onnx_mlir),
                                new_llvm_history_len)
    logging.info('commit {} index={}, history length={}'.format(
        llvm_onnx_mlir, llvm_onnx_mlir_index, new_llvm_history_len))

    # Advance tail of new_llvm_history just past llvm_onnx_mlir
    llvm_history_all = new_llvm_history['history'][0:llvm_onnx_mlir_index]

    # The bisect search algorithm depends on the assumption that
    # commit dates are monotonically increasing/decreasing.
    # Unfortunately, this is not exactly the case. There are
    # llvm-project commits that are "out-of-order". We could
    # get around the problem by assigning each commit a monotonically
    # increasing sequence number we maintain ourselves and
    # use the sequence number instead of date for comparison
    # when searching. But this adds unncessary complexities.
    #
    # So instead, we run the commit history through the
    # longest increasing/decreasing subseqence and drop those
    # commits that are "out-of-order". The number of dropped
    # commits should typically be small and shouldn't affect
    # our search too much.
    llvm_history_lds = longest_decreasing_subsequence(
        llvm_history_all, key = lambda x:x['date'])
    commits_dropped = [ x for x in llvm_history_all if x not in llvm_history_lds ]

    # llvm_history_github is the raw llvm-project commit history
    # from the github.
    #
    # llvm_history is the longest decreasing subsequence of
    # llvm_history_github with "out-of-order" commits dropped.
    llvm_history_github = index_commit_history(llvm_history_all)
    llvm_history = index_commit_history(llvm_history_lds)

    # llvm_history will be empty if llvm_onnx_mlir has advanced
    # all the way to the latest sha1, unlikely but can happen.
    if not llvm_history['history']:
        watch_state = INIT_WATCH_STATE
        watch_state['recent']['succeeded'] = [ new_llvm_history['history'][0], '0' ]
        write_watch_files(curr_state, watch_state, [])
        return

    # LLVM commits all the way back to llvm_onnx_mlir
    llvm_head = llvm_history['history'][0]
    llvm_tail = llvm_history['history'][-1]

    # Various fields from curr_state
    curr_converged       = curr_state['converged']
    curr_recent          = curr_state['recent']
    curr_failed          = curr_recent['failed'][0]
    curr_succeeded       = curr_recent['succeeded'][0]

    build_history        = curr_state['build_history']
    curr                 = build_history[-1]
    curr_head            = curr['head']
    curr_middle          = curr['middle']
    curr_tail            = curr['tail']
    curr_status          = curr['status']
    curr_build           = curr['build']

    #                        llvm_head
    #                            |
    #                            |
    #                  <---- llvm_tail (1)
    #                            |
    #      +---> curr_head       |
    #      |         |           |
    # next |         | <---- llvm_tail (2)
    #      |         |           |
    #      +---< curr_middle     |
    #           (curr_succeeded) |
    #                |           |
    #                | <---- llvm_tail (3)
    #                |           |
    #            curr_tail       |
    #                            |
    #                  <---- llvm_tail (4)
    #
    # curr_middle build is successful, we are about to traverse upwards
    #
    #   [curr_head, curr_middle)
    #
    # Note that if a failure was found before, curr_head will always be
    # set to the curr_failed.
    #
    if curr_status:
        # Case (1)
        #
        # If llvm_tail has moved above curr_head/failed, llvm_tail+1
        # (which is what onnx-mlir uses and is known to work) is >=
        # curr_failed. We know that onnx-mlir now works with everything
        # between [curr_failed, llvm_tail+1]. So we can start traverse
        # upwards again between [llvm_head, llvm_tail].
        #
        if llvm_tail['date'] > curr_head['date']:
            next_head, next_tail, next_history = set_range(
                llvm_head, 0, llvm_tail, 0, llvm_history)

            logging.info(('curr: {}, llvm_tail {} > curr_head/failed {}\n' +
                          'next range:\n{}').format(
                'success' if curr_status else 'failure',
                json.dumps(llvm_tail),
                json.dumps(curr_head),
                json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                         if next_head else -1),
                               'head':  next_head },
                             { 'index': (llvm_history['index'][next_tail['sha1']]
                                         if next_tail else -1),
                               'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

        # Case (2)
        #
        # If llvm_tail has moved above curr_middle, we can set our range to
        #
        #   [?, llvm_tail]
        #
        # For the head, if we haven't found a failure yet (curr_failed
        # doesn't exist), we can move head to llvm_head. However, if
        # we did have a failure, we cannot move above curr_failed.
        #
        elif llvm_tail['date'] > curr_middle['date']:
            # failure exists: (curr_failed, llvm_tail]
            if curr_failed['sha1']:
                next_head, next_tail, next_history = set_range(
                    curr_failed, 1, llvm_tail, 0, llvm_history)

                logging.info(('curr: {}, llvm_tail {} > curr_middle {}, w/ failure\n' +
                              'next range:\n{}').format(
                    'success' if curr_status else 'failure',
                    json.dumps(llvm_tail),
                    json.dumps(curr_middle),
                    json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                             if next_head else -1),
                                   'head':  next_head },
                                 { 'index': (llvm_history['index'][next_tail['sha1']]
                                             if next_tail else -1),
                                   'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

            # no failure yet: [llvm_head, llvm_tail]
            else:
                next_head, next_tail, next_history = set_range(
                    llvm_head, 0, llvm_tail, 0, llvm_history)

                logging.info(('curr: {}, llvm_tail {} > curr_middle {}, no failure\n' +
                              'next range:\n{}').format(
                    'success' if curr_status else 'failure',
                    json.dumps(llvm_tail),
                    json.dumps(curr_middle),
                    json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                             if next_head else -1),
                                   'head':  next_head },
                                 { 'index': (llvm_history['index'][next_tail['sha1']]
                                             if next_tail else -1),
                                   'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

        # Case (3) and (4)
        #
        # llvm_tail has not moved above curr_middle, we can set our range to
        #
        #  [?, curr_middle)
        #
        # For the head, it's the same as in case (2).
        else:
            # failure exists: (curr_failed, curr_middle)
            if curr_failed['sha1']:
                next_head, next_tail, next_history = set_range(
                    curr_failed, 1, curr_middle, -1, llvm_history)

                logging.info(('curr: {}, llvm_tail {} <= curr_middle {}, w/ failure\n' +
                              'next range:\n{}').format(
                    'success' if curr_status else 'failure',
                    json.dumps(llvm_tail),
                    json.dumps(curr_middle),
                    json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                             if next_head else -1),
                                   'head':  next_head },
                                 { 'index': (llvm_history['index'][next_tail['sha1']]
                                             if next_tail else -1),
                                   'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

            # no failure yet: [llvm_head, curr_middle)
            else:
                next_head, next_tail, next_history = set_range(
                    llvm_head, 0, curr_middle, -1, llvm_history)

                logging.info(('curr: {}, llvm_tail {} <= curr_middle {}, no failure\n' +
                              'next range:\n{}').format(
                    'success' if curr_status else 'failure',
                    json.dumps(llvm_tail),
                    json.dumps(curr_middle),
                    json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                             if next_head else -1),
                                   'head':  next_head },
                                 { 'index': (llvm_history['index'][next_tail['sha1']]
                                             if next_tail else -1),
                                   'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

        # After computing the new range, compute the next_middle we will build.
        # Note that the middle index is now within the new next_history starting
        # from 0, not the index within the llvm_history.
        next_size = len(next_history['history'])
        if next_size:
            next_middle = next_history['history'][(next_size-1) // 2]
            converged = False
            logging.info('converged: {}\nnext middle:\n{}'.format(
                converged,
                json.dumps({ 'index':  next_history['index'][next_middle['sha1']],
                             'middle': next_middle,
                             'size':   next_size }, indent=JSON_DUMPS_INDENT)))
        else:
            converged = True

    #                        llvm_head
    #                            |
    #                            |
    #                  <---- llvm_tail (1)
    #                            |
    #            curr_head       |
    #                |           |
    #                | <---- llvm_tail (2)
    #                |           |
    #      +---< curr_middle     |
    #      |    (curr_failed)    |
    #      |         |           |
    # next |         | <---- llvm_tail (3)
    #      |         |           |
    #      +---> curr_tail       |
    #                            |
    #                  <---- llvm_tail (4)
    #
    # curr_middle build is a failure, we are about to traverse downwards
    #
    #   (curr_middle, curr_tail]
    #
    # Note that if a failure was found before, curr_head will always be
    # set to the curr_failed.
    #
    else:
        # Case (1) and (2)
        #
        # If llvm_tail has moved above curr_middle/failed, llvm_tail+1
        # (which is what onnx-mlir uses and is known to work) is >=
        # curr_failed. We know that onnx-mlir now works with everything
        # between [curr_failed, llvm_tail+1]. So we can start traverse
        # upwards again between [llvm_head, llvm_tail].
        #
        if llvm_tail['date'] > curr_middle['date']:
            next_head, next_tail, next_history = set_range(
                llvm_head, 0, llvm_tail, 0, llvm_history)

            logging.info(('curr: {}, llvm_tail {} > curr_middle/failed {}\n' +
                          'next range:\n{}').format(
                'success' if curr_status else 'failure',
                json.dumps(llvm_tail),
                json.dumps(curr_middle),
                json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                         if next_head else -1),
                               'head':  next_head },
                             { 'index': (llvm_history['index'][next_tail['sha1']]
                                         if next_tail else -1),
                               'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

        # Case (3)
        #
        # llvm_tail has moved above curr_tail, we can set our range to
        #
        #  (curr_middle, llvm_tail]
        #
        elif llvm_tail['date'] > curr_tail['date']:
            next_head, next_tail, next_history = set_range(
                curr_middle, 1, llvm_tail, 0, llvm_history)

            logging.info(('curr: {}, llvm_tail {} > curr_tail {}\n' +
                          'next range:\n{}').format(
                'success' if curr_status else 'failure',
                json.dumps(llvm_tail),
                json.dumps(curr_tail),
                json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                         if next_head else -1),
                               'head':  next_head },
                             { 'index': (llvm_history['index'][next_tail['sha1']]
                                         if next_tail else -1),
                               'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

        # Case (4)
        #
        # llvm_tail has not moved above curr_tail, we can set our range to
        #
        #  (curr_middle, curr_tail]
        else:
            next_head, next_tail, next_history = set_range(
                curr_middle, 1, curr_tail, 0, llvm_history)

            logging.info(('curr: {}, llvm_tail {} <= curr_tail {}\n' +
                          'next range:\n{}').format(
                'success' if curr_status else 'failure',
                json.dumps(llvm_tail),
                json.dumps(curr_tail),
                json.dumps([ { 'index': (llvm_history['index'][next_head['sha1']]
                                         if next_head else -1),
                               'head':  next_head },
                             { 'index': (llvm_history['index'][next_tail['sha1']]
                                         if next_tail else -1),
                               'tail':  next_tail } ], indent=JSON_DUMPS_INDENT)))

        # After computing the new range, compute the next_middle we will build.
        # Note that the middle index is now within the new next_history starting
        # from 0, not the index within the llvm_history.
        next_size = len(next_history['history'])
        if next_size:
            next_middle = next_history['history'][next_size // 2]
            converged = False
            logging.info('converged: {}\nnext middle:\n{}'.format(
                converged,
                json.dumps({ 'index':  next_history['index'][next_middle['sha1']],
                             'middle': next_middle,
                             'size':   next_size }, indent=JSON_DUMPS_INDENT)))
        else:
            converged = True

    # If we converged, we don't add a new watch state but simply add
    # our build number to the last watch state history. Otherwise,
    # we try the next build and add the result to the watch state history.
    if converged:
        logging.info('Converged, nothing to do')

        if len(watch_state['build_history'][-1]['build']) < 2:
            watch_state['build_history'][-1]['build'] += [ build_number ]
        else:
            watch_state['build_history'][-1]['build'][-1] = build_number
    # We have not converged, build to check the next commit
    else:
        next_status = (build_watch_image('LLVM_PROJECT',
                                         next_middle,
                                         LLVM_PROJECT_DOCKERFILE,
                                         LLVM_PROJECT_BASE_IMAGE,
                                         LLVM_PROJECT_WATCH_IMAGE,
                                         build_number) and
                       build_watch_image('ONNX_MLIR',
                                         get_local_repo_sha1_date('.'),
                                         ONNX_MLIR_DOCKERFILE,
                                         ONNX_MLIR_BASE_IMAGE,
                                         ONNX_MLIR_WATCH_IMAGE,
                                         build_number))
        remove_dangling_images(build_number)

        # Build successful
        if next_status:
            # clean up previous succeeded image
            remove_recent_image(curr_state, watch_state, 'succeeded')
            watch_state['recent']['succeeded'] = [ next_middle, build_number ]

            # If converged changes from true to false, it means that
            # we are starting a new search cycle so we start a new
            # watch_state['recent']['failed'].
            if curr_converged:
                remove_recent_image(curr_state, watch_state, 'failed', reset=True)
        # Build failed
        else:
            # clean up previous failed image
            remove_recent_image(curr_state, watch_state, 'failed')
            watch_state['recent']['failed'] = [ next_middle, build_number ]

            # If converged changes from true to false, it means that
            # we are starting a new search cycle so we start a new
            # watch_state['recent']['succeeded'].
            if curr_converged:
                remove_recent_image(curr_state, watch_state, 'succeeded', reset=True)

        # If converged changes from true to false, it means that
        # we are starting a new search cycle so we start a new
        # watch_state['build_history'].
        hist = [ { 'head':   next_head,
                   'middle': next_middle,
                   'tail':   next_tail,
                   'size':   next_size,
                   'status': next_status,
                   'build':  [ build_number ] } ]
        if curr_converged:
            watch_state['build_history'] = hist
        else:
            watch_state['build_history'] += hist
                   
    # Update watch state and write watch history
    watch_state['converged'] = converged
    watch_state['llvm_history_github'] = llvm_history_github
    watch_state['llvm_history'] = llvm_history
    watch_state['commits_dropped'] = commits_dropped

    # logging.info('watch state:\n{}'.format(json.dumps(watch_state,
    #                                                   indent=JSON_DUMPS_INDENT)))

    # Generate watch state, log data, and HTML report files
    write_watch_files(curr_state, watch_state, next_history)

def main():
    os.makedirs(report_dir)
    try:
        if not check_running_job():
            compute_range_build_next()
        else:
            # Copy from publish_dir to report_dir in case we didn't run
            # due to active build in ONNX-MLIR-Pipeline-Docker-Build.
            shutil.copytree(publish_dir, report_dir, dirs_exist_ok=True)
    except:
        logging.info(traceback.format_exc())

        # Copy from publish_dir to report_dir in case compute_range_build_next
        # failed somehow.
        shutil.copytree(publish_dir, report_dir, dirs_exist_ok=True)

if __name__ == "__main__":
    main()

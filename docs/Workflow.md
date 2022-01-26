<!--- SPDX-License-Identifier: Apache-2.0 -->
# Contribution Guide

## Step 1: Fork ONNX-MLIR in the cloud

We strongly encourage contributors to work in their own forks of the ONNX-MLIR project.
Creating a new fork is easy.

1. Visit https://github.com/onnx/onnx-mlir.
2. Click `Fork` button (top right) to establish a fork.

For the rest of this guide, we assume that you have set `user` to match your github profile name:

```sh
user={your github profile name}
```

## Step 2: Setup MLIR 

### Setup using Docker

Follow the directions provided [here](Docker.md#building-onnx-mlir-in-a-docker-environment) to establish a docker image that uses your ONNX-MLIR fork.
Make sure to uncomment the additional commands in Step 5 of the `Dockerfile` listed there and replace the `<<GitID>>` string with your own GitHub ID.

If you are already using a docker image, you need to check if the MLIR version is correct. 
To do so, compare the most recent commit ID from a `git log` command with the specific branch version extracted by the `git checkout` command listed [here](BuildOnLinuxOSX.md#MLIR).

If your MLIR is not up to date, you have a choice. 
You may either recreate a new Docker image using the above directions. 
Or you may update the MLIR inside the docker by either reinstalling it or updating it with `git fetch`, `git merge`, and `git checkout` commands. 
Installation steps are  described in the [Linux](BuildOnLinuxOSX.md#MLIR) building directions. 
You may elect to remove the entire LLVM `build` directory to rebuild from scratch.

###  Setup without Docker

Define a local working directory:

```sh
working_dir={your working directory}
```

Then follow the directions in this section of the top level [README](../README.md)
and OS specific instructions [Linux](BuildOnLinuxOSX.md#MLIR) or [Windows](BuildOnWindows.md#MLIR) for installing the currently supported MLIR version in your working directory.

If you already have an MLIR copy in your working directory, you should ensure that you have the latest copy.
To do so, compare the most recent commit ID from a `git log` command with the specific branch version extracted by the `git checkout` command listed [here](BuildOnLinuxOSX.md#MLIR). 
If your MLIR in not up to date, you must  bring it up to the correct commit level by either reinstalling it or updating it with `git fetch`, `git merge`, and `git checkout` commands.

## Step 3: Create your clone of ONNX-MLIR

If you are working with a Docker image, skip this step as it was taken care of by the Dockerfile script. 
Otherwise, follow the instructions below to create your ONNX-MLIR clone.

```sh
mkdir -p $working_dir
cd $working_dir
git clone --recursive https://github.com/$user/onnx-mlir.git
# the following is recommended
# or: git clone --recursive git@github.com:$user/onnx-mlir.git

cd $working_dir/onnx-mlir
git remote add upstream https://github.com/onnx-mlir/onnx-mlir.git
# or: git remote add upstream git@github.com:onnx-mlir/onnx-mlir.git

# Never push to upstream main since you do not have write access.
git remote set-url --push upstream no_push

# Confirm that your remotes make sense:
# It should look like:
# origin    https://github.com/$user/onnx-mlir.git (fetch)
# origin    https://github.com/$user/onnx-mlir.git (push)
# upstream  https://github.com/onnx-mlir/onnx-mlir.git (fetch)
# upstream  no_push (push)
git remote -v
```

## Step 4: Create a branch for your changes

Get your local main up to date:

```sh
cd $working_dir/onnx-mlir
git fetch upstream main
git merge upstream/main
```

Branch from main:

```sh
git checkout -b myfeature
```

## Step 5: Develop

### Edit the code

You can now edit the code on the `myfeature` branch.

### Run cmake & make

Follow the directions to build ONNX-MLIR for the OS that you are using [Linux](BuildOnLinuxOSX.md#Build) or [Windows](BuildOnWindows.md#Build).


### Run Test

```sh
# Run unit test to make sure all test passed.
make check-onnx-lit
make check-onnx-backend
make check-onnx-backend-dynamic
make check-onnx-backend-constant
make test
```

## Step 6: Commit

Commit your changes, always using the `-s` flag in order to sign your commits.

```sh
git commit -s
```

Likely you'll go back and edit/build/test some more than `commit --amend -s`
in a few cycles.

## Step 7: Keep your branch in sync

```sh
# While on your myfeature branch.
git fetch upstream main
git merge upstream/main
```

If the administrator merges other's PR on main branch while you're working on the `myfeature` branch,
conflict may occurs. You're responsible for solving the conflict. 

## Step 8: Push

When ready to review (or just to establish an offsite backup or your work),
push your branch to your fork on `github.com`:

```sh
git push -f -u origin myfeature
```

Note that even if branches are pushing to one's own fork, the PR will be created on the shared https://github.com/onnx/onnx-mlir/pulls site for everyone to review.

## Step 9: Create a pull request

1. Visit your fork at https://github.com/$user/onnx-mlir (replace `$user` obviously).
2. Click the `Compare & pull request` button next to your `myfeature` branch.

## Step 10: Get a code review

Once your pull request has been opened, it will be assigned to at least one
reviewer. The reviewer(s) will do a thorough code review, looking for
correctness, bugs, opportunities for improvement, documentation and comments,
and style.

Commit changes made in response to review comments to the same branch on your
fork. Continue to do a sequence of `git commit -s` and `git push` commands (Steps 6 and 8) to update GitHub of your changes. 

If others change the `origin/main` branch, perform a sequence of `git fetch` and `git merge` commands (Step 7) to keep your branch up to date. 
This step can also be performed on the GitHub website by visiting your PR page and clicking the `Update` button.


## Step 11: Final merge into origin/main branch

When the PR has been approved by one or more reviewers and all the CIs have passed, the PR can now be merged into the main branch of ONNX-MLIR. 
When doing so, the log message associated with this PR must be short and informative. 
By default, the log will include the messages of every `commit` performed during the development, which is necessary for smooth reviewing but is unnecessarily long. 
One approach to fix the log is to perform interactive rebase git commands. 
There is a much easier way using the GitHub interface, listed below.

1. In the web page associated with your PR, click the `Squash and Merge` button
2. In the text box above the green `Confirm squash and merge` button, edit the log. Ideally, it should have only one short paragraph describing the work, plus the relevant `Sign-off-by` and `Co-authored-by` information.
3. Only once the log is properly edited, click on the Confirm squash and merge` button.


## Final recommendations and code style

* Very small PRs are easy to review. Very large PRs are very difficult to
review.

* Follow the [coding style](https://llvm.org/docs/CodingStandards.html) used by LLVM for your code. We use the `clang-format` command to get the proper style, which is also tested by our CIs. It is acceptable to run the command on all of the files that were modified by your PR.


<!--- SPDX-License-Identifier: Apache-2.0 -->
# Contribution Guide

## Step 1: Fork ONNX-MLIR on the GitHub web interface

We strongly encourage contributors to work in their own forks of the ONNX-MLIR project.

Creating a new fork is easy:

1. Visit https://github.com/onnx/onnx-mlir
2. Click `Fork` button (top right) to establish a fork.
3. Navigate to your newly created fork, click on the the green `Code` button to get the link to *your* newly-created ONNX-MLIR fork:
```sh
git@github.com:<user>/onnx-mlir.git
```
or
```sh
https://github.com/<user>/onnx-mlir.git
```

where `<user>` is your GitHub username.

## Step 2: Setup MLIR

Depending on whether you are using docker or not, either follow Step 2a or Step 2b below.

### Step 2a: Setup using Docker

Use the template provided in [here](Docker.md#building-onnx-mlir-in-a-docker-environment) to establish a docker image that uses your ONNX-MLIR fork by modifying it as follows:

1. Since the base image used by the template already contains a clone of the ONNX-MLIR main repository, in step 5, add your fork as a remote repository by uncommenting:
```sh
RUN git remote add origin https://github.com/<user>/onnx-mlir.git
```

Replace `<user>` with your GitHub user name.

As a best practice, uncomment the line which disables the pushing to upstream to avoid accidental pushes:
```sh
RUN git remote set-url --push upstream no_push
```

At the end of the commands in Step 5:
- `upstream` will refer to the original ONNX-MLIR repository.
- `origin` will refer to your own fork of ONNX-MLIR.


2. Uncomment either step 3 or 4 depending on whether you plan to use VSCode in conjunction with the ONNX-MLIR image.

3. By default, ONNX-MLIR is built in `Debug` mode. Make the appropriate changes in step 6 if you wish to build ONNX-MLIR in `Release` mode.

At any point you can access your Docker image interactively:
```sh
docker run -it myImageName /bin/bash
```

Once inside the image you can navigate to the ONNX-MLIR GitHub repository:
```sh
cd /workdir/onnx-mlir
```

Once inside the repository you can interact with Git via the usual Git commands.


### Step 2b: Setup without Docker

Define a local working directory:

```sh
working_dir={your working directory}
```

Then follow the directions in this section of the top level [README](../README.md) and OS specific instructions [Linux](BuildOnLinuxOSX.md#MLIR) or [Windows](BuildOnWindows.md#MLIR) for installing the currently supported MLIR version in your working directory.

If you already have an MLIR copy in your working directory, you should ensure that you have the latest copy. To do so, compare the most recent commit ID from a `git log` command with the specific branch version extracted by the `git checkout` command listed [here](BuildOnLinuxOSX.md#MLIR). If your MLIR in not up to date, you must  bring it up to the correct commit level by either reinstalling it or updating it with `git fetch`, `git merge`, and `git checkout` commands.

Create your clone (replace `<user>` with your GitHub username):

```sh
mkdir -p $working_dir
cd $working_dir
git clone --recursive https://github.com/<user>/onnx-mlir.git
# or: git clone --recursive git@github.com:<user>/onnx-mlir.git

cd $working_dir/onnx-mlir
git remote add upstream https://github.com/onnx/onnx-mlir.git
# or: git remote add upstream git@github.com:onnx/onnx-mlir.git

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

## Step 3: Understanding the repository structure

Regardless of whether you are using a Docker image or not, the steps below are again common to both environments.

At the end of the repository setup commands above:
- `upstream` will refer to the original ONNX-MLIR repository.
- `origin` will refer to your own fork of ONNX-MLIR.

Never commit anything to your fork's `main` branch, the only way you should update `main` is from `upstream`. The procedure to update your fork's `main` branch is listed in Step 4.

## Step 4: Keeping your repository up to date

To keep your ONNX-MLIR fork's `main` up to date perform the following steps:

1. Fetch the latest versions of your fork (`origin`) and the `upstream` repositories:
```sh
git fetch --all
```

2. Update the `main` branch on your fork:
```sh
git checkout main
git merge origin/main
git merge upstream/main
git push origin main
```

Provided you have never committed anything to your fork's `main` branch directly, all the updates to your fork's `main` should be fast forwards.

3. The `main` branch of your fork should now be identical to the `main` branch of `upstream`. To check you can do:
```sh
git diff upstream/main
```
and the command will return immediately signaling that no differences exist between `upstream/main` and `origin/main`

## Step 5: Create a branch for your changes

To create a branch off your fork's `main` branch ensure your current branch is `main` by doing:

```sh
git checkout main
```

Then create your new branch:

```sh
git checkout -b my-branch
```

At this point you are ready to develop the code.

## Step 6: Develop

### Edit your code

You can now edit the code on the `my-branch` branch.

### Run cmake & make

Follow the directions to build ONNX-MLIR for the OS that you are using [Linux](BuildOnLinuxOSX.md#Build) or [Windows](BuildOnWindows.md#Build).

We expect code to compile without generating any compiler warnings.

### Run Test

In general, the new features must be tested in one or more of our test suite.
At a high level, our testing strategy includes `literal` tests (`check-onnx-lit` below), end-to-end tests derived from the ONNX Standard (`check-onnx-backend` and derivatives below, and semi-exhaustive numerical tests (`test` below).

```sh
# Run unit test to make sure all test passed.
make check-onnx-lit
make check-onnx-backend
make check-onnx-backend-dynamic
make check-onnx-backend-constant
make check-onnx-numerical
```
Specific testing help is provided in these pages to [run](TestingHighLevel.md) and[generate new tests](Testing.md).

## Step 7: Commit & Push

ONNX-MLIR requires committers to sign their code using the [Developer Certificate of Origin (DCO)](https://developercertificate.org).
There is a one time setup to register your name and email.
The commands are listed below, where you substitute your name and email address in the "John Doe" fields.

```sh
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

You may also be asked to sign a Developer Certificate of Origin (DCO)
at some times during the PR review.
If you do, you will have to accept in order to contribute code.

Once these initial tasks are done, you are ready to sign your code by using the `-s` flag during your commits.

```sh
git commit -s
```

Push your changes:
```sh
git push origin my-branch
```

Note that even if branches are pushing to one's own fork, the PR will be created on the shared https://github.com/onnx/onnx-mlir/pulls site for everyone to review.

## Step 8: Update your branch

Assuming your `main` is up to date (Step 4), to update any branches you are currently working on to use the latest ONNX-MLIR, you need to do the following:

```sh
git checkout my-branch
git merge origin/main
```

If no conflicts are signaled and the merge is complete do:

```sh
git push origin my-branch
```

However, if conflicts appear, the merge will be interrupted until the conflicts are resolved. A list of files will be marked as containing conflicts. To identify those files do:

```sh
git status -uno
```

The files in red are the files containing conflicts. Go to all the files which contain a conflict and resolve the conflicts.
When the conflicts are resolved do a `git add` on each conflicted file:

```sh
git add path/to/file1
git add path/to/file2
...
```

When all conflicted files have been added do:
```sh
git commit -s
```

Followed by a git push:
```sh
git push origin my-branch
```

Your branch is now up to date with the latest ONNX-MLIR.


## Step 9: Create a pull request

1. Visit your fork at `https://github.com/<user>/onnx-mlir` (replace `<user>` obviously).
2. Click the `Compare & pull request` button next to your `my-branch` branch.

## Step 10: Get a code review

Once your pull request has been opened and is not in draft mode anymore, one of us will review the code.
The reviewer(s) will do a thorough code review, looking for correctness, bugs, opportunities for improvement, testing, documentation and comments, and style.

Commit changes made in response to review comments to the same branch on your fork. Continue to do a sequence of `git commit -s` and `git push` commands (Step 7) to update GitHub of your changes.

If you wish to update your branch to contain the latest ONNX-MLIR changes perform Step 8.

This step can also be performed on the GitHub website by visiting your PR page and clicking the `Update` button. This step will merge the latest `upstream/main` branch into your branch without updating the `main` branch of your fork.

## Step 11: Pull request approval

When the PR has been approved by one or more reviewers and all the CIs have passed, the PR can now be merged into the main branch of ONNX-MLIR.

Your PR will be squashed into a single commit before being merged into the ONNX-MLIR main branch.

This step will be performed by an ONNX-MLIR admin.

By default, the log of your commit will be made to consist of:
- description consisting of the title of your PR
- the reviewer sign-off
- any co-authors

For contributors who wish to provide a custom description you will have to do the squashing of the commits in your PR yourself by performing an interactive rebase on the latest ONNX-MLIR.

For lengthy, detailed descriptions please use the main comment box in your PR.

### Collaborators with Write access guidelines

By default, the log will include the messages of every `commit` performed during the development, which is necessary for smooth reviewing but is unnecessarily long. During the merge phase this message will be replaced with the title of the patch unless the author of the patch has already squashed all his commits via an interactive rebase and provided his own custom (but brief) description of the patch.

Using the GitHub interface:
 1. In the web page associated with the PR, click the `Squash and Merge` button;
 2. In the text box above the green `Confirm squash and merge` button, edit the log.
 3. Ideally, it should have only one short paragraph describing the work, plus the relevant `Sign-off-by` and `Co-authored-by` information. If the user has provided this already do step 4. If not, clear the intermediate commit messages and use the patch title as the description, add sign-off and co-author information. 
 4. Only once the log is properly edited, click on the `Confirm squash and merge` button.

## Code style

Very small PRs are easy to review. Very large PRs are very difficult to review.

Follow the [coding style](https://llvm.org/docs/CodingStandards.html) used by LLVM for your code. We use the `clang-format` command to get the proper style, which is also tested by our CIs. It is acceptable to run the command on all of the files that were modified by your PR. We recommend using VS code where the clang formatter will be run automatically using the clang format configuration file already present in the repository.

For python code, we use the [black](https://pypi.org/project/black) code formatter. You should run the `black` command on all the python code modified by your PR, which must pass the black code formatter CI check before it can be merged.

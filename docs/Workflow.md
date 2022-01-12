<!--- SPDX-License-Identifier: Apache-2.0 -->
# Contribution Guide

## Prerequisite: Install LLVM

Follow directions in this section of the top level [README](../README.md#mlir) 
for installing the currently supported LLVM version. We assume here a
LINUX installation. 

### Step 1: Fork in the cloud

1. Visit https://github.com/onnx/onnx-mlir
2. Click `Fork` button (top right) to establish a fork.

### Step 2: Clone fork to local storage

Define a local working directory:

```sh
working_dir={your working directory}
```

Set `user` to match your github profile name:

```sh
user={your github profile name}
```

Create your clone:

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

### Step 3: Branch

Get your local main up to date:

```sh
cd $working_dir/onnx-mlir
git fetch upstream
git checkout main
git rebase upstream/main
```

Branch from main:

```sh
git checkout -b myfeature
```

### Step 4: Develop

#### Edit the code

You can now edit the code on the `myfeature` branch.

#### Run cmake & make

Follow the directions to build ONNX-MLIR in this section of the main 
[README](../README.md#onnx-mlir-this-project)


#### Run Test

```sh
# Run unit test to make sure all test passed.
make check-onnx-lit
make check-onnx-backend
make check-onnx-backend-dynamic
make check-onnx-backend-constant
make test
```

### Step 5: Commit

Commit your changes, always using the `-s` flag in order to sign your commits.

```sh
git commit -s
```

Likely you'll go back and edit/build/test some more than `commit --amend -s` in a few cycles. To avoid polluting the history you should squash all your feature commits together by using the following command:

```sh
git commit -i rebase HEAD~n (where n is the number of commits you want to squash)
```

An editor will pop up, and you should mark all commits except the first one with an 's' to indicate that those are the commits you want to squash with the first one. At this point you will be able to edit the commit message (remove all individual commit messages except for the first one). After rebasing, the `git log` should contain only one commit for your feature.

### Step 6: Keep your branch in sync

```sh
# While on your myfeature branch.
git fetch upstream
git rebase upstream/main
```

If the administrator merges other's PR on main branch while you're working on the `myfeature` branch,
conflict may occurs. You're responsible for solving the conflict. Then continue:

```sh
git add --all
git rebase --continue
git commit --amend --no-edit -s
```

### Step 7: Push

When ready to review (or just to establish an offsite backup or your work),
push your branch to your fork on `github.com`:

```sh
git push -f origin myfeature
```

### Step 8: Create a pull request

1. Visit your fork at https://github.com/$user/onnx-mlir (replace `$user` obviously).
2. Click the `Compare & pull request` button next to your `myfeature` branch.
3. New pull requests should be created in "draft mode", unless they are ready for code review (you know they already pass all tests).
4. When the pull request is clean (i.e. all build bots are green) it is ready for code review. At this point you should change it into a regular pull request by clicking the 'Ready for Review' in the GUI.

Pull requests that are in draft mode can be modified by the author. Should you need to modify a pull request in **draft** mode, squashing additional commits together is encouraged (to keep the log history clean).
Important: when a pull request is under active review (not in draft mode), any additional commits should **not** be squashed (in order to allow reviewers to more easily determine their code review comments have been addressed).

### Step 9: Get a code review

Once your pull request is ready for code review, you need to request code review from one or more reviewers. The reviewer(s) will do a thorough code review, looking for correctness, bugs, opportunities for improvement, documentation and comments, and style.

Commit changes made in response to review comments to the same branch on your fork. At this point in the process you should **not** rebase your feature commits in order to preserve history while the pull request is under active review. If the branch goes out of sync with `main` you are encouraged to rebase your feature commit(s) on the latest version of the `main` branch (please do not squash your feature commits while doing so):

```sh
  git rebase -i origin/main
```

Very small PRs are easy to review. Very large PRs are very difficult to review. Submission of minimal pull requests for review is highly encouraged.

### Step 10: Committing an approved pull request

Once code review comments have been addressed, and at least a reviewer has approved the pull requests, it is time to merge it into the `main` branch. 

If the pull request is "out-of-date" with the main branch, it needs to be rebased. Rebasing can be done either through the command line `git rebase -i origin/main`, or by using the GUI. Either way this action will trigger a new test cycle.

Once the tests are green again, and the pull request is no longer out-of-date with the `main` branch it can be merged. The final merge can be done by using the GUI. Remember to edit the final commit message so that the git log will show a clean and concise commit message (rather than a set of individual commits).

## Code style

Please follow the coding style used by LLVM (https://llvm.org/docs/CodingStandards.html).


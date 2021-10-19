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

# Never push to upstream master since you do not have write access.
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

Get your local master up to date:

```sh
cd $working_dir/onnx-mlir
git fetch upstream
git checkout master
git rebase upstream/master
```

Branch from master:

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

Likely you'll go back and edit/build/test some more than `commit --amend -s`
in a few cycles.

### Step 6: Keep your branch in sync

```sh
# While on your myfeature branch.
git fetch upstream
git rebase upstream/master
```

If the administrator merges other's PR on master branch while you're working on the `myfeature` branch,
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

### Step 9: Get a code review

Once your pull request has been opened, it will be assigned to at least one
reviewer. The reviewer(s) will do a thorough code review, looking for
correctness, bugs, opportunities for improvement, documentation and comments,
and style.

Commit changes made in response to review comments to the same branch on your
fork.

Very small PRs are easy to review. Very large PRs are very difficult to
review.

## Code style

Please  follow the coding style used by LLVM (https://llvm.org/docs/CodingStandards.html).


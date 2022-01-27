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
[README](../README.md#onnx-mlir-this-project).

We expect code to compile without generating any compiler warnings.

#### Run Test

In general, the new features must be tested in one or more of our test suite.
At a high level, our testing strategy includes `literal` tests (`check-onnx-lit` below), end-to-end tests derived from the ONNX Standard (`check-onnx-backend` and derivatives below, and semi-exhaustive numerical tests (`test` below).

```sh
# Run unit test to make sure all test passed.
make check-onnx-lit
make check-onnx-backend
make check-onnx-backend-dynamic
make check-onnx-backend-constant
make test
```

Specific testing help is provided in these documents to [run](TestingHighLevel.md) and[generate new tests](Testing.md).

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
3. Please provide a meaningful description of the pull request's new functionality.
4. If your pull request is not currently ready for review, we recommend to activate the `draft mode` button. Pull requests that are failing some CIs should probably remain in draft mode.

### Step 9: Get a code review

Once your pull request has been opened and is not in draft mode anymore, one of us will review the code.
The reviewer(s) will do a thorough code review, looking for correctness, bugs, opportunities for improvement, testing, documentation and comments, and style.

Commit changes made in response to review comments to the same branch on your
fork.

Very small PRs are easy to review. Very large PRs are very difficult to
review.

## Code style

Please  follow the coding style used by LLVM (https://llvm.org/docs/CodingStandards.html).


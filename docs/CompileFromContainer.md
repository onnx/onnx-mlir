<!--- SPDX-License-Identifier: Apache-2.0 -->

# Container Compilation from Inside a Container

`OMCompile` can compile a model by running onnx-mlir in a container image
(container mode). This page covers the case where **your own program already
runs inside a container** — a dev container, a CI job, a Kubernetes pod — and
calls `OMCompile` in container mode from there.

## How it works: sibling containers

`OMCompile` does not run a container engine inside your container. It runs the
`docker` or `podman` command-line tool, which sends the request to a daemon.
Usually that daemon is the host's, reached through a socket mounted into your
container. The compiler container is then created on the host, **next to**
yours, not inside it:

```
Host
├── daemon (dockerd / podman)  <── /var/run/docker.sock mounted into ──┐
├── your container (runs OMCompile, has the docker/podman CLI) ─────────┘
└── compiler container (onnx-mlir), started by the host daemon
```

This is often called Docker-outside-of-Docker (DooD). It needs no privileged
mode and no nested daemon.

The one rule to keep in mind: **paths given to `-v` are resolved on the
daemon's machine, not inside your container.** `OMCompile` mounts the
directory of the input model and the directory of the output file into the
compiler container at the same paths, so those directories must exist at those
paths as seen by the daemon.

## Setup

### 1. Install the CLI in your image

Your container needs the `docker` or `podman` command-line tool (the daemon is
not needed):

```dockerfile
FROM ghcr.io/onnxmlir/ubuntu:noble
RUN apt-get update && apt-get install -y docker.io   # or: podman
```

### 2. Give the CLI a reachable daemon

Mount the host socket. With Docker:

```bash
docker run -v /var/run/docker.sock:/var/run/docker.sock \
           -v /your/workspace:/your/workspace \
           your-app
```

With rootless Podman (no root on the host), first enable the user socket once:

```bash
systemctl --user enable --now podman.socket
```

```bash
podman run -v $XDG_RUNTIME_DIR/podman/podman.sock:/var/run/docker.sock \
           -v /your/workspace:/your/workspace \
           your-app
```

Here the `docker` CLI inside the container talks to Podman through its
Docker-compatible socket.

The socket does not have to be at `/var/run/docker.sock`. Anything the CLI can
reach works, for example `DOCKER_HOST=unix:///path/to/sock` or
`DOCKER_HOST=tcp://host:2376` for Docker, and `CONTAINER_HOST` for Podman.
Before compiling, `OMCompile` runs `docker info` / `podman info` and stops with
an explanatory error if the daemon cannot be reached.

### 3. Make the model and output paths match the host

Mount your work directory at **the same path** inside your container as on the
host (`-v /your/workspace:/your/workspace` above). Then a path such as
`/your/workspace/model.onnx` means the same file to your program and to the
host daemon, and nothing else is needed.

If you cannot use the same path, set `OM_DOCKER_HOST_PATH_PREFIX` (see below).

## Using it from code

Nothing is different from container mode on the host:

```cpp
#include "src/Compiler/OMCompile.hpp"
#include <iostream>

using namespace onnx_mlir;

int main() {
  try {
    OMCompile compiler("ghcr.io/onnxmlir/onnx-mlir", /*compilerPath=*/{},
        OMCompile::ContainerEngine::Auto, /*autoPull=*/true,
        /*verbose=*/true);
    if (compiler.isRunningInContainer())
      std::cout << "Running inside a container" << std::endl;
    compiler.compile("/your/workspace/model.onnx", "-O3");
    std::cout << "Output: " << compiler.getOutputFilename() << std::endl;
  } catch (const OMCompileException &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
```

Use absolute paths, or relative paths under a directory that is mounted at the
same path on the host.

## Detecting that it runs in a container

`OMCompile::isRunningInContainer()` returns true when any of these holds:

1. `/.dockerenv` exists (Docker).
2. `/proc/1/cgroup` mentions `docker`, `containerd`, or `podman`.
3. The `container` environment variable is set.
4. `/run/.containerenv` exists (Podman).
5. `/proc/1/cgroup` is `0::/` (cgroups v2) and the hostname is 12 hex digits.

Detection only decides whether `OM_DOCKER_HOST_PATH_PREFIX` is applied and what
the error messages suggest. The daemon check in step 2 runs either way.

## Environment variables

- **`OM_DOCKER_HOST_PATH_PREFIX=<prefix>`**: when running in a container,
  `<prefix>` is prepended to the model and output directories to form the
  host-side source of each `-v` mount. The target inside the compiler
  container stays the original path. For example, with the prefix `/host`,
  `/workspace` is mounted as `-v /host/workspace:/workspace`. This is a plain
  prefix: it fits a layout where the host path equals `<prefix>` followed by
  the container path. It does not translate arbitrary mounts. Without it,
  paths are used unchanged.
- **`OM_DIND_DISABLE=1`**: skip detection and treat the process as running on
  the host, so no prefix is applied.

## A real nested daemon

Some CI systems run a separate daemon in its own container, for example
GitLab's `docker:dind` service, reached with
`DOCKER_HOST=tcp://docker:2375`. The same rule applies: `-v` paths are
resolved on the machine that runs that daemon. The compile works only if that
daemon sees your model and output directories at the same paths, for example
because the CI shares the build directory with the service. If it does not,
the mounted directories are empty and the compile fails with a missing input
file.

## Other setups

VS Code dev container (`devcontainer.json`):

```json
{
  "image": "my-dev-image",
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind",
    "source=${localWorkspaceFolder},target=${localWorkspaceFolder},type=bind"
  ],
  "workspaceFolder": "${localWorkspaceFolder}"
}
```

Kubernetes pod with the node's Docker socket (the node must run Docker; work
files must be on a `hostPath` volume mounted at the same path):

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: compiler
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
    - name: work
      mountPath: /work
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
  - name: work
    hostPath:
      path: /work
```

If no daemon can be made reachable, install onnx-mlir in your container and
use local mode (`OMCompile(compilerPath)`) instead.

## Troubleshooting

**"Container engine ... cannot reach its daemon/service".** The CLI is
installed but has no daemon. Mount the socket or set `DOCKER_HOST` /
`CONTAINER_HOST` (step 2).

**Permission denied on the socket.** With Docker, the user in your container
needs access to the socket (for example, the socket's group). With Podman,
use the rootless user socket shown in step 2.

**Input file not found, or no output file, although the file exists in your
container.** The daemon cannot see that path. Check that the directory is
mounted at the same path on the host (step 3), or set
`OM_DOCKER_HOST_PATH_PREFIX`. When the source of a `-v` mount does not exist on
the host, Docker creates an empty directory there, so the compiler sees no
input.

Run with `verbose=true` to see the detection result, the daemon check, the
resolved paths, and the full `run` command:

```
Docker-in-Docker (DinD) environment detected
Container engine reachable: docker
Resolved DinD path: /workspace -> /host/workspace
Container command: onnx-mlir -O3 /workspace/model.onnx
```

## Security

Access to the host's Docker socket gives full control of the host's daemon,
which is equivalent to root on the host. Mount it only into containers you
trust. Rootless Podman limits this to the rights of the host user that owns
the socket.

## References

- `src/Compiler/OMCompile.hpp`: the `OMCompile` API.
- [Docker: protect the daemon socket](https://docs.docker.com/engine/security/protect-access/)
- [Podman rootless tutorial](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)

## Work Items
** Test rootless Podman
** Test OM_DOCKER_HOST_PATH_PREFIX
** Handle `--config-file`
** Update user manual on how to start the app container and how the compiler related directory should be located

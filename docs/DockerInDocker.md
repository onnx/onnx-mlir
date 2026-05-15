# Docker-in-Docker (DinD) Support

## Overview

`OMUnifiedCompile` automatically handles Docker-in-Docker scenarios - when your application runs inside a container and needs to compile ONNX models using containerized onnx-mlir.

**Key Features:**
- Automatic detection (Docker, Podman, Containerd)
- Transparent path resolution
- No code changes required
- Works with Podman rootless (no sudo needed)

## Quick Start

### Setup Container Socket

**Option 1: Podman Rootless (Recommended - No Sudo)**

**Step 1: Build Container Image with Docker CLI**

Your container image must have Docker CLI installed:

```dockerfile
FROM ghcr.io/onnxmlir/ubuntu:noble

# Install Docker CLI to communicate with Podman socket
RUN apt-get update && apt-get install -y docker.io

# Copy your application
COPY your-app /usr/local/bin/
```

Build and push:
```bash
podman build -t your-app:latest .
podman push your-app:latest
```

**Step 2: Enable Podman Socket (one-time on host)**
```bash
systemctl --user enable --now podman.socket
```

**Step 3: Run Container**
```bash
# Mount Podman socket and workspace
podman run -v $XDG_RUNTIME_DIR/podman/podman.sock:/var/run/docker.sock \
           -v /your/workspace:/your/workspace \
           your-app:latest

# Explanation:
# - $XDG_RUNTIME_DIR/podman/podman.sock: User-owned Podman socket (no sudo)
# - /var/run/docker.sock: Standard Docker socket location
# - Docker CLI in container communicates with Podman on host via socket
# - /your/workspace: Same path inside/outside for DinD path resolution
```

**Important:** Simply having a Dockerfile in the directory does NOT install Docker CLI in the running container. You must build a new image with Docker CLI included, then use that image.

**Option 2: Docker (Requires Docker Group/Sudo)**
```bash
docker run -v /var/run/docker.sock:/var/run/docker.sock \
           -v /your/workspace:/your/workspace \
           your-app
```

### Use in Code

```cpp
#include "OMUnifiedCompile.hpp"

// Works automatically in both regular and DinD scenarios
OMUnifiedCompile compiler(
    "ghcr.io/onnxmlir/onnx-mlir-dev",
    "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"
);

compiler.compile("/workspace/model.onnx", "-O3");
```

## How It Works

### Automatic Detection

Detects container environments using 5 methods:
1. `/.dockerenv` file (Docker)
2. `/proc/1/cgroup` content (Docker/Podman/Containerd)
3. `container` environment variable
4. `/run/.containerenv` file (Podman)
5. Hostname pattern + cgroups v2 (modern Podman)

### Path Resolution

Automatically resolves paths from outer container to host:
```
Outer container: /workspace/model.onnx
→ Host: /data/models/model.onnx (auto-resolved)
→ Inner container: /data/models/model.onnx (mounted)
```

## Requirements

1. **Mount container socket** (see Quick Start)
2. **Use absolute paths** in your code
3. **Ensure host accessibility** - paths must exist on host, not just outer container

## Configuration

### Check DinD Status
```cpp
if (compiler.isRunningInContainer()) {
    std::cout << "Running in DinD mode" << std::endl;
}
```

### Enable Verbose Mode
```cpp
OMUnifiedCompile compiler(
    "image", "compiler",
    OMUnifiedCompile::ContainerEngine::Auto,
    true,  // auto-pull
    true   // verbose - shows detection and path resolution
);
```

### Environment Variables

**`DIND_DISABLE=1`** - Disable DinD detection

**`DOCKER_HOST_PATH_PREFIX=/host`** - Set host path prefix when host filesystem is mounted at different location

## Common Use Cases

### CI/CD Pipeline
```yaml
# .gitlab-ci.yml
build:
  image: my-build-image
  services:
    - docker:dind
  script:
    - ./my-app compile model.onnx
```

### VS Code Dev Container
```json
{
  "image": "my-dev-image",
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind"
  ]
}
```

### Kubernetes
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: compiler
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
```

## Troubleshooting

### Socket Not Found
Mount the socket in your outer container (see Quick Start).

### Socket Permission Denied
**For Docker:** Add user to docker group or use `--privileged`
**For Podman:** Use rootless socket (recommended):
```bash
systemctl --user enable --now podman.socket
podman run -v $XDG_RUNTIME_DIR/podman/podman.sock:/var/run/docker.sock ...
```

### File Not Found in Inner Container
1. Enable verbose mode to see path resolution
2. Verify paths are mounted from host to outer container
3. Use `DOCKER_HOST_PATH_PREFIX` if needed
4. Ensure absolute paths in code

### Debug Output Example
```
Docker-in-Docker (DinD) environment detected
Docker socket verified: /var/run/docker.sock
Resolved DinD path: /workspace/model.onnx -> /host/workspace/model.onnx
```

## Complete Example

```cpp
#include "OMUnifiedCompile.hpp"
#include <iostream>

int main() {
    try {
        OMUnifiedCompile compiler(
            "ghcr.io/onnxmlir/onnx-mlir-dev",
            "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
            OMUnifiedCompile::ContainerEngine::Auto,
            true,  // auto-pull
            true   // verbose
        );

        if (compiler.isRunningInContainer()) {
            std::cout << "DinD mode active" << std::endl;
        }

        compiler.compile("/workspace/model.onnx", "-O3");
        std::cout << "Output: " << compiler.getOutputFilename() << std::endl;

        return 0;
    } catch (const OMCompileException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

**Run it:**
```bash
# Regular
./my-app

# DinD with Podman (rootless)
podman run -v $XDG_RUNTIME_DIR/podman/podman.sock:/var/run/docker.sock \
           -v /workspace:/workspace \
           my-app-image
```

## Security Notes

- Mounting Docker socket gives container full Docker daemon access (equivalent to root)
- Use only in trusted environments
- Podman rootless is more secure than Docker with sudo
- Consider alternatives like Kaniko or Buildah for production

## References

- [OMUnifiedCompile API](OMUnifiedCompile.hpp)
- [Docker-in-Docker Docs](https://docs.docker.com/engine/reference/run/)
- [Podman Rootless](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)
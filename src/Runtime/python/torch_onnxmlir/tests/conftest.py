# SPDX-License-Identifier: Apache-2.0

###############################################################################
# conftest.py — pytest session-wide fixtures for torch_onnxmlir tests.
#
# om_pyrt is the compiled onnx-mlir Python runtime (a C extension).  It is
# only present when onnx-mlir has been built and installed.  Tests that only
# exercise pure-Python logic (hash-key helpers, session-cache integrity …)
# do not need a working compiler or runtime; they just need the module to be
# importable.
#
# This conftest installs a minimal stub for om_pyrt into sys.modules *before*
# pytest collects any test module, so that imports of torch_onnxmlir.backend
# succeed in environments where onnx-mlir has not been compiled.
###############################################################################

import sys
import types


def _make_om_pyrt_stub():
    """Return a minimal stub module that satisfies backend.py's top-level import."""
    stub = types.ModuleType("om_pyrt")

    class _Stub:
        """Placeholder class; tests that need a real session must mock further."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "om_pyrt stub: this test requires a real onnx-mlir build"
            )

    stub.CompileSession = _Stub
    stub.InferenceSession = _Stub
    return stub


# Only inject the stub when om_pyrt is not already available (i.e. no real build).
if "om_pyrt" not in sys.modules:
    try:
        import om_pyrt  # noqa: F401  — real module present, nothing to do
    except ModuleNotFoundError:
        sys.modules["om_pyrt"] = _make_om_pyrt_stub()

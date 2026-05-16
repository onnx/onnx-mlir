def CompileWithStandalone(model, options=""):
    try:
        import OMPyCompile
    except ImportError as e:
        print(f"Error: {e.msg}")
        print(f"Module name: {e.name}")
        print("Please install package for standalone compiler")
        exit(-1)
    try:
        compile_session = OMPyCompile.OMCompile(model, options)
    except RuntimeError as e:
        print(f"Compilation with standalone compiler failed: {e.msg}")
        exit(-1)

    return compile_session.get_output_file_name()


def CompileWithLocal(model, options="", compiler_path=""):
    from .PyOMCompile import OMCompile

    try:
        compile_session = OMCompile(model, options, compiler_path)
    except RuntimeError as e:
        print(f"Compilation with local compiler failed: {e.msg}")
        exit(-1)

    return compile_session.get_output_file_name()


def compile(model, method="local", **kwargs):
    compile_option = kwargs.get("compile_options", "")
    if method == "container":
        from .CompileWithContainer import CompileWithContainer

        return CompileWithContainer(model, **kwargs)
    elif method == "standalone":
        return CompileWithStandalone(model, compile_option)
    elif method == "local":
        compiler_path = kwargs.get("compiler_path", "")
        return CompileWithLocal(model, compile_option, compiler_path)

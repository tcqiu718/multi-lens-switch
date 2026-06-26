import cupy


def compile_kernel(function_name, kernel_code):
    """Compile a CuPy CUDA kernel across old and new CuPy releases.

    The original FI models used cupy.cuda.compile_with_cache, which is not
    available in recent CuPy versions. RawModule is the supported replacement.
    """
    compile_with_cache = getattr(cupy.cuda, "compile_with_cache", None)
    if compile_with_cache is not None:
        return compile_with_cache(kernel_code).get_function(function_name)

    module = cupy.RawModule(code=kernel_code)
    return module.get_function(function_name)

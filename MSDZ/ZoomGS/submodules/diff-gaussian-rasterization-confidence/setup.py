#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

cxx_flags = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]
nvcc_flags = [
    "-I" + os.path.join(ROOT, "third_party/glm/"),
    "--use_fast_math",
]
if os.name == "nt":
    nvcc_flags.extend(["-Xcompiler", "/std:c++17"])
else:
    nvcc_flags.extend(["-std=c++17"])

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

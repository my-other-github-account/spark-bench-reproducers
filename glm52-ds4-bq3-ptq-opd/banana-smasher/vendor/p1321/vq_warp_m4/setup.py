# SPDX-License-Identifier: Apache-2.0
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vq-warp-gemv",
    version="0.2.0",
    description="Warp-per-output learned-VQ BF16 small-M GEMV for GB10",
    license="Apache-2.0",
    packages=["vq_warp_gemv"],
    ext_modules=[
        CUDAExtension(
            "vq_warp_gemv._C",
            ["csrc/vq_warp_gemv.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "--threads=2", "-Xptxas=-v"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)

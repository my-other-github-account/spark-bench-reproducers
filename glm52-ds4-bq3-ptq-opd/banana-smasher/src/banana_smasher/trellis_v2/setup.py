from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="banana-smasher-trellis-v2-exact",
    ext_modules=[
        CUDAExtension(
            name="trellis_v2_cuda_exact",
            sources=["csrc/binding_exact.cpp", "csrc/trellis_v2_exact.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17", "-lineinfo", "--fmad=false",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)

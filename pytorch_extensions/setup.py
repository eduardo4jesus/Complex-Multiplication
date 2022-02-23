from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ~~~~~~~~~~~~~~~
# ~~~~~ NOTE ~~~~
# ~~~~~~~~~~~~~~~
  # Since this file can be loaded from anywhere as a python script,
  # Let's point to the source files in an absolute way.
from pathlib import Path
__parent__ = Path(__file__).absolute().parent
# ~~~~~~~~~~~~~~~

setup(
    name='my_pytorch_extensions',
    ext_modules=[
        CUDAExtension('complex_multiplication',
            [
                f"{__parent__/'pytorch_extensions_cuda.cpp'}",
                f"{__parent__/'complex_multiplication_kernel.cu'}"
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

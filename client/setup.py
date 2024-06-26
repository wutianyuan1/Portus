##
# @file rdma_wrap.cpp
# @brief Implementation of Portus client RDMA communacation protocols
# @author madoka, stevelee477

from setuptools import setup
from torch.utils import cpp_extension
import os

os.system("cp ../common/* ./")
libgpu_rdma_access = ('gpu_rdma_access', {'sources': ['gpu_direct_rdma_access.c', 'utils.c'], 'libraries': ["ibverbs", "mlx5", "rdmacm"]})

setup(name='gpu_rpma',
      ext_modules=[
            cpp_extension.CUDAExtension(
                  'gpu_rpma', 
                  ['raw_tensor.cpp', 'rdma_wrap.cpp', 'checkpoint.cpp', 'py_interface.cpp'], 
                  libraries=['gpu_rdma_access', "ibverbs", "mlx5", "rdmacm"]),
            ],
      libraries=[libgpu_rdma_access],
      package_data={'my_package': ['gpu_rpma.pyi']},
      cmdclass={
            'build_ext': cpp_extension.BuildExtension
            })

# cleanup
for filename in os.listdir('../common/'):
      os.remove('./' + filename)

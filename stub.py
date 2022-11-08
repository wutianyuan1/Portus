import os, sys
import torch

# pyd所在路径
sys.path.append(os.path.join("xxxx", "bin", "Release"))

module_name = "gpu_rpma"
exec("import %s" % module_name)

from pybind11_stubgen import ModuleStubsGenerator

module = ModuleStubsGenerator(module_name)
module.parse()
module.write_setup_py = False

with open("%s.pyi" % module_name, "w") as fp:
    fp.write("#\n# Automatically generated file, do not edit!\n#\n\n")
    fp.write("\n".join(module.to_lines()))
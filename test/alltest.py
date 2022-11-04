import os

device = "cuda"
for pkgsize in [2**i for i in range(12)]:
    print(pkgsize, "KB")
    os.system("python test/tensor_test.py {} {}".format(pkgsize, device))
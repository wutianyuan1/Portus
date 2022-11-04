import torch
import gpu_rpma
import time


def make_nn(layer_size, num_layers):
    ret = {}
    name_str = "layer{}"
    for i in range(num_layers):
        name = name_str.format(i)
        ret[name] = torch.full((layer_size, 1), 10, dtype=torch.float, device="cuda")
    return ret


def main():
    assert torch.cuda.is_available()
    fake_params = {
        "module.layer1": torch.full((10240, 10240), 15, dtype=torch.float, device="cuda"),
        "module.layer2": torch.full((10240, 10240), 20, dtype=torch.float, device="cuda"),
        "module.layer3": torch.full((10240, 10240), 100, dtype=torch.float, device="cuda")
    }

    total_size = 1024*1024*2 # 2MB
    # for layer_size in [1024, 4096, 16*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024]:
    layer_size = int(input())
    print("SBNetwork{}".format(round(layer_size)))
    nn_params = make_nn(layer_size, total_size//layer_size)
    gpu_rpma.init_checkpoint("SBNetwork{}".format(round(layer_size)), nn_params)
    t1 = time.time()
    for _ in range(100):
        gpu_rpma.checkpoint()
    t2 = time.time()
    print("pack size = {}, time = {}".format(layer_size, t2 - t1))


if __name__ == '__main__':
    main()

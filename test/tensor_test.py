import torch
import gpu_rpma
import time
from sys import argv

def make_nn(layer_shape, num_layers, device_):
    ret = {}
    name_str = "layer{}"
    for i in range(num_layers):
        name = name_str.format(i)
        ret[name] = torch.full((layer_shape, 1), 10, dtype=torch.float, device=device_)
    return ret


def main():
    assert torch.cuda.is_available()
    # fake_params = {
    #     "module.layer1": torch.full((10240, 10240), 15, dtype=torch.float, device="cuda"),
    #     "module.layer2": torch.full((10240, 10240), 20, dtype=torch.float, device="cuda"),
    #     "module.layer3": torch.full((10240, 10240), 100, dtype=torch.float, device="cuda")
    # }
    
    # torch.save(fake_params, "./chkpt_torch.pt")
    # print("original checkpointing time:", t2-t1)
    # gpu_rpma.init_checkpoint("SBNetwork", fake_params)
    # t1 = time.time()
    # for i in range(10):
        # torch.save(fake_params, "/mnt/beegfs/shared/chkpt_torch.pt")
        # torch.save(fake_params, "/home/wuty/workspace/bert-cloth/nvm/chkpt_torch.pt")
        # gpu_rpma.checkpoint()
    # t2 = time.time()
    # print(t2-t1)
    # gpu_rpma.checkpoint()
    # exit()

    total_size = 1024*1024*2  # 2MB
    # for layer_size in [1024, 4096, 16*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024]:
    layer_size = int(argv[1])*1024
    print("SBNetwork{}".format(round(layer_size)))
    layer_shape = layer_size//4  # sizeof(float)=4
    nn_params = make_nn(layer_shape, total_size//layer_size, argv[2])
    gpu_rpma.init_checkpoint("SBNetwork{}".format(round(layer_size)), nn_params)
    iters = 100
    t1 = time.time()
    for _ in range(iters):
       gpu_rpma.checkpoint()
    t2 = time.time()
    print("pack size = {}, time = {}".format(layer_size, t2 - t1))
    with open("result.csv", "a+") as csv_file:
        num_layers = total_size//layer_size
        num_packages = num_layers * iters
        csv_file.write("{}, {}, {}\n".format(num_packages, layer_size, t2 - t1))


if __name__ == '__main__':
    main()

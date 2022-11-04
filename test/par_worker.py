import torch
import gpu_rpma
import time
from sys import argv

torch_t0 = time.time()
fake_params1 = {
    "module.layer1": torch.full((10240, 10240), 15, dtype=torch.float, device="cuda"),
    "module.layer2": torch.full((10240, 10240), 20, dtype=torch.float, device="cuda"),
    "module.layer3": torch.full((10240, 10240), 100, dtype=torch.float, device="cuda")
}

fake_params2 = {
    "module.layer1": torch.full((10240, 10240), 15, dtype=torch.float, device="cuda"),
    "module.layer2": torch.full((10240, 10240), 20, dtype=torch.float, device="cuda"),
    "module.layer3": torch.full((10240, 10240), 100, dtype=torch.float, device="cuda")
}

fake_params3 = {
    "module.layer1": torch.full((10240, 10240), 15, dtype=torch.float, device="cuda"),
    "module.layer2": torch.full((10240, 10240), 20, dtype=torch.float, device="cuda"),
    "module.layer3": torch.full((10240, 10240), 100, dtype=torch.float, device="cuda")
}

fake_params4 = {
    "module.layer1": torch.full((10240, 10240), 15, dtype=torch.float, device="cuda"),
    "module.layer2": torch.full((10240, 10240), 20, dtype=torch.float, device="cuda"),
    "module.layer3": torch.full((10240, 10240), 100, dtype=torch.float, device="cuda")
}

fake_params = [fake_params1, fake_params2, fake_params3, fake_params4]
torch_t1 = time.time()
print("Torch init: ", torch_t1 - torch_t0, "s")


def main(model_name, idx):
    global fake_params
    print("init", model_name, idx)
    t0 = time.time()
    gpu_rpma.init_checkpoint(model_name, fake_params[idx])
    t1 = time.time()
    print("rpma init: ", t1 - t0, "s")
    for _ in range(5):
        #torch.save(fake_params[idx], "/mnt/beegfs/shared/checkpoint{}.pt".format(idx))
        gpu_rpma.checkpoint()
    t2 = time.time()
    print(model_name, (t2-t1)/5)

if __name__ == '__main__':
    main(argv[1], int(argv[2]))

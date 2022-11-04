import torch
import gpu_rpma
import time
from sys import argv

def main():
    fake_params = {
        "module.layer1": torch.full((10240, 10240), 1, dtype=torch.float, device="cuda"),
        "module.layer2": torch.full((10240, 10240), 1, dtype=torch.float, device="cuda"),
        "module.layer3": torch.full((10240, 10240), 1, dtype=torch.float, device="cuda")
    }
    
    # torch.save(fake_params, "./chkpt_torch.pt")
    # print("original checkpointing time:", t2-t1)
    gpu_rpma.init_checkpoint("SBNetwork", fake_params)
    t1 = time.time()
    #for i in range(10):
        # torch.save(fake_params, "/mnt/beegfs/shared/chkpt_torch.pt")
        # torch.save(fake_params, "/home/wuty/workspace/bert-cloth/nvm/chkpt_torch.pt")
        #gpu_rpma.checkpoint()
    gpu_rpma.checkpoint()
    #gpu_rpma.restore()
    t2 = time.time()
    print(t2-t1)
    print(fake_params['module.layer1'][0][0])
    # gpu_rpma.checkpoint()

if __name__ == '__main__':
    main()

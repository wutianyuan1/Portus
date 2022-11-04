import torch
import time
import multiprocessing
import os
import sys

def sb(model_name, i):
    # 别问我为啥非要这么写，sb pytorch里边自己有sb锁不然会变成大sb
    # 在MSR的时候我已经遇到过这个问题了 艹，这个torch有bug
    os.system("CUDA_VISIBLE_DEVICES={} python test/par_worker.py {} {}".format(i % 4, model_name, i % 4))

def main():
    assert torch.cuda.is_available()
    pool = multiprocessing.Pool(4)
    t1 = time.time()
    rets = []
    for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
        model_name = "SBNetwork" + str(i)
        rets.append(pool.apply_async(sb, (model_name, i) ))
    pool.close()
    pool.join()
    t2 = time.time()
    # print("All works finished: ", t2-t1)

if __name__ == '__main__':
    main()

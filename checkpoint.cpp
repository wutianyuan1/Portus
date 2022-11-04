#include "checkpoint.h"
#include "gpu_direct_rdma_access.h"
#include "raw_tensor.h"
#include "utils.h"
#include "rdma_wrap.h"

#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

Client *client;

int init_checkpoint(std::string name, torch_network_t& network) {
    client = new Client("192.168.10.4", 12345);
    client->register_network(name, network);

    return 0;
}

int checkpoint(bool async){
    return client->transmit(async);
}

int restore(){
    return client->receive();
}

int wait_checkpoint_done(){
    return client->wait(1);
}

void optimizer_step(){
    // sock_recv();
    // optimizer.step();
}
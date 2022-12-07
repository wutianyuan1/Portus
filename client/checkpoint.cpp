/**
 * @file checkpoint.cpp
 * @brief Implementation of Portus client checkpoint&restore interface
 * @author madoka, stevelee477
 */
#include "checkpoint.h"
#include "../common/gpu_direct_rdma_access.h"
#include "raw_tensor.h"
#include "../common/utils.h"
#include "rdma_wrap.h"

#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

Client *client;

int init_checkpoint(std::string name, torch_network_t& network, std::string ib_local_addr, std::string server_addr) {
    client = new Client(server_addr, 12345, ib_local_addr);
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
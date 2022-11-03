#pragma once

#include "common.h"
#include "chksystem.h"
#include "khash.h"
#include "utils.h"
#include "gpu_direct_rdma_access.h"
#include "tpool.h"
#include "cqueue.h"

extern "C" {
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
}

#define ERROR_EXIT(msg) do { std::cerr << msg; return -1; } while (0);

enum job_type {
    pmem_to_gpu = 0,
    gpu_to_pmem = 1
};

class CheckpointServer {
public:
    CheckpointServer(std::string host, int sockfd, std::shared_ptr<CheckpointSystem> chksystem);
    ~CheckpointServer();

    int checkpoint();
    int restore();
    int init_chekcpoint_system();
    int open_server_socket();

    const int get_sock_fd() const;

private:
    int rdma_step();
    int add_rdma_task(byte_t* pmem_layer_buff, size_t layer_size, int wr_id, std::string desc_str);

private:
    int _port;
    int _chkpt_idx;
    int _sockfd;
    job_type _job_type; // type=0: restore, PMEM->GPU; type=1: checkpoint, GPU->PMEM
    struct rdma_device* _rdma_dev;
    std::shared_ptr<CheckpointSystem> _chksystem;
    std::vector<std::shared_ptr<rdma_task_attr> > _rdma_tasks;
    std::vector<rdma_exec_params*> _exec_tasks;
};
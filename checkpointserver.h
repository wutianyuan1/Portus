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

#define ACK_MSG "rdma_task completed"
#define ERROR_EXIT(msg) do { std::cerr << msg; return -1; } while (0);

class CheckpointServer {
public:
    CheckpointServer(std::string host, int sockfd, std::shared_ptr<CheckpointSystem> chksystem);
    // ~CheckpointServer()==default;

    int checkpoint_step();
    int init_chekcpoint_system();
    int open_server_socket();

    const int get_sock_fd() const;

private:
    int _port;
    int _chkpt_idx;
    int _sockfd;
    struct rdma_device* _rdma_dev;
    std::shared_ptr<CheckpointSystem> _chksystem;
    std::vector<std::shared_ptr<rdma_task_attr> > _rdma_tasks;
    std::vector<rdma_exec_params*> _exec_tasks;
};
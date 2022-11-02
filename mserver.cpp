#include "common.h"
#include "chksystem.h"
#include "argparse.h"
#include "khash.h"
#include "utils.h"
#include "gpu_direct_rdma_access.h"
#include "tpool.h"
#include "cqueue.h"
#include "checkpointserver.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

extern "C" {
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
}


#define ACK_MSG "rdma_task completed"
#define ERROR_EXIT(msg) do { std::cerr << msg; return -1; } while (0);

static volatile int keep_running = 1;
// extern int debug;

// int debug = 1;

void sigint_handler(int dummy) {
    keep_running = 0;
}

void workerThread(int tid, user_params params, std::shared_ptr<ConcurrentQueue<int>> q, std::shared_ptr<CheckpointSystem> chksystem) {
    std::stringstream thread_prefix;
    thread_prefix << "[Thread " << std::setw(2) << tid << "]";
    std::cout << thread_prefix.str() << " Worker init\n";
    while (true) {
        int client_fd;
        q->pop(client_fd);
        // exit
        if (client_fd == -1) {
            std::cout << "Bye!\n";
            return;
        }
        std::cout << thread_prefix.str() <<" Get connection\n";
        CheckpointServer chkserver(params.hostaddr, client_fd, chksystem);
        
        // Init checkpoint system
        if (chkserver.init_chekcpoint_system() < 0) {
            std::cerr << "Fail to initialize checkpointing system on " << params.dax_device << "\n";
            continue;
        }

        int one = 1;

        // setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        int req = 0, ret = 0;
        do {
            recv(client_fd, &req, sizeof(req), MSG_WAITALL);
            if (req == 1){
                using namespace std::chrono;
                auto t1 = high_resolution_clock::now();
                chkserver.checkpoint_step();
                auto t2 = high_resolution_clock::now();
                std::cout << "Time:" << (double)duration_cast<microseconds>(t2 - t1).count() / 1000000.0 << "\n";
            }
                
            else break;
            req = 0;
        } while (keep_running);

        std::cout << "Fin\n";
    }
}

int open_server_socket(int _port) {
    struct addrinfo *res, *t;
    struct addrinfo hints = {
        .ai_flags    = AI_PASSIVE,
        .ai_family   = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM
    };
    char* service;
    char err_info[256];
    int ret_val;
    int sockfd;
    int tmp_sockfd = -1;

    ret_val = asprintf(&service, "%d", _port);
    ret_val = getaddrinfo(NULL, service, &hints, &res);
    if (ret_val < 0) {
        sprintf(err_info, "%s for port %d\n", gai_strerror(ret_val), _port);
        free(service);
        ERROR_EXIT(err_info);
    }

    for (t = res; t; t = t->ai_next) {
        tmp_sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (tmp_sockfd >= 0) {
            int optval = 1;

            setsockopt(tmp_sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof optval);

            if (!bind(tmp_sockfd, t->ai_addr, t->ai_addrlen))
                break;
            close(tmp_sockfd);
            tmp_sockfd = -1;
        }
    }
    freeaddrinfo(res);
    free(service);
    if (tmp_sockfd < 0) {
        sprintf(err_info, "Couldn't listen to port %d\n", _port);
        ERROR_EXIT(err_info);
    }
    listen(tmp_sockfd, 1);
    return tmp_sockfd;
}

int main(int argc, char *argv[]) {
    char err_info[256];
    
    user_params params;
    if (parse_command_line(argc, argv, &params))
        return 1;

    ThreadPool pool(params.worker);

    auto q = std::make_shared<ConcurrentQueue<int>>();
    auto chksystem = std::shared_ptr<CheckpointSystem>(
        new CheckpointSystem(params.dax_device, params.pmem_size, params.init));

    for (int i = 0; i < params.worker; i++) {
        pool.enqueue([&, params=params, i=i] {
            workerThread(i, params, q, chksystem);
        });
    }

    // listen port
    int sockfd = open_server_socket(params.port);

    while (true) {
        int newconnection = accept(sockfd, NULL, 0);
        if (sockfd < 0) {
            sprintf(err_info, "accept() failed\n");
            ERROR_EXIT(err_info);
            goto CLEANUP;
        }
        q->push(newconnection);
    }

    close(sockfd);

CLEANUP:

    // Make all thread return
    for (int i = 0; i < params.worker; i++) {
        q->push(-1);
    }
}
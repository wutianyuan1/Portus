#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "gpu_direct_rdma_access.h"
#include "raw_tensor.h"

extern "C" {
    #include <sys/socket.h>
    #include <netdb.h>
    #include <netinet/tcp.h>
}

const int TASK_READ = 1;
const int TASK_WRITE = 0;

class Client {
public:
    Client(std::string _server_name, int _port, std::string local_addr) : server_name(_server_name), port(_port), need_wait(false) {
        get_addr(const_cast<char*>(local_addr.c_str()), (struct sockaddr *)&hostaddr);
        sockfd = open_client_socket(server_name.c_str(), port);
        int one = 1;
        setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        if (sockfd < 0) {
	        std::cout << "Fuck no socket\n";
            // TODO: error handling
        }

        rdma_dev = rdma_open_device_client(&hostaddr);
        if (!rdma_dev) {
            std::cout << "Fuck no IB device\n";
            //TODO: error handling
        }
    }
    ~Client() {
        for (auto&& buf : buffers) {
            rdma_buffer_dereg(buf.rdma_buffer);
            // free(buf.package);
        }

        close(sockfd);
        rdma_close_device(rdma_dev);
    }
    int register_network(std::string name, torch_network_t& network);
    int transmit(bool async=false); // 摆烂transmit
    int receive();
    int wait(int expect_msg);

private:
    int register_var(std::string name, void *addr, size_t length);
    struct rdma_buf{
        std::string name;
        size_t size;
        struct rdma_buffer *rdma_buffer;
        std::string desc_str;
        std::string task_opt_str;
        int package_size;
        void *package;
        int buff_package_size;
    };
    std::vector<rdma_buf> buffers;

    std::string server_name;
    int port;
    struct sockaddr hostaddr;
    int sockfd;
    bool need_wait;

    struct rdma_device *rdma_dev;
};

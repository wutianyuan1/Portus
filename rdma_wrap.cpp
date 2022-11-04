#include "rdma_wrap.h"

#include <sstream>
#include <chrono>

#define ACK_MSG "rdma_task completed"

const int CHECKPOINT_MSG = 1;
const int RESTORE_MSG = 2;
const int SUCCESS_MSG = 1;
const int INIT_ALREADY_HAVE_ACK = 0;
const int INIT_NEW_ACK = 1;

int Client::register_network(std::string name, torch_network_t& network) {
    auto raw_network = get_raw_network(network);
    std::stringstream ss;
    ss << name << " " << raw_network.size() << " ";
    // write(sockfd, ss.str().c_str(), ss.str().size());
    for (auto&& kv: raw_network){
        // std::cout << kv.first << std::endl;
        auto &tensor = kv.second;
        register_var(kv.first, static_cast<void*>(tensor->addr), tensor->size_in_bytes);
    }
    
    for (auto&& buf : buffers) {
        ss << buf.name << " " << buf.size << " " << buf.desc_str << " ";
    }
    int size = ss.str().size();
    write(sockfd, &size, sizeof(size));
    write(sockfd, ss.str().c_str(), ss.str().size());

    int acki;
    if (read(sockfd, &acki, sizeof(acki)) != sizeof(acki)) {
        std::cout << "Error: acki" << std::endl;
    }

    if (acki == INIT_ALREADY_HAVE_ACK) {
        std::cout << "Already have checkpoint" << std::endl;
    } else if (acki == INIT_NEW_ACK) {
        std::cout << "New checkpoint" << std::endl;
    } else {
        std::cout << "Error: acki" << std::endl;
    }

    return 0;
}

int Client::register_var(std::string name, void *addr, size_t length) {
    struct rdma_buffer *rdma_buff = rdma_buffer_reg(rdma_dev, addr, length);
    char desc_str[256], task_opt_str[16];

    int ret_desc_str_size = rdma_buffer_get_desc_str(rdma_buff, desc_str, sizeof(desc_str));
    int ret_task_opt_str_size = rdma_task_attr_flags_get_desc_str(TASK_READ, task_opt_str, sizeof(task_opt_str));
    // printf("%s\n", desc_str);

    /* Package memory allocation */
    const int package_size = (ret_desc_str_size + ret_task_opt_str_size) * sizeof(char) + 2 * sizeof(uint16_t) + 2 * sizeof(uint8_t);
    void *package = malloc(package_size);
    memset(package, 0, package_size);

    /* Packing RDMA buff desc str */
    struct payload_attr pl_attr = { .data_t = RDMA_BUF_DESC, .payload_str = desc_str };
    int buff_package_size = pack_payload_data(package, package_size, &pl_attr);
    if (!buff_package_size) {
        //TODO
    }
    
    /* Packing RDMA task attrs desc str */
    pl_attr.data_t = TASK_ATTRS;
    pl_attr.payload_str = task_opt_str;
    buff_package_size += pack_payload_data(package + buff_package_size, package_size, &pl_attr);
    if (!buff_package_size) {
        //TODO
    }

    buffers.emplace_back((rdma_buf){
        .name = name,
        .size = length,
        .rdma_buffer = rdma_buff,
        .desc_str = desc_str,
        .task_opt_str = task_opt_str,
        .package = package,
        .buff_package_size = buff_package_size
    });

    return 0;
}

int Client::transmit() {
    int  ret_size;
    int cnt = 0;
    int i = 1;
    int acki;

    ret_size = write(sockfd, &CHECKPOINT_MSG, sizeof(CHECKPOINT_MSG));
    if (ret_size != sizeof(i)) {
        // TODO
    }
    
    // Wating for confirmation message from the socket that rdma_read/write from the server has beed completed
    ret_size = recv(sockfd, &acki, sizeof acki, MSG_WAITALL);

    // std::cout << "Received ack length: " << ret_size << std::endl;
    // std::cout << "Received ack: " << (int)acki << std::endl;
    if (ret_size != sizeof SUCCESS_MSG) {
        // TODO
	// std::cout << "Received not enough ack: " << ret_size << std::endl;
    } else if (acki != SUCCESS_MSG) {
        // TODO
    }
    
    return 0;
}

int Client::receive() {
    int  ret_size;
    int cnt = 0;
    int i = 1;
    int acki;

    int one = 1;

    ret_size = write(sockfd, &RESTORE_MSG, sizeof(RESTORE_MSG));
    if (ret_size != sizeof(RESTORE_MSG)) {
        // TODO
    }
    
    // Wating for confirmation message from the socket that rdma_read/write from the server has beed completed
    ret_size = recv(sockfd, &acki, sizeof acki, MSG_WAITALL);

    // std::cout << "Received ack length: " << ret_size << std::endl;
    // std::cout << "Received ack: " << (int)acki << std::endl;
    if (ret_size != sizeof acki) {
        // TODO
    // std::cout << "Received not enough ack: " << ret_size << std::endl;
    }
    
    return 0;
}
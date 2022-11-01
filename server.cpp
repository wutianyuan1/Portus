#include "common.h"
#include "chksystem.h"
#include "argparse.h"
#include "khash.h"
#include "utils.h"
#include "gpu_direct_rdma_access.h"
#include <chrono>

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


class CheckpointServer {
public:
    CheckpointServer(user_params& params);
    ~CheckpointServer();

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


CheckpointServer::CheckpointServer(user_params& params)
: _port(params.port), _chkpt_idx(0) {
    struct sockaddr hostaddr;
    get_addr((char*)params.hostaddr.c_str(), (struct sockaddr *)&hostaddr);
    _rdma_dev = rdma_open_device_server(&hostaddr);
    if (!_rdma_dev) 
        throw std::runtime_error("Cannot open RDMA device\n");

    struct sigaction act;
    act.sa_handler = sigint_handler;
    sigaction(SIGINT, &act, NULL);

    _chksystem = std::shared_ptr<CheckpointSystem>(
        new CheckpointSystem(params.dax_device, params.pmem_size, params.init));
};


CheckpointServer::~CheckpointServer(){
    close(_sockfd);
    for (auto&& task : _rdma_tasks) {
        rdma_buffer_dereg(task->local_buf_rdma);
        delete[] task->remote_buf_desc_str;
    }
    rdma_close_device(_rdma_dev);
}


int
CheckpointServer::open_server_socket() {
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
    _sockfd = accept(tmp_sockfd, NULL, 0);
    close(tmp_sockfd);
    if (_sockfd < 0) {
        sprintf(err_info, "accept() failed\n");
        ERROR_EXIT(err_info);
    }

    return _sockfd;
}


int
CheckpointServer::init_chekcpoint_system() {
    int msg_size = 0,  ret = 0;
    // receive the size of DNN structure info
    ret = recv(_sockfd, &msg_size, sizeof(msg_size), MSG_WAITALL);
    if (ret != sizeof(int))
        ERROR_EXIT("Failed to recv DNN structure info size\n");

    // receive the DNN structure info string
    std::string msg_str(msg_size, (char)0);
    ret = recv(_sockfd, &msg_str[0], msg_size, MSG_WAITALL);
    if (ret != msg_size)
        ERROR_EXIT("Failed to recv DNN structure info string\n");
    

    // parse the info string, and construct indexes on PMEM
    std::stringstream ss(msg_str);
    std::string str_nlayers, chkpt_name;
    ss >> chkpt_name >> str_nlayers;
    int nlayers = std::stoi(str_nlayers);
    printf("New checkpoint! name=%s, layers=%d\n", chkpt_name.c_str(), nlayers);
    _chksystem->new_chkpt(chkpt_name, nlayers);

    for (int i = 0; i < nlayers; i++) {
        // parse layer_name, layer_size  and desc_str
        std::string layer_name, str_layer_size, desc_str;
        ss >> layer_name >> str_layer_size >> desc_str;
        size_t layer_size = std::stol(str_layer_size);

        // register layer
        _chksystem->register_network_layer(chkpt_name, layer_name, layer_size);
        byte_t* pmem_layer_buff = _chksystem->get_pmem_addr(chkpt_name, layer_name);
        memset(pmem_layer_buff, 0, layer_size);

        // register RDMA buffer
        struct rdma_buffer *rdma_buff;
        rdma_buff = rdma_buffer_reg(_rdma_dev, pmem_layer_buff, layer_size);
        if (!rdma_buff)
            ERROR_EXIT("Cannot register RDMA buffer\n");

        // construct the RDMA task struct
        std::shared_ptr<rdma_task_attr> layer_task(new rdma_task_attr);
        memset(&(*layer_task), 0, sizeof(rdma_task_attr));
        layer_task->remote_buf_desc_length   = sizeof("0102030405060708:01020304:01020304:0102:010203:1:0102030405060708090a0b0c0d0e0f10");
        layer_task->remote_buf_desc_str      = new char[layer_task->remote_buf_desc_length];
        layer_task->local_buf_rdma           = rdma_buff;
        layer_task->flags                    = 0x1;     // RDMA read
        layer_task->wr_id                    = i;       // use the layer id to be wr_id
        memcpy(layer_task->remote_buf_desc_str, &(desc_str[0]), layer_task->remote_buf_desc_length);
        rdma_exec_params* exec_task_params = rdma_get_exec_params(&(*layer_task));

        // Add this task to task buffer
        _rdma_tasks.push_back(layer_task);
        _exec_tasks.push_back(exec_task_params);
    }
    printf("Network structure inited\n");
    return 0;
}


int
CheckpointServer::checkpoint_step() {
    char err_info[256];
    printf("checkpoint step %d\n", _chkpt_idx++);
    for (auto&& task : _exec_tasks) {
        if (rdma_exec_task(task) ){
            sprintf(err_info, "Submit RDMA task failed\n");
            ERROR_EXIT(err_info);
        }
    }

     /* Completion queue polling loop */
    std::vector<rdma_completion_event> rdma_comp_ev(_exec_tasks.size());
    int    reported_ev  = 0;
    do {
        reported_ev += rdma_poll_completions(_rdma_dev, &rdma_comp_ev[reported_ev], _exec_tasks.size()/*expected_comp_events-reported_ev*/);
        //TODO - we can put sleep here
    } while (reported_ev < _exec_tasks.size());

    for (int i = 0; i < reported_ev; ++i) {
        if (rdma_comp_ev[i].status != (rdma_completion_status)IBV_WC_SUCCESS) {
            sprintf(err_info, "FAILURE: status \"%s\" (%d) for wr_id %d\n",
                    ibv_wc_status_str((ibv_wc_status)rdma_comp_ev[i].status),
                    rdma_comp_ev[i].status, (int) rdma_comp_ev[i].wr_id);
            ERROR_EXIT(err_info);
        }
    }

    // Sending ack-message to the client, confirming that RDMA read/write has been completet
    int ackmsg = 1;
    if (write(_sockfd, &ackmsg, sizeof(ackmsg)) != sizeof(ackmsg)) {
        sprintf(err_info, "FAILURE: Couldn't send \"%c\" msg (errno=%d '%m')\n", ACK_MSG, errno);
        ERROR_EXIT(err_info);
    }

    return 0;
}


const int
CheckpointServer::get_sock_fd() const {
    return _sockfd;
}


int main(int argc, char *argv[]) {
    user_params params;
    if (parse_command_line(argc, argv, &params))
        return 1;
    
    // Construct a checkpointing server
    CheckpointServer ser(params);

    // Open socket for connection
    if (ser.open_server_socket() < 0) {
        std::cerr << "Cannot open server socket\n";
        return 1;
    }

    printf("Connection accepted.\n");

    // Init checkpoint system
    if (ser.init_chekcpoint_system() < 0) {
        std::cerr << "Fail to initialize checkpointing system on " << params.dax_device << "\n";
        return 1;
    }

    int one = 1;
   
    setsockopt(ser.get_sock_fd(), IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int req = 0, ret = 0;
    do {
        recv(ser.get_sock_fd(), &req, sizeof(req), MSG_WAITALL);
        if (req == 1){
            using namespace std::chrono;
            auto t1 = high_resolution_clock::now();
            ser.checkpoint_step();
            auto t2 = high_resolution_clock::now();
            std::cout << "Time:" << (double)duration_cast<microseconds>(t2 - t1).count() / 1000000.0 << "\n";
        }
            
        else break;
        req = 0;
    } while (keep_running);

    return 0;
}


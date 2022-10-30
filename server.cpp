#include "common.h"
#include "chksystem.h"
#include "argparse.h"
#include "khash.h"
#include "utils.h"
#include "gpu_direct_rdma_access.h"
#include <chrono>

#define ACK_MSG "rdma_task completed"

extern int debug;
extern int debug_fast_path;

#define DEBUG_LOG if (debug) printf
#define DEBUG_LOG_FAST_PATH if (debug_fast_path) printf
#define FDEBUG_LOG if (debug) fprintf
#define FDEBUG_LOG_FAST_PATH if (debug_fast_path) sprintf
#define SDEBUG_LOG if (debug) fprintf
#define SDEBUG_LOG_FAST_PATH if (debug_fast_path) sprintf

static volatile int keep_running = 1;

int debug = 1;
int debug_fast_path = 1;


void sigint_handler(int dummy) {
    keep_running = 0;
}


class CheckpointServer {
public:
    CheckpointServer(user_params& params);
    ~CheckpointServer();

    void checkpoint_step();
    void init_chekcpoint_system();
    int open_server_socket();
    int sock_fd();

private:
    int _port;
    int _chkpt_idx;
    int _sockfd;
    struct rdma_device* _rdma_dev;
    std::shared_ptr<CheckpointSystem> _chksystem;
    std::vector<std::shared_ptr<rdma_task_attr> > _rdma_tasks;
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
    printf("pool start addr: %p\n", _chksystem->_pool.base_addr());
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
    int ret_val;
    int sockfd;
    int tmp_sockfd = -1;

    ret_val = asprintf(&service, "%d", _port);
    if (ret_val < 0)
        return -1;

    ret_val = getaddrinfo(NULL, service, &hints, &res);
    if (ret_val < 0) {
        fprintf(stderr, "%s for port %d\n", gai_strerror(ret_val), _port);
        free(service);
        return -1;
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
        fprintf(stderr, "Couldn't listen to port %d\n", _port);
        return -1;
    }
    listen(tmp_sockfd, 1);
    _sockfd = accept(tmp_sockfd, NULL, 0);
    close(tmp_sockfd);
    if (_sockfd < 0) {
        fprintf(stderr, "accept() failed\n");
        return -1;
    }
    printf("Connection accepted.\n");
    return _sockfd;
}


void
CheckpointServer::init_chekcpoint_system() {
    int msg_size = 0;
    recv(_sockfd, &msg_size, sizeof(msg_size), MSG_WAITALL);
    std::string msg_str(msg_size, (char)0);
    recv(_sockfd, &msg_str[0], msg_size, MSG_WAITALL);
    std::stringstream ss(msg_str);

    // checkpoint name, nlayers
    std::string str_nlayers, chkpt_name;
    ss >> chkpt_name >> str_nlayers;
    int nlayers = std::stoi(str_nlayers);
    _chksystem->new_chkpt(chkpt_name, nlayers);
    
    for (int i = 0; i < nlayers; i++) {
        // layer_name, layer_size
        std::string layer_name, str_layer_size, desc_str;
        ss >> layer_name >> str_layer_size >> desc_str;
        printf("!!!! layer_name: %s, size: %s, desc: %s\n", layer_name.c_str(), str_layer_size.c_str(), desc_str.c_str());
        size_t layer_size = std::stol(str_layer_size);
        _chksystem->register_network_layer(chkpt_name, layer_name, layer_size);
        byte_t* pmem_layer_buff = _chksystem->get_pmem_addr(chkpt_name, layer_name);
        memset(pmem_layer_buff, 0, layer_size);

        struct rdma_buffer *rdma_buff;
        rdma_buff = rdma_buffer_reg(_rdma_dev, pmem_layer_buff, layer_size);


        std::shared_ptr<rdma_task_attr> layer_task(new rdma_task_attr);
        memset(&(*layer_task), 0, sizeof(rdma_task_attr));
        layer_task->remote_buf_desc_length   = sizeof("0102030405060708:01020304:01020304:0102:010203:1:0102030405060708090a0b0c0d0e0f10");
        layer_task->remote_buf_desc_str      = new char[layer_task->remote_buf_desc_length];
        layer_task->local_buf_rdma           = rdma_buff;
        layer_task->flags                    = 0x1;     // RDMA read
        layer_task->wr_id                    = i;       // use the layer id to be wr_id
        memcpy(layer_task->remote_buf_desc_str, &(desc_str[0]), layer_task->remote_buf_desc_length);

        _rdma_tasks.push_back(layer_task);
    }
}


void
CheckpointServer::checkpoint_step() {
    printf(">>> checkpoint job starts\n");
    for (auto&& rdma_task : _rdma_tasks) {
        printf("@@ task attrs: %s %d %p", rdma_task->remote_buf_desc_str, rdma_task->remote_buf_desc_length, rdma_task->local_buf_rdma);
        if (rdma_submit_task(&(*rdma_task)) )
            throw std::runtime_error("Submit RDMA task failed\n");
    }

     /* Completion queue polling loop */
    DEBUG_LOG_FAST_PATH("Polling completion queue\n");
    struct rdma_completion_event rdma_comp_ev[10];
    int    reported_ev  = 0;
    do {
        reported_ev += rdma_poll_completions(_rdma_dev, &rdma_comp_ev[reported_ev], 10/*expected_comp_events-reported_ev*/);
        //TODO - we can put sleep here
    } while (reported_ev < _rdma_tasks.size());
    DEBUG_LOG_FAST_PATH("Finished polling\n");

    for (int i = 0; i < reported_ev; ++i) {
        printf("%d\n", i);
        if (rdma_comp_ev[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "FAILURE: status \"%s\" (%d) for wr_id %d\n",
                    ibv_wc_status_str((ibv_wc_status)rdma_comp_ev[i].status),
                    rdma_comp_ev[i].status, (int) rdma_comp_ev[i].wr_id);
            throw std::runtime_error("nmsl\n");
        }
    }

    // Sending ack-message to the client, confirming that RDMA read/write has been completet
    if (write(_sockfd, ACK_MSG, sizeof(ACK_MSG)) != sizeof(ACK_MSG)) {
        char err_info[] = "FAILURE: Couldn't send \"%c\" msg (errno=%d '%m')\n";
        sprintf(err_info, ACK_MSG, errno);
        throw std::runtime_error(std::string(err_info));
    }
}

int
CheckpointServer::sock_fd() {
    return _sockfd;
}


int main(int argc, char *argv[]) {
    user_params params;
    if (parse_command_line(argc, argv, &params))
        return 1;
    printf("%s %s %d %d %d\n", params.dax_device.c_str(), params.hostaddr.c_str(), params.port, params.pmem_size, params.init);
    CheckpointServer ser(params);
    ser.open_server_socket();
    ser.init_chekcpoint_system();

    int req = 0;
    do {
        recv(ser.sock_fd(), &req, sizeof(req), MSG_WAITALL);
        if (req == 1){
            using namespace std::chrono;
            auto t1 = high_resolution_clock::now();
            ser.checkpoint_step();
            auto t2 = high_resolution_clock::now();
            std::cout << (double)duration_cast<microseconds>(t2 - t1).count() / 1000000.0 << "\n";
        }
            
        else break;
        req = 0;
    } while (keep_running);

    return 0;
}


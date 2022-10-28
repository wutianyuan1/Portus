#include "common.h"
#include "gpu_direct_rdma/khash.h"
#include "gpu_direct_rdma/utils.h"
#include "gpu_direct_rdma/gpu_direct_rdma_access.h"


#define MAX_SGES 512
#define ACK_MSG "rdma_task completed"
#define PACKAGE_TYPES 2

extern int debug;
extern int debug_fast_path;

#define DEBUG_LOG if (debug) printf
#define DEBUG_LOG_FAST_PATH if (debug_fast_path) printf
#define FDEBUG_LOG if (debug) fprintf
#define FDEBUG_LOG_FAST_PATH if (debug_fast_path) sprintf
#define SDEBUG_LOG if (debug) fprintf
#define SDEBUG_LOG_FAST_PATH if (debug_fast_path) sprintf

static volatile int keep_running = 1;


void sigint_handler(int dummy) {
    keep_running = 0;
}


class CheckpointServer {
public:
    CheckpointServer(int port, std::string host_addr) {
        _port = port;
        _chkpt_idx = 0;
        get_addr((char*)host_addr.c_str(), (struct sockaddr *)&_hostaddr);
        _rdma_dev = rdma_open_device_server(&_hostaddr);
        if (!_rdma_dev) 
            throw std::runtime_error("Cannot open RDMA device\n");

        struct sigaction act;
        act.sa_handler = sigint_handler;
        sigaction(SIGINT, &act, NULL);

        printf("Listening to remote client...\n");
        _sockfd = open_server_socket(_port);
        if (_sockfd < 0)
            throw std::runtime_error("Cannot open server socket\n");
        printf("Connection accepted.\n");
    };

    ~CheckpointServer() {
        rdma_close_device(_rdma_dev);
    }

    void init_chekcpoint_system(){

    }

    void checkpoint_step();

private:
    int open_server_socket(int port);

private:
    int _persistent;
    int _port;
    struct sockaddr _hostaddr;
    struct rdma_device* _rdma_dev;
    int _sockfd;
    int _chkpt_idx;
};


int CheckpointServer::open_server_socket(int port) {
    struct addrinfo *res, *t;
    struct addrinfo hints = {
        .ai_flags    = AI_PASSIVE,
        .ai_family   = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM
    };
    char   *service;
    int     ret_val;
    int     _sockfd;
    int     tmp_sockfd = -1;

    ret_val = asprintf(&service, "%d", port);
    if (ret_val < 0)
        return -1;

    ret_val = getaddrinfo(NULL, service, &hints, &res);
    if (ret_val < 0) {
        fprintf(stderr, "%s for port %d\n", gai_strerror(ret_val), port);
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
        fprintf(stderr, "Couldn't listen to port %d\n", port);
        return -1;
    }

    listen(tmp_sockfd, 1);
    _sockfd = accept(tmp_sockfd, NULL, 0);
    close(tmp_sockfd);
    if (_sockfd < 0) {
        fprintf(stderr, "accept() failed\n");
        return -1;
    }
    return _sockfd;
}


void CheckpointServer::checkpoint_step() {
    int                            r_size;
    char                           desc_str[sizeof "0102030405060708:01020304:01020304:0102:010203:1:0102030405060708090a0b0c0d0e0f10"];
    char                           ackmsg[sizeof ACK_MSG];
    struct rdma_task_attr          task_attr;
    int                            i;
    uint32_t                       flags; /* Use enum rdma_task_attr_flags */
    // payload attrs
    uint8_t                        pl_type;
    uint16_t                       pl_size; 
    //int     expected_comp_events = usr_par.num_sges? (usr_par.num_sges+MAX_SEND_SGE-1)/MAX_SEND_SGE: 1;
    
    for (i = 0; i < PACKAGE_TYPES; i++) {
        r_size = recv(_sockfd, &pl_type, sizeof(pl_type), MSG_WAITALL);
        r_size = recv(_sockfd, &pl_size, sizeof(pl_size), MSG_WAITALL);
        switch (pl_type) {
            case 0: // RDMA_BUF_DESC
                /* Receiving RDMA data (address, size, rkey etc.) from socket as a triger to start RDMA Read/Write operation */
                DEBUG_LOG_FAST_PATH("Iteration %d: Waiting to Receive message of size %lu\n", _chkpt_idx, sizeof desc_str);   
                r_size = recv(_sockfd, desc_str, pl_size * sizeof(char), MSG_WAITALL);
                if (r_size != sizeof desc_str) {
                    fprintf(stderr, "FAILURE: Couldn't receive RDMA data for iteration %d (errno=%d '%m')\n", _chkpt_idx, errno);
                }
                break;
            case 1: // TASK_ATTRS
                /* Receiving rw attr flags */;
                int s = pl_size * sizeof(char);
                char t[16];
                r_size = recv(_sockfd, &t, s, MSG_WAITALL);
                if (r_size != s) {
                    fprintf(stderr, "FAILURE: Couldn't receive RDMA data for iteration %d (errno=%d '%m')\n", _chkpt_idx, errno);
                }
                sscanf(t, "%08x", &flags);
                break;
        }
    }
    
    DEBUG_LOG_FAST_PATH("Received message \"%s\"\n", desc_str);
    memset(&task_attr, 0, sizeof task_attr);
    task_attr.remote_buf_desc_str      = desc_str;
    task_attr.remote_buf_desc_length   = sizeof desc_str;
    task_attr.local_buf_rdma           = _rdma_buff;
    task_attr.flags                    = flags;
    task_attr.wr_id                    = cnt;// * expected_comp_events;

    /* Executing RDMA read */
    SDEBUG_LOG_FAST_PATH ((char*)buff, "Read iteration N %d", cnt);

    if (rdma_submit_task(&task_attr))
        throw std::runtime_error("Submit RDMA task failed\n");

/* Completion queue polling loop */
    DEBUG_LOG_FAST_PATH("Polling completion queue\n");
    struct rdma_completion_event rdma_comp_ev[10];
    int    reported_ev  = 0;
    do {
        reported_ev += rdma_poll_completions(rdma_dev, &rdma_comp_ev[reported_ev], 10/*expected_comp_events-reported_ev*/);
        //TODO - we can put sleep here
    } while (reported_ev < 1 && keep_running /*expected_comp_events*/);
    DEBUG_LOG_FAST_PATH("Finished polling\n");

    for (i = 0; i < reported_ev; ++i) {
        if (rdma_comp_ev[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "FAILURE: status \"%s\" (%d) for wr_id %d\n",
                    ibv_wc_status_str((ibv_wc_status)rdma_comp_ev[i].status),
                    rdma_comp_ev[i].status, (int) rdma_comp_ev[i].wr_id);
            if (usr_par.persistent && keep_running) {
                rdma_reset_device(rdma_dev);
            }
            goto clean_socket;
        }
    }

    // Sending ack-message to the client, confirming that RDMA read/write has been completet
    if (write(_sockfd, ACK_MSG, sizeof(ACK_MSG)) != sizeof(ACK_MSG)) {
        fprintf(stderr, "FAILURE: Couldn't send \"%c\" msg (errno=%d '%m')\n", ACK_MSG, errno);
        goto clean_socket;
    }
}


int main(int argc, char *argv[]) {

    return 0;
}


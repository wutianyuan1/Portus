/**
 * @file checkpointserver.cpp
 * @brief Implementation of Portus checkpointing server
 * @author madoka, stevelee477
 */
#include "checkpointserver.h"

CheckpointServer::CheckpointServer(std::string host, int sockfd, std::shared_ptr<CheckpointSystem> chksystem)
: _chkpt_idx(0), _sockfd(sockfd), _job_type(job_type::gpu_to_pmem), _chksystem(chksystem) {
    struct sockaddr hostaddr;
    get_addr((char*)host.c_str(), (struct sockaddr *)&hostaddr);
    _rdma_dev = rdma_open_device_server(&hostaddr);
    if (!_rdma_dev) 
        throw std::runtime_error("Cannot open RDMA device\n");
}

CheckpointServer::~CheckpointServer(){
    close(_sockfd);
    for (auto&& task : _rdma_tasks) {
        rdma_buffer_dereg(task->local_buf_rdma);
        delete[] task->remote_buf_desc_str;
    }
    rdma_close_device(_rdma_dev);
}

int
CheckpointServer::add_rdma_task(byte_t* pmem_layer_buff, size_t layer_size, int wr_id, std::string desc_str) {
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
    layer_task->flags                    = uint32_t(_job_type);     // RDMA read
    layer_task->wr_id                    = wr_id;       // use the layer id to be wr_id
    memcpy(layer_task->remote_buf_desc_str, &(desc_str[0]), layer_task->remote_buf_desc_length);
    rdma_exec_params* exec_task_params = rdma_get_exec_params(&(*layer_task));

    // Add this task to task buffer
    _rdma_tasks.push_back(layer_task);
    _exec_tasks.push_back(exec_task_params);
    return 0;
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
    int ackmsg = 0;

    // check if the checkpoint is already in system
    auto chkpt_ptr = _chksystem->get_chkpt(chkpt_name);
    if (chkpt_ptr != nullptr) {
        chkpt_ptr->load_params(&_chksystem->_pool);
        auto pmem_layers = chkpt_ptr->get_layers_info();
        std::map<std::string, size_t> layer_map(pmem_layers.begin(), pmem_layers.end());
        for (int i = 0; i < nlayers; i++) {
            // parse layer_name, layer_size  and desc_str
            std::string layer_name, str_layer_size, desc_str;
            ss >> layer_name >> str_layer_size >> desc_str;
            size_t layer_size = std::stol(str_layer_size);
            if (layer_size != layer_map[layer_name]) {
                printf("DNN structure not match! layer %s: size %ld!=%ld\n", layer_name.c_str(), layer_size, layer_map[layer_name]);
                return 1;
            }
            byte_t* pmem_layer_buff = _chksystem->get_pmem_addr(chkpt_name, layer_name);

            // add RDMA task to task list
            add_rdma_task(pmem_layer_buff, layer_size, i, desc_str);
        }

        ackmsg = RESTORE_FINISH_MSG;
    }
    else {
        // If no such checkpoint, create a new one!
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

            // add RDMA task to task list
            add_rdma_task(pmem_layer_buff, layer_size, i, desc_str);
        }
        ackmsg = NEW_CHKPT_FINISH_MSG;
    }
    if (write(_sockfd, &ackmsg, sizeof(int)) != sizeof(int))
        ERROR_EXIT("Cannot send finish message\n");
    printf("Network structure inited\n");
    return 0;
}


int
CheckpointServer::rdma_step() {
    char err_info[256];
    // printf("checkpoint step %d, job type %d\n", _chkpt_idx++, int(_job_type));
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
    int ackmsg = TASK_FINISH_MSG;
    if (write(_sockfd, &ackmsg, sizeof(ackmsg)) != sizeof(ackmsg)) {
        sprintf(err_info, "FAILURE: Couldn't send \"%d\" msg (errno=%d '%m')\n", TASK_FINISH_MSG, errno);
        ERROR_EXIT(err_info);
    }

    return 0;
}


int
CheckpointServer::checkpoint() {
    if (_job_type != job_type::gpu_to_pmem) {
        for (auto&& task : _rdma_tasks)
            task->flags = (uint32_t)job_type::gpu_to_pmem;
        for (auto&& exec_task : _exec_tasks)
            exec_task->flags = (uint32_t)job_type::gpu_to_pmem;
        _job_type = job_type::gpu_to_pmem;
    }
    return rdma_step();
}


int
CheckpointServer::restore() {
    if (_job_type != job_type::pmem_to_gpu) {
        for (auto&& task : _rdma_tasks)
            task->flags = (uint32_t)job_type::pmem_to_gpu;
        for (auto&& exec_task : _exec_tasks)
            exec_task->flags = (uint32_t)job_type::pmem_to_gpu;
        _job_type = job_type::pmem_to_gpu;
    }
    return rdma_step();
}

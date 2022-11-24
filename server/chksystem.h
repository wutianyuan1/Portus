#pragma once
#include "common.h"
#include "object.h"


class CheckpointSystem {
public:
    CheckpointSystem(std::string dev_name, size_t map_size, bool init=false, bool use_dram=false);
    ~CheckpointSystem();

    int new_chkpt(std::string chkpt_name, size_t nlayers);
    int remove_chkpt(std::string chkpt_name);

    int register_network_layer(std::string chkpt_name, std::string layer_name, size_t layer_size);
    int load_network_params(std::string chkpt_name);
    int chkpt_summary(std::string chkpt_name, int verbose=0);

    std::shared_ptr<PMemDNNCheckpoint> get_chkpt(std::string chkpt_name);
    std::vector<std::string> existing_chkpts();
    byte_t* get_pmem_addr(std::string chkpt_name, std::string layer_name);

private:
    bool is_valid_offset(off64_t offset);

public:
    PMemPool _pool;
    off64_t* _chkpt_table_ptr;
    off64_t _chkpt_table_offset;
    std::atomic<size_t> _n_chkpts;
    std::map<std::string, std::shared_ptr<PMemDNNCheckpoint> > _chkpts;
    std::map<std::string, int> _name_to_tableidx;
};

#include "chksystem.h"


CheckpointSystem::CheckpointSystem(std::string dev_name, size_t map_size, bool init) {
    _pool.open_pmem(dev_name, map_size, init);
    // The chkpt table always hold the first 2MB available chunk
    if (init){
        auto ret = _pool.alloc(CHKPT_TABLE_SIZE);
        _chkpt_table_offset = ret.first;
        _chkpt_table_ptr = reinterpret_cast<off64_t*>(ret.second);
        _mm_mfence();
        memset(_chkpt_table_ptr, 0, CHKPT_TABLE_SIZE);
        _mm_mfence();
        clwb(reinterpret_cast<byte_t*>(_chkpt_table_ptr), CHKPT_TABLE_SIZE);
        _mm_mfence();
    }
    else{
        _chkpt_table_offset = 0 + ALLOC_TABLE_SIZE;
        _chkpt_table_ptr = reinterpret_cast<off64_t*>(_pool.get_obj(_chkpt_table_offset));
    }
    _n_chkpts = _chkpt_table_ptr[0];
    if (_n_chkpts != 0){
#ifdef DEBUG_
        printf("We already have %d chkpts on this PMem\n", _n_chkpts);
#endif
        for (int i = 0; i < _n_chkpts; i++) {
            off64_t chkpt_offset = _chkpt_table_ptr[i + 1];
            // skip invalid (removed) checkpoints
            if (!is_valid_offset(chkpt_offset))
                continue;
            std::shared_ptr<PMemDNNCheckpoint> chkpt(new PMemDNNCheckpoint());
            chkpt->from_pmem(&_pool, chkpt_offset);
            _chkpts[chkpt->name()] = chkpt;
            _name_to_tableidx[chkpt->name()] = i + 1;
        }
    }
#ifdef DEBUG_
    else
        printf("Empty chkpt pool\n");
#endif
}

int
CheckpointSystem::new_chkpt(std::string chkpt_name, size_t nlayers) {
    _mutex.lock();
    if (_chkpts.find(chkpt_name) != _chkpts.end()) {
        std::cerr << "Checkpoint " << chkpt_name << " exists\n";
        // _mutex.unlock();
        // return 1;
    }
    _n_chkpts++;
    std::shared_ptr<PMemDNNCheckpoint> chkpt(new PMemDNNCheckpoint(chkpt_name, nlayers));
    _chkpts[chkpt_name] = chkpt;
    _name_to_tableidx[chkpt_name] = _n_chkpts;
    _mm_mfence();
    _chkpt_table_ptr[_n_chkpts] = chkpt->persist(&_pool);
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[_n_chkpts]);
    _mm_mfence();
    _chkpt_table_ptr[0] = _n_chkpts;
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[0]);
    _mutex.unlock();
    return 0;
}

bool
CheckpointSystem::is_valid_offset(off64_t offset) {
    return !((offset >> 63) & 1);
}

int
CheckpointSystem::remove_chkpt(std::string chkpt_name) {
    _mutex.lock();
    if (_name_to_tableidx.find(chkpt_name) == _name_to_tableidx.end()) {
        std::cerr << "Cannot remove " << chkpt_name << ": checkpoint does not exist\n";
        _mutex.unlock();
        return 1;
    }
    int table_idx = _name_to_tableidx[chkpt_name];
    _mm_mfence();
    // mark the left-most bit on PMEM to be 1 -> invalidate
    _chkpt_table_ptr[table_idx] = (_chkpt_table_ptr[table_idx] | ((size_t)1 << 63) );
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[table_idx]);
    _mm_mfence();
    // remove it from DRAM std::map
    _chkpts.erase(chkpt_name);
    _name_to_tableidx.erase(chkpt_name);
    _mutex.unlock();
    return 0;
}

int
CheckpointSystem::register_network_layer(std::string chkpt_name, std::string layer_name, size_t layer_size) {
    auto iter = _chkpts.find(chkpt_name);
    if (iter == _chkpts.end()) {
        std::cerr << "Register layer of " << chkpt_name << " failed: checkpoint does not exist\n";
        return 1;
    }
    (iter->second)->register_layer(layer_name, layer_size, &_pool);
    return 0;
}

int
CheckpointSystem::load_network_params(std::string chkpt_name) {
    auto iter = _chkpts.find(chkpt_name);
    if (iter == _chkpts.end()) {
        std::cerr << "Load parameters of " << chkpt_name << " failed: checkpoint does not exist\n";
        return 1;
    }
    (iter->second)->load_params(&_pool);
    return 0;
}

int
CheckpointSystem::chkpt_summary(std::string chkpt_name, int verbose) {
    auto iter = _chkpts.find(chkpt_name);
    if (iter == _chkpts.end()) {
        std::cerr << "Get summary of " << chkpt_name << " failed: checkpoint does not exist\n";
        return 1;
    }
    (iter->second)->summary(verbose);
    return 0;
}

std::shared_ptr<PMemDNNCheckpoint>
CheckpointSystem::get_chkpt(std::string chkpt_name) {
    auto iter = _chkpts.find(chkpt_name);
    if (iter == _chkpts.end()) {
        std::cerr << "Get checkpoint " << chkpt_name << " failed: checkpoint does not exist\n";
        return nullptr;
    }
    (iter->second)->load_params(&_pool);
    return iter->second;
}

std::vector<std::string>
CheckpointSystem::existing_chkpts() {
    std::vector<std::string> ret;
    std::for_each(_chkpts.begin(), _chkpts.end(), [&](auto&& item) { ret.push_back(item.first); });
    return ret;
}

byte_t* 
CheckpointSystem::get_pmem_addr(std::string chkpt_name, std::string layer_name) {
    auto iter = _chkpts.find(chkpt_name);
    if (iter == _chkpts.end()) {
        std::cerr << "Get PMEM address of " << chkpt_name << " failed: checkpoint does not exist\n";
        return nullptr;
    }
    return (iter->second)->get_layer_data(layer_name);
}

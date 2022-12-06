#include "chksystem.h"


CheckpointSystem::CheckpointSystem(std::string dev_name, size_t map_size, bool init, bool use_dram) {
    _pool.open_pmem(dev_name, map_size, init, use_dram);
    // The chkpt table always hold the first 2MB available chunk
    if (init){
        auto ret = _pool.alloc(CHKPT_TABLE_SIZE);
        _chkpt_table_offset = ret.first;
        _chkpt_table_ptr = reinterpret_cast<off64_t*>(ret.second);
        _mm_mfence();
        memset(_chkpt_table_ptr, 0, CHKPT_TABLE_SIZE);
        _chkpt_table_ptr[CHKPT_TABLE_SIZE/sizeof(off64_t) - 1] = 0xdeadbeef;
        _mm_mfence();
        clwb(reinterpret_cast<byte_t*>(_chkpt_table_ptr), CHKPT_TABLE_SIZE);
        _mm_mfence();
    }
    else{
        _chkpt_table_offset = 0 + ALLOC_TABLE_SIZE;
        _chkpt_table_ptr = reinterpret_cast<off64_t*>(_pool.get_obj(_chkpt_table_offset));
        // check if it's a valid chksystem, if fails, reopen the pool to init it
        if (_chkpt_table_ptr[CHKPT_TABLE_SIZE/sizeof(off64_t) - 1] != 0xdeadbeef) {
            // 要什么优雅，我服了，我把他复制一坨不就完了吗。。。
            std::cerr << "Invalid partition: Incorrect Magic Number" << std::endl;
            _pool.close_pmem();
            _pool.open_pmem(dev_name, map_size, true, use_dram);
            auto ret = _pool.alloc(CHKPT_TABLE_SIZE);
            _chkpt_table_offset = ret.first;
            _chkpt_table_ptr = reinterpret_cast<off64_t*>(ret.second);
            _chkpt_table_ptr[CHKPT_TABLE_SIZE/sizeof(off64_t) - 1] = 0xdeadbeef;
            _mm_mfence();
            memset(_chkpt_table_ptr, 0, CHKPT_TABLE_SIZE);
            _mm_mfence();
            clwb(reinterpret_cast<byte_t*>(_chkpt_table_ptr), CHKPT_TABLE_SIZE);
            _mm_mfence();
        }
    }
    _n_chkpts = _chkpt_table_ptr[0];
    if (_n_chkpts != 0){
#ifdef DEBUG_
        printf("We already have %d chkpts on this PMem\n", _n_chkpts.load());
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
    if (_n_chkpts.is_lock_free())
        printf("Lock free!\n");
#endif
}


CheckpointSystem::~CheckpointSystem(){
    _pool.close_pmem();
}


int
CheckpointSystem::new_chkpt(std::string chkpt_name, size_t nlayers) {
    if (_chkpts.find(chkpt_name) != _chkpts.end()) {
        std::cerr << "Checkpoint " << chkpt_name << " exists\n";
        // return 1;
    }
    size_t old_chkpts = _n_chkpts;
    _n_chkpts++;
    std::shared_ptr<PMemDNNCheckpoint> chkpt(new PMemDNNCheckpoint(chkpt_name, nlayers));
    _chkpts[chkpt_name] = chkpt;
    _name_to_tableidx[chkpt_name] = _n_chkpts;
    _mm_mfence();
    _chkpt_table_ptr[_n_chkpts] = chkpt->persist(&_pool);
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[_n_chkpts]);
    _mm_mfence();
    // _chkpt_table_ptr[0] = _n_chkpts;
    __sync_bool_compare_and_swap(&_chkpt_table_ptr[0], old_chkpts, _n_chkpts);
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[0]);
    return 0;
}

bool
CheckpointSystem::is_valid_offset(off64_t offset) {
    return !((offset >> 63) & 1);
}

int
CheckpointSystem::remove_chkpt(std::string chkpt_name) {
    if (_name_to_tableidx.find(chkpt_name) == _name_to_tableidx.end()) {
        std::cerr << "Cannot remove " << chkpt_name << ": checkpoint does not exist\n";
        return 1;
    }
    int table_idx = _name_to_tableidx[chkpt_name];
    _mm_mfence();
    // mark the left-most bit on PMEM to be 1 -> invalidate
    _chkpt_table_ptr[table_idx] = (_chkpt_table_ptr[table_idx] | ((size_t)1 << 63) );
    // decrease chkpt count
    _n_chkpts--;
    // _chkpt_table_ptr[0] = _n_chkpts;
    __sync_bool_compare_and_swap(&_chkpt_table_ptr[0], _n_chkpts + 1, _n_chkpts);
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[table_idx]);
    _mm_mfence();
    // remove it from DRAM std::map
    _chkpts.erase(chkpt_name);
    _name_to_tableidx.erase(chkpt_name);
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

PMemDNNCheckpoint
CheckpointSystem::get_chkpt_obj(std::string chkpt_name) {
    auto iter = _chkpts.find(chkpt_name);
    if (iter == _chkpts.end()) {
        std::cerr << "Get checkpoint " << chkpt_name << " failed: checkpoint does not exist\n";
        return PMemDNNCheckpoint("none", 0);
    }
    auto obj_ptr = iter->second;
    (obj_ptr)->load_params(&_pool);
    return *(obj_ptr);
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

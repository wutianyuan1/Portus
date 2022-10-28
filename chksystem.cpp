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
            std::shared_ptr<PMemDNNCheckpoint> chkpt(new PMemDNNCheckpoint());
            chkpt->from_pmem(&_pool, chkpt_offset);
            _chkpts[chkpt->name()] = chkpt;
        }
    }
#ifdef DEBUG_
    else
        printf("Empty chkpt pool\n");
#endif
}

void
CheckpointSystem::new_chkpt(std::string chkpt_name, size_t nlayers) {
    _mutex.lock();
    _n_chkpts++;
    std::shared_ptr<PMemDNNCheckpoint> chkpt(new PMemDNNCheckpoint(chkpt_name, nlayers));
    _chkpts[chkpt_name] = chkpt;
    _mm_mfence();
    _chkpt_table_ptr[_n_chkpts] = chkpt->persist(&_pool);
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[_n_chkpts]);
    _mm_mfence();
    _chkpt_table_ptr[0] = _n_chkpts;
    _mm_mfence();
    _mm_clwb(&_chkpt_table_ptr[0]);
    _mutex.unlock();
}

void
CheckpointSystem::register_network_layer(std::string chkpt_name, std::string layer_name, size_t layer_size) {
    _chkpts[chkpt_name]->register_layer(layer_name, layer_size, &_pool);
}

void
CheckpointSystem::load_network_params(std::string chkpt_name) {
    _chkpts[chkpt_name]->load_params(&_pool);
}

void
CheckpointSystem::chkpt_summary(std::string chkpt_name) {
    _chkpts[chkpt_name]->summary();
}

std::shared_ptr<PMemDNNCheckpoint>
CheckpointSystem::get_chkpt(std::string chkpt_name) {
    return _chkpts[chkpt_name];
}

std::vector<std::string>
CheckpointSystem::existing_chkpts() {
    std::vector<std::string> ret;
    std::for_each(_chkpts.begin(), _chkpts.end(), [&](auto&& item) { ret.push_back(item.first); });
    return ret;
}

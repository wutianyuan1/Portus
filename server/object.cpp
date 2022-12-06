#include "object.h"


// -------------------- PMemDNNLayer --------------------
off64_t
PMemDNNLayer::persist(PMemPool* pool) {
    // [this_size, data_size_in_bytes, data_offset, name]
    size_t space_needed = sizeof(uint64_t) + (layer_name.size() + 1) + sizeof(size_in_bytes) + sizeof(data_offset);
    std::tie(this_offset, this_ptr) = pool->alloc(space_needed);
    memset(this_ptr, 0, space_needed);
    off64_t* pmem_ptr_i64 = reinterpret_cast<off64_t*>(this_ptr);
    _mm_mfence();
    pmem_ptr_i64[0] = space_needed;
    pmem_ptr_i64[1] = size_in_bytes;
    pmem_ptr_i64[2] = data_offset;
    memcpy(this_ptr + 3*sizeof(off64_t), layer_name.c_str(), sizeof(char)*(layer_name.size() + 1));
    _mm_mfence();
    clwb(this_ptr, space_needed);
    _mm_mfence();
    return this_offset;
}

void
PMemDNNLayer::from_pmem(PMemPool* pool, off64_t offset) {
    this_offset = offset;
    this_ptr = pool->get_obj(offset);
    off64_t* obj_i64 = reinterpret_cast<off64_t*>(this_ptr);
    size_t space = obj_i64[0];
    size_in_bytes = obj_i64[1];
    data_offset = obj_i64[2];
    layer_name = std::string(reinterpret_cast<char*>(this_ptr + 3*sizeof(uint64_t)));
}

const std::string& PMemDNNLayer::name() const {
    return layer_name;
}

void
PMemDNNLayer::summary(int verbose) const {
    printf("    layer object name: %s\n", layer_name.c_str());
    printf("    data size: %ld\n", size_in_bytes);
    printf("    data_offset: %ld\n", data_offset);
}



// -------------------- PMemDNNLayer --------------------
// New checkpoint
PMemDNNCheckpoint::PMemDNNCheckpoint() 
    : _chkpt_name("None"), _nlayers(0), _cur_layer_idx(0) {}

PMemDNNCheckpoint::PMemDNNCheckpoint(std::string chkpt_name, int nlayers)
    : _chkpt_name(chkpt_name), _nlayers(nlayers), _cur_layer_idx(0) {}

PMemDNNCheckpoint::PMemDNNCheckpoint(const PMemDNNCheckpoint& other)
    : _chkpt_name(other._chkpt_name), _nlayers(other._nlayers), 
      _cur_layer_idx(other._cur_layer_idx), _nn_params(other._nn_params) {
        this_ptr = other.this_ptr;
        this_offset = other.this_offset;
}


// Re-construct from PMem - deserialization
// Note: the detailed parameters are not loaded for better performance
// You can call load_params() if you need the param offsets 
void 
PMemDNNCheckpoint::from_pmem(PMemPool* pool, off64_t offset) {
    this_offset = offset;
    this_ptr = pool->get_obj(offset);
    off64_t* obj_i64 = reinterpret_cast<off64_t*>(this_ptr);
    size_t space = obj_i64[0];
    _nlayers = obj_i64[1];
    _chkpt_name = std::string(reinterpret_cast<char*>(this_ptr + (2 + _nlayers)*sizeof(uint64_t)));
}

void
PMemDNNCheckpoint::load_params(PMemPool* pool) {
    off64_t* obj_i64 = reinterpret_cast<off64_t*>(this_ptr);
    for (int i = 0; i < _nlayers; i++) {
        off64_t layer_offset = obj_i64[i + 2];
        std::shared_ptr<DRAMDNNLayer> dram_layer(new DRAMDNNLayer());
        dram_layer->pmem_layer.from_pmem(pool, layer_offset);
        dram_layer->pmem_data_addr = pool->get_obj(dram_layer->pmem_layer.data_offset);
        _nn_params[dram_layer->pmem_layer.layer_name] = dram_layer;
    }
}

void
PMemDNNCheckpoint::register_layer(std::string layer_name, size_t size_in_bytes, PMemPool* pool) {
    _mutex.lock();
    auto [data_offset, data_ptr] = pool->alloc(size_in_bytes);
    std::shared_ptr<DRAMDNNLayer> dram_layer(new DRAMDNNLayer());
    dram_layer->pmem_data_addr = data_ptr;
    dram_layer->pmem_layer.layer_name = layer_name;
    dram_layer->pmem_layer.size_in_bytes = size_in_bytes;
    dram_layer->pmem_layer.data_offset = data_offset;
    _nn_params[layer_name] = dram_layer;
    off64_t* this_ptr_i64 = reinterpret_cast<off64_t*>(this_ptr);
    _mm_mfence();
    this_ptr_i64[2 + _cur_layer_idx] = dram_layer->pmem_layer.persist(pool);
    _mm_mfence();
    _mm_clwb(&this_ptr_i64[2 + _cur_layer_idx]);
    _cur_layer_idx++;
    _mutex.unlock();
}

// Persist to PMem - serialization
off64_t
PMemDNNCheckpoint::persist(PMemPool* pool) {
    // [this_size, N_layers, layer_offsets[], name]
    size_t space_needed = sizeof(uint64_t) + sizeof(_nlayers) + sizeof(off64_t)*_nlayers + (_chkpt_name.size() + 1);
    std::tie(this_offset, this_ptr) = pool->alloc(space_needed);
    memset(this_ptr, 0, space_needed);
    off64_t* this_ptr_i64 = reinterpret_cast<off64_t*>(this_ptr);
    _mm_mfence();
    this_ptr_i64[0] = space_needed;
    this_ptr_i64[1] = _nlayers;
    memcpy(this_ptr + (2 + _nlayers)*sizeof(off64_t), _chkpt_name.c_str(), _chkpt_name.size() + 1);
    _mm_mfence();
    _mm_clwb(&this_ptr_i64[0]);
    _mm_clwb(&this_ptr_i64[1]);
    clwb(this_ptr + (2 + _nlayers)*sizeof(off64_t), _chkpt_name.size() + 1);
    return this_offset;
}

std::vector<std::pair<std::string, size_t> >
PMemDNNCheckpoint::get_layers_info() {
    std::vector<std::pair<std::string, size_t> > ret;
    for (auto&& [layer_name, dram_layer] : _nn_params) {
        ret.push_back({layer_name, dram_layer->pmem_layer.size_in_bytes});
    }
    return ret;
}

byte_t*
PMemDNNCheckpoint::get_layer_data(std::string layer_name) {
    return _nn_params[layer_name]->pmem_data_addr;
}

const std::string&
PMemDNNCheckpoint::name() const {
    return _chkpt_name;
}

const size_t
PMemDNNCheckpoint::nlayers() const {
    return _nlayers;
}

void 
PMemDNNCheckpoint::summary(int verbose) const {
    printf("================== DNN Checkpoint Summary ==================\n");
    printf("chkpt name: %s, layers=%ld\n", _chkpt_name.c_str(), _nlayers);
    for (auto&& [layer_name, dram_layer] : _nn_params) {
        printf("  layer: %s\n", layer_name.c_str());
        dram_layer->pmem_layer.summary(verbose);
        if (verbose > 1) {
            float* ptr = reinterpret_cast<float*>(dram_layer->pmem_data_addr);
            printf("    data: ");
            for (int i = 0; i < dram_layer->pmem_layer.size_in_bytes/sizeof(float); i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n");            
        }
    }
}

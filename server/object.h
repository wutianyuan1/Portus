#pragma once
#include "common.h"
#include "pool.h"


class PMemObject {
public:
    PMemObject() : this_offset(0), this_ptr(nullptr) {};
    ~PMemObject() = default;

    virtual off64_t persist(PMemPool* pool) = 0;
    virtual void from_pmem(PMemPool* pool, off64_t offset) = 0;
    virtual const std::string& name() const = 0;
    virtual void summary(int verbose) const = 0;

protected:
    off64_t this_offset;
    byte_t* this_ptr;
};


class PMemDNNLayer : public PMemObject {
public:
    off64_t persist(PMemPool* pool);
    void from_pmem(PMemPool* pool, off64_t offset);
    const std::string& name() const;
    void summary(int verbose=0) const;

    size_t size_in_bytes;
    off64_t data_offset;
    std::string layer_name;
};


struct DRAMDNNLayer {
    PMemDNNLayer pmem_layer;
    byte_t* pmem_data_addr;
};


class PMemDNNCheckpoint : public PMemObject {
public:
    // New checkpoint
    PMemDNNCheckpoint();
    PMemDNNCheckpoint(const PMemDNNCheckpoint& other);
    PMemDNNCheckpoint(std::string chkpt_name, int nlayers);

    // Re-construct from PMem - deserialization
    // Note: the detailed parameters are not loaded for better performance
    // You can call load_params() if you need the param offsets 
    void from_pmem(PMemPool* pool, off64_t offset);
    void load_params(PMemPool* pool);

    // Persist to PMem - serialization
    off64_t persist(PMemPool* pool);

    void register_layer(std::string layer_name, size_t size_in_bytes, PMemPool* pool);
    byte_t* get_layer_data(std::string layer_name);
    std::vector<std::pair<std::string, size_t> > get_layers_info();

    const std::string& name() const;
    const size_t nlayers() const;
    void summary(int verbose=0) const;

private:
    std::string _chkpt_name;
    size_t _nlayers;
    std::map<std::string, std::shared_ptr<DRAMDNNLayer> > _nn_params;
    int _cur_layer_idx;
    std::mutex _mutex;
};

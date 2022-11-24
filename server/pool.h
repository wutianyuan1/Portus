#pragma once
#include "common.h"


// A very stupid PM allocator, replace it with better solutions
// It can only alloc objects, free is not supported . It contains at most
// (2MB - 16B)/16B = 131071 objects.
class PMemPool {
public:
    PMemPool();
    ~PMemPool();

    int open_pmem(std::string dev_name, size_t map_size, bool init=false, bool use_dram=false);
    int close_pmem();

    std::pair<off64_t, byte_t*> alloc(size_t alloc_size);

    const byte_t* base_addr() const;
    byte_t* get_obj(off64_t offset) const;
    size_t allocated_chunks() const;
    off64_t availble_offset() const;
    void print_stats() const;

private:
    std::string _dev_name;
    size_t _map_size;
    int _dev_fd;
    byte_t* _base_addr;
    std::atomic<size_t> _allocated_chunks;
    std::atomic<off64_t> _tail_offset;
    std::mutex _mutex;
};
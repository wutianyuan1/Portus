#include "pool.h"


PMemPool::PMemPool() 
    : _dev_name("None"), _map_size(0), _dev_fd(0), _base_addr(nullptr),
    _allocated_chunks(0), _tail_offset(0) {}

PMemPool::~PMemPool() {
    close_pmem();
}

int
PMemPool::open_pmem(std::string dev_name, size_t map_size, bool init, bool use_dram) {
    _mutex.lock();
    // open and map the device
    _dev_name = dev_name;
    _map_size = map_size;
    _dev_fd = open(_dev_name.c_str(), O_RDWR);
#ifdef DEBUG_
    printf("Open fd: %d, map_size: %ld, %s\n", _dev_fd, map_size, strerror(errno));
#endif
    if (_dev_fd < 0){
        _mutex.unlock();
        return -1;
    }
    if (use_dram) {
        std::cout << "Use Dram\n";
        _base_addr = static_cast<byte_t*>(malloc(map_size));
    } else {
        std::cout << "Use Pmem\n";
        _base_addr = static_cast<byte_t*>(mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, _dev_fd, 0));
    }
#ifdef DEBUG_
    printf("Status: %s, addr: %p\n", strerror(errno), _base_addr);
#endif
    if (_base_addr == (byte_t*)-1){
        _mutex.unlock();
        return -1;
    }
        

    // initialize the pool states
    off64_t* pm_base_i64 = reinterpret_cast<off64_t*>(_base_addr);
    if (init){
        _mm_mfence();
        memset(_base_addr, 0, ALLOC_TABLE_SIZE);
        _mm_mfence();
        clwb(_base_addr, ALLOC_TABLE_SIZE);
        _mm_mfence();
        pm_base_i64[0] = 0; // number of allocated chunks
        pm_base_i64[1] = ALLOC_TABLE_SIZE; // first availble offset
        _mm_mfence();
        _mm_clwb(_base_addr + sizeof(off64_t));
        _mm_mfence();
    }
    _allocated_chunks = pm_base_i64[0];
    _tail_offset = pm_base_i64[1];
    _mutex.unlock();
    return 0;
}

int
PMemPool::close_pmem() {
    if (_base_addr){
        munmap(_base_addr, _map_size);
        return close(_dev_fd);
    }
    std::cerr << "No opened PMem devices\n";
    return -1;
}

std::pair<off64_t, byte_t*>
PMemPool::alloc(size_t alloc_size) {
    _mutex.lock();
    size_t aligned_size = (alloc_size%256 == 0 ? alloc_size : (1 + (alloc_size/256))*256);
    off64_t cur_offset = _tail_offset;
    byte_t* ptr = _base_addr + cur_offset;
    off64_t* pm_base_i64 = reinterpret_cast<off64_t*>(_base_addr);
    // in memory updates
    if (_allocated_chunks + 1 >= ((ALLOC_TABLE_SIZE - 16)/16) 
        || _tail_offset + aligned_size >=  _map_size) {
            std::cerr << "PMem pool is full, allocation failed\n";
            return {0, nullptr};
    }
    _allocated_chunks += 1;
    _tail_offset += aligned_size;
    // in PM updates
    // (1) record this entry to alloc table
    _mm_mfence();
    pm_base_i64[2*_allocated_chunks] = alloc_size;
    pm_base_i64[2*_allocated_chunks + 1] = cur_offset;
    _mm_mfence();
    _mm_clwb(&pm_base_i64[2*_allocated_chunks]);
    _mm_clwb(&pm_base_i64[2*_allocated_chunks + 1]);
    // (2) update global counts
    _mm_mfence();
    pm_base_i64[0] = _allocated_chunks; // update number of allocated chunks
    pm_base_i64[1] = _tail_offset; // update the offset of last obj
    _mm_mfence();
    _mm_clwb(pm_base_i64);
    _mm_mfence();
    _mutex.unlock();
    return {cur_offset, ptr};
}

const byte_t*
PMemPool::base_addr() const {
    return _base_addr;
}

byte_t*
PMemPool::get_obj(off64_t offset) const {
    return _base_addr + offset;
}

size_t
PMemPool::allocated_chunks() const {
    return _allocated_chunks;
}

off64_t
PMemPool::availble_offset() const {
    return _tail_offset;
}

void
PMemPool::print_stats() const {
    off64_t* pm_base_i64 = reinterpret_cast<off64_t*>(_base_addr);
    printf("Current allocated chunks: %ld, current available offset: %ld\n",
        _allocated_chunks, _tail_offset);
    for (int i = 1; i <= _allocated_chunks; i++)
        printf("  chunk %d: size=%ld, relative offset=%ld\n",
            i - 1, pm_base_i64[2*i], pm_base_i64[2*i + 1] - ALLOC_TABLE_SIZE);
}

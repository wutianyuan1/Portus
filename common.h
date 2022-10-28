#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <map>
#include <memory>
#include <vector>
#include <functional>
#include <mutex>

#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/time.h>
#include <netdb.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <x86intrin.h>

#define ALLOC_TABLE_SIZE (2*1024*1024)
#define CHKPT_TABLE_SIZE (2*1024*1024)
#define CACHELINE_SIZE 64
using byte_t = char;


inline void clwb(byte_t* addr, size_t size) {
    for (int i = 0; i < size; i += CACHELINE_SIZE)
        _mm_clwb(addr + i);
}

#include <sys/mman.h>
#include <stdio.h>
#include <sys/file.h>
#include <assert.h>

int main(){
    // check if the PMEM contens after RDMA is correct
    // note: it may need root permission
    int fd = open("/dev/dax0.0", O_RDONLY);
    printf("fd=%d\n", fd);
    int len = 2*1024*1024; // 2MB
    printf("length = %ld\n", len);
    char* buff = mmap(0, len, PROT_READ, MAP_SHARED, fd, 0);
    for (int i = 0; i < len; i++) {
        if (buff[i] != 'a'){
            printf("pos: %d, char: %c\n", i, buff[i]);
            return 1;
        }
    }
    printf("check passed\n");
    return 0;
}
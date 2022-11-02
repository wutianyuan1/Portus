#pragma once
#include <getopt.h>
#include "common.h"


struct user_params {
    std::string hostaddr;
    std::string dax_device;
    size_t pmem_size;
    int port;
    bool init;
    int worker;
};


void
usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s            start a server and wait for connection\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -d, --device              dax device for saving checkpoints\n");
    printf("  -a, --addr=<ipaddr>       ip address of the local host net device <ipaddr v4> (mandatory)\n");
    printf("  -p, --port=<port>         listen on/connect to port <port> (default 18515)\n");
    printf("  -s, --size=<size>         size of mapped region of the PMEM in MB (default 16GB)\n");
    printf("  -i, --init=<0/1>          whether to clean-up the PMEM device and init a new FS on it\n");
    printf("  -w, --worker=<num>        multithread worker num\n");
}


int
parse_command_line(int argc, char *argv[], struct user_params *usr_par) {
    memset(usr_par, 0, sizeof *usr_par);
    /*Set defaults*/
    usr_par->hostaddr.assign("192.168.10.4");
    usr_par->dax_device.assign("/dev/dax0.0");
    usr_par->pmem_size = (size_t)128*1024*1024*1024;
    usr_par->port = 12345;
    usr_par->init = true;
    usr_par->worker = 4;

    while (1) {
        int c;

        static struct option long_options[] = {
            { .name = "device", .has_arg = 1, .val = 'd' },
            { .name = "addr", .has_arg = 1, .val = 'a' },
            { .name = "port", .has_arg = 1, .val = 'p' },
            { .name = "size", .has_arg = 1, .val = 's' },
            { .name = "init", .has_arg = 1, .val = 'i' },
            { .name = "worker", .has_arg = 1, .val = 'w'},
            { 0 }
        };

        c = getopt_long(argc, argv, "d:a:p:s:i:w:",
                        long_options, NULL);
        
        if (c == -1)
            break;

        switch (c) {
        case 'd':
            usr_par->dax_device = std::string(optarg);
            break;

        case 'a':
            usr_par->hostaddr = std::string(optarg);
            break;

        case 'p':
            usr_par->port = strtol(optarg, NULL, 0);
            if (usr_par->port < 0 || usr_par->port > 65535) {
                usage(argv[0]);
                return 1;
            }
            break;

        case 's':
            usr_par->pmem_size = (size_t)1024*1024*strtol(optarg, NULL, 0);
            break;

        case 'i':
            usr_par->init = strtol(optarg, NULL, 0);
            break;

        case 'w':
            usr_par->worker = strtol(optarg, NULL, 0);
            break;
            
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (optind < argc) {
        usage(argv[0]);
        return 1;
    }

    return 0;
}

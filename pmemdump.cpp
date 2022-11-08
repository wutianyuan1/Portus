#include "chksystem.h"


static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s            Dump the checkpoints from a DAX device managed by GPU-RPMA\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -d, --device              dax device for saving checkpoints\n");
    printf("  -s, --size=<size>         size of mapped region of the PMEM in MB (default 16GB)\n");
    printf("  -v, --verbose=<level>     verbose level, it can be 0, 1, 2\n");
}


int main(int argc, char** argv){
    // Set defaults
    std::string dax_device("/dev/dax0.0");
    size_t pmem_size = (size_t)16*1024*1024*1024;
    int verbose = 0;
    // Parse user's arguments
    while (1) {
        static struct option long_options[] = {
            { .name = "device", .has_arg = 1, .val = 'd' },
            { .name = "size", .has_arg = 1, .val = 's' },
            { .name = "verbose", .has_arg = 1, .val = 'v' },
            { 0 }
        };

        int c = getopt_long(argc, argv, "d:s:v:", long_options, NULL);
        if (c == -1)
            break;

        switch (c) {
        case 'd':
            dax_device = std::string(optarg);
            break;
        case 's':
            pmem_size = (size_t)1024*1024*strtol(optarg, NULL, 0);
            break;
        case 'v':
            verbose = strtol(optarg, NULL, 0);
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

    CheckpointSystem chksys(dax_device, pmem_size, false);

    printf("Current chkpts:\n");
    for (auto& chkpt_name : chksys.existing_chkpts()){
        if (verbose > 0)
            chksys.load_network_params(chkpt_name);
        chksys.chkpt_summary(chkpt_name, verbose);
    }

    return 0;
}

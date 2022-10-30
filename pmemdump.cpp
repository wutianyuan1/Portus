#include "chksystem.h"


int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "Usage: ./" << argv[0] << " -v VERBOSE_LEVEL\n";
        return 1;
    }
    size_t pmsize = (size_t)16*1024*1024*1024; // 16GB
    assert(strcmp(argv[1], "-v") == 0);
    int verbose = std::atoi(argv[2]);
    CheckpointSystem chksys("/dev/dax0.0", pmsize, false);

    printf("current chkpts:\n");
    for (auto& chkpt_name : chksys.existing_chkpts()){
        if (verbose > 0)
            chksys.load_network_params(chkpt_name);
        chksys.chkpt_summary(chkpt_name, verbose);
    }
    return 0;
}
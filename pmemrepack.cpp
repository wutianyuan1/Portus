#include "chksystem.h"


struct layer_data_t {
    byte_t* data;
    size_t size;
};


static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s            Dump the checkpoints from a DAX device managed by GPU-RPMA\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -d, --device              dax device for saving checkpoints\n");
}


int main(int argc, char** argv){
    // Set defaults
    std::string dax_device("/dev/dax0.0");
    size_t pmem_size = (size_t)128*1024*1024*1024;
    int verbose = 0;
    // Parse user's arguments
    while (1) {
        static struct option long_options[] = {
            { .name = "device", .has_arg = 1, .val = 'd' },
            { 0 }
        };

        int c = getopt_long(argc, argv, "d:", long_options, NULL);
        if (c == -1)
            break;

        switch (c) {
        case 'd':
            dax_device = std::string(optarg);
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

    std::cout << "Repacking checkpoints on device " << dax_device << std::endl;

    CheckpointSystem* chksys = new CheckpointSystem(dax_device, pmem_size, false);
    std::map<std::string, std::map<std::string, layer_data_t> > all_models;
    // load all valid checkpoints to main memory, invalid chkpts will not be loaded
    for (auto& chkpt_name : chksys->existing_chkpts()) {
        auto chkpt = chksys->get_chkpt(chkpt_name);
        printf("load model %s, layers: %d\n", chkpt_name.c_str(), chkpt->nlayers());
        for (auto&& [layer_name, layer_size] : chkpt->get_layers_info()) {
            all_models[chkpt_name][layer_name].data = new byte_t[layer_size];
            all_models[chkpt_name][layer_name].size = layer_size;
            byte_t* layer_data = chkpt->get_layer_data(layer_name);
            memcpy(all_models[chkpt_name][layer_name].data, layer_data, layer_size);
        }
    }
    delete chksys;

    CheckpointSystem* newsys = new CheckpointSystem(dax_device, pmem_size, true);
    // write to a new system
    for (auto&& [chkpt_name, layer_info]: all_models) {
        newsys->new_chkpt(chkpt_name, layer_info.size());
        printf("new model %s, layers: %d\n", chkpt_name.c_str(), layer_info.size());
        auto chkpt = newsys->get_chkpt(chkpt_name);
        for (auto&& [layer_name, layer_data] : layer_info) {
            newsys->register_network_layer(chkpt_name, layer_name, layer_data.size);
            byte_t* pmem_addr = chkpt->get_layer_data(layer_name);
            memcpy(pmem_addr, layer_data.data, layer_data.size);
        }
    }

    return 0;
}

#include "chksystem.h"
#include <H5Cpp.h>
#include <hdf5.h>

enum functions_t {
    VIEW_CHECKPOINTS,
    DUMP_TO_FILE,
    REPACK,
    DELETE_CHECKPOINT,
    UNKOWN
};

enum output_t {
    HDF5,
    PICKLE
};

struct layer_data_t {
    byte_t* data;
    size_t size;
};


static void usage(const char *argv0, functions_t func=functions_t::UNKOWN) {
    printf("Usage:\n");
    printf("  %s  <function> <args>*     manage the DAX device managed by GPU-RPMA\n", argv0);
    printf("\n");
    if (func == functions_t::UNKOWN) {
        printf("Available functions: view, dump, repack, delete\n");
        return;
    }
    printf("Options:\n");
    printf("  -d, --device              dax device for saving checkpoints\n");
    printf("  -s, --size=<size>         size of mapped region of the PMEM in MB (default 16GB)\n");
    switch (func) {
        case functions_t::VIEW_CHECKPOINTS:
            printf("  -v, --verbose=<level>     verbose level, it can be 0, 1, 2\n");
            break;
        case functions_t::DUMP_TO_FILE:
            printf("  -n, --name=<name>             checkpoint name to copy out\n");
            printf("  -o, --outfile=<filename>      output file name or path\n");
            printf("  -t, --type=<hdf5/pickle>      output file type: HDF5 or pickle\n");
        case functions_t::REPACK:
            break;
        case functions_t::DELETE_CHECKPOINT:
            printf("  -n, --name=<name>             checkpoint name to copy out\n");
            break;
        default:
            printf("unknown function!\n");
            break;
    }    
}


static void dump_to_hdf5(std::shared_ptr<PMemDNNCheckpoint> chkpt, std::string out_fn) {
    H5::H5File file(out_fn, H5F_ACC_CREAT | H5F_ACC_RDWR);

    H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);
    H5::Group chkpt_group(file.createGroup(chkpt->name()));

    /* write layers to HDF5 file */
    for (auto&& [layer_name, layer_size] : chkpt->get_layers_info()) {
        hsize_t dims = layer_size / sizeof(float);
        H5::DataSpace data_space(1, &dims);
        H5::DataSet dataset(chkpt_group.createDataSet(layer_name, datatype, data_space));
        dataset.write(chkpt->get_layer_data(layer_name), datatype);
        data_space.close();
        dataset.close();
    }
    file.close();
}


static void view_chkpts(CheckpointSystem* chksys, int verbose) {
    printf("Current chkpts:\n");
    for (auto& chkpt_name : chksys->existing_chkpts()){
        if (verbose > 0)
            chksys->load_network_params(chkpt_name);
        chksys->chkpt_summary(chkpt_name, verbose);
    }
}

static void repack(CheckpointSystem* chksys, std::string dax_device, size_t pmem_size) {
    std::cout << "Repacking checkpoints on device " << dax_device << std::endl;

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
}


static void delete_chkpt(CheckpointSystem* chksys, std::string chkpt_name) {
    chksys->remove_chkpt(chkpt_name);
}


int main(int argc, char** argv){
    if (argc <= 1) {
        usage(argv[0]);
        return 1;
    }

    // Set defaults
    std::string dax_device("/dev/dax0.0");
    size_t pmem_size = (size_t)16*1024*1024*1024;
    int verbose = 0;
    std::string output_file, chkpt_name;
    output_t output_type;
    functions_t function;
    // parse user's command
    if (strcmp(argv[1], "view") == 0)
        function = functions_t::VIEW_CHECKPOINTS;
    else if (strcmp(argv[1], "dump") == 0)
        function = functions_t::DUMP_TO_FILE;
    else if (strcmp(argv[1], "repack") == 0)
        function = functions_t::REPACK;
    else if (strcmp(argv[1], "delete") == 0)
        function = functions_t::DELETE_CHECKPOINT;
    else {
        usage(argv[0]); 
        return 1;
    }

    // Parse user's arguments
    while (1) {
        static struct option long_options[] = {
            { .name = "device", .has_arg = 1, .val = 'd' },
            { .name = "size", .has_arg = 1, .val = 's' },
            { .name = "verbose", .has_arg = 1, .val = 'v' },
            { .name = "name", .has_arg = 1, .val = 'n' },
            { .name = "output", .has_arg = 1, .val = 'o' },
            { .name = "type", .has_arg = 1, .val = 't' },
            { 0 }
        };

        int c = getopt_long(argc, argv, "d:s:v:n:o:t:", long_options, NULL);
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
        case 'n':
            chkpt_name = std::string(optarg);
            break;
        case 'o':
            output_file = std::string(optarg);
            break;
        case 't':
            output_type = (output_t)strtol(optarg, NULL, 0);
            break;
        default:
            usage(argv[0], function);
            return 1;
        }
    }

    CheckpointSystem* chksys = new CheckpointSystem(dax_device, pmem_size, false);
    switch (function) {
    case functions_t::VIEW_CHECKPOINTS:
        view_chkpts(chksys, verbose);
        break;
    case functions_t::DUMP_TO_FILE:
        if (output_type == output_t::HDF5)
            dump_to_hdf5(chksys->get_chkpt(chkpt_name), output_file);
        else
            printf("Not implemented yet...\n");
        break;
    case functions_t::REPACK:
        repack(chksys, dax_device, pmem_size);
        break;
    case functions_t::DELETE_CHECKPOINT:
        delete_chkpt(chksys, chkpt_name);
        break;
    default:
        printf("unknown function!\n");
        break;
    }
    return 0;
}

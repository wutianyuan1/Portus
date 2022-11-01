#include <H5Cpp.h>
// #include <boost/python.hpp>

#include "common.h"
#include "chksystem.h"

enum OutputType {
    HDF5,
    Pickle
};

static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s            Copy a checkpoint out of PMEM to a Pickle/HDF5 file\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -d, --device                  dax device for saving checkpoints\n");
    printf("  -n, --name=<name>             checkpoint name to copy out\n");
    printf("  -o, --outfile=<filename>      output file name or path\n");
    printf("  -t, --type=<hdf5/pickle>      output file type: HDF5 or pickle\n");
}


void copy_to_hdf5(std::shared_ptr<PMemDNNCheckpoint> chkpt, std::string out_fn) {
    H5::H5File file(out_fn, H5F_ACC_CREAT | H5F_ACC_RDWR);

    H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);
    H5::Group chkpt_group(file.createGroup(chkpt->name()));

    /* write layers to HDF5 file */
    for (auto&& [layer_name, layer_size] : chkpt->get_layers_info()) {
        hsize_t dims = layer_size / sizeof(float);
        H5::DataSpace dataSpace(1, &dims);
        H5::DataSet dataSet(chkpt_group.createDataSet(layer_name, datatype, dataSpace));
        dataSet.write(chkpt->get_layer_data(layer_name), datatype);
        dataSpace.close();
        dataSet.close();
    }
    file.close();
}


void copy_to_pickle(std::shared_ptr<PMemDNNCheckpoint> chkpt, std::string out_fn) {
    // Unimplemented...
    // I don't know how to write pickles in C++...
}


int main(int argc, char** argv){
    // Set defaults
    std::string dax_device("/dev/dax0.0");
    size_t pmem_size = (size_t)128*1024*1024*1024;
    std::string chkpt_name, out_fn;
    OutputType otype = OutputType::HDF5;
    // Parse user's arguments
    while (1) {
        static struct option long_options[] = {
            { .name = "device", .has_arg = 1, .val = 'd' },
            { .name = "name", .has_arg = 0, .val = 'n' },
            { .name = "outfile", .has_arg = 0, .val = 'o' },
            { .name = "type", .has_arg = 1, .val = 't'},
            { 0 }
        };

        int c = getopt_long(argc, argv, "d:n:o:t:", long_options, NULL);
        if (c == -1)
            break;

        switch (c) {
        case 'd':
            dax_device = std::string(optarg);
            break;
        case 'n':
            chkpt_name = std::string(optarg);
            break;
        case 'o':
            out_fn = std::string(optarg);
            break;
        case 't':
            if (strcmp(optarg, "HDF5") == 0 || strcmp(optarg, "hdf5") == 0)
                otype = OutputType::HDF5;
            else if (strcmp(optarg, "Pickle") == 0 || strcmp(optarg, "pickle") == 0)
                otype = OutputType::Pickle;
            else {
                std::cerr << "Unsupported type: " << optarg << std::endl;
                return 1;                
            }
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
    auto chkpt = chksys.get_chkpt(chkpt_name);
    if (otype == OutputType::HDF5)
        copy_to_hdf5(chkpt, out_fn);
    else if (otype == OutputType::Pickle)
        copy_to_pickle(chkpt, out_fn);
    else
        std::cerr << "Unsupported output type: " << (int)otype << "\n";

    return 0;
}


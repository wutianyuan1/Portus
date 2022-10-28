#include "chksystem.h"


int main(int argc, char** argv){
    size_t pmsize = (size_t)16*1024*1024*1024; // 16GB
    bool init = bool(std::atoi(argv[1]));
    CheckpointSystem chksys("/dev/dax0.0", pmsize, init);

    if (init) {
        chksys.new_chkpt("MyCNN", 5);
        chksys.register_network_layer("MyCNN", "module.conv1", 100);
        chksys.register_network_layer("MyCNN", "module.conv2", 50);
        chksys.register_network_layer("MyCNN", "module.pool2D", 80);
        chksys.register_network_layer("MyCNN", "module.fc1", 810975);
        chksys.register_network_layer("MyCNN", "module.fc2", 260817);

        chksys.new_chkpt("MySB", 1);
        chksys.register_network_layer("MySB", "nmsl", 50);

        chksys.chkpt_summary("MyCNN");
        printf("----------------------------------------------\n");
        chksys.chkpt_summary("MySB");
    } else {
        printf("current chkpts:\n");
        for (auto& chkpt_name : chksys.existing_chkpts()){
            chksys.chkpt_summary(chkpt_name);
            chksys.load_network_params(chkpt_name);
            chksys.chkpt_summary(chkpt_name);
            printf("----------------------------------------------\n");
        }
    }

    return 0;
}
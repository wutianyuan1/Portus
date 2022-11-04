#include <torch/extension.h>

#include "utils.h"
#include "checkpoint.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_checkpoint", &init_checkpoint, "init checkpoint");
    m.def("checkpoint", &checkpoint, "do checkpointing");
    m.def("restore", &restore, "do restore");
    m.def("wait_checkpoint_done", &wait_checkpoint_done, "wait checkpoint done");
}

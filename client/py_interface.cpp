#include <torch/extension.h>

#include "utils.h"
#include "checkpoint.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "fast DNN checkpoint with RPMA";
    m.def("init_checkpoint", &init_checkpoint, "init checkpoint", py::arg("network_name"), py::arg("state_dict"), py::arg("ib_local_addr") = std::string("192.168.10.101"), py::arg("server_addr") = std::string("192.168.10.4"));
    m.def("checkpoint", &checkpoint, "do checkpointing", py::arg("use_async") = true);
    m.def("restore", &restore, "do restore");
    m.def("wait_checkpoint_done", &wait_checkpoint_done, "wait checkpoint done");
}

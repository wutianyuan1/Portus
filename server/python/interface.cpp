#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "../chksystem.h"
#include "../object.h"


namespace py = pybind11;

PYBIND11_MODULE(rpma_server, m) {
    m.doc() = "Python interfaces for GPU-RPMA server";

    py::class_<PMemDNNCheckpoint>(m, "PMemDNNCheckpoint")
        .def(py::init<std::string, int>(), "Constructor of PMemDNNCheckpoint")
        .def("get_layers_info", &PMemDNNCheckpoint::get_layers_info, "get info of all layers")
        .def("__repr__",
            [](PMemDNNCheckpoint& chkpt) {
            return "<rpma_server.PMemDNNCheckpoint name=" + chkpt.name() + ">";
        })
        ;

    py::class_<CheckpointSystem>(m, "CheckpointSystem")
        .def(py::init<std::string, size_t, bool, bool>(), "Constructor of CheckpointSystem")
        .def("new_chkpt", &CheckpointSystem::new_chkpt, "add a new DNN checkpoint")
        .def("remove_chkpt", &CheckpointSystem::remove_chkpt, "remove a DNN checkpoint")
        .def("existing_chkpts", &CheckpointSystem::existing_chkpts, "get existing checkpoint names")
        .def("invalid_chkpts", &CheckpointSystem::invalid_chkpts, "get invalid checkpoint info")
        .def("get_chkpt", &CheckpointSystem::get_chkpt_obj, "get checkpoint info by name")
        .def("__repr__",
            [](CheckpointSystem& chksys) {
            return "<rpma_server.CheckpointSystem current_models=" + 
                std::to_string(chksys.existing_chkpts().size()) + ">";
        });
}
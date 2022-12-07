/**
 * @file raw_tensor.cpp
 * @brief Implementation of Portus tensor wrapper
 * @author madoka, stevelee477
 */
#include "raw_tensor.h"

// The "raw" view of NN parameters, where key is layer name, value is
// a raw_tensor of this layer
raw_network_t get_raw_network(torch_network_t& network_params){
    raw_network_t ret;
    for (auto&& kv: network_params){
        ret[kv.first] = std::shared_ptr<raw_tensor>(new raw_tensor(kv.second));
    }
    return ret;
}

// To re-build a torch::Tensor from the raw tensor and metadata, just for test
// This may be used for checkpoint restoring - unefficient because of cudaMemcpy
// return 0 for success, 1 for error
int from_raw(std::shared_ptr<raw_tensor> raw, torch::Tensor& tensor_out){
    int success = 0;
    // raw tensor and tensor to restore should have same size
    if (raw->size_in_bytes != tensor_out.nbytes()){
        std::cerr << "size " << raw->size_in_bytes << "!=" << tensor_out.nbytes() << "\n";
        return 1;
    }
    // we assume they are all on cuda device
    if (raw->device != torch::DeviceType::CUDA || tensor_out.device() != torch::DeviceType::CUDA){
        std::cerr << "both raw and tensor to restore should on CUDA devices\n";
        return 1;
    }
    return cudaMemcpy(tensor_out.data_ptr(), raw->addr, raw->size_in_bytes, cudaMemcpyDeviceToDevice);
}
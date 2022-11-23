#pragma once
#include <map>
#include <iostream>
#include <memory>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Type alias
struct raw_tensor;
using torch_network_t = std::map<std::string, torch::Tensor>;
// Here we use a shared ptr to avoid many move semantics - torch::Device has
// no default constructors, f**k...
using raw_network_t = std::map<std::string, std::shared_ptr<raw_tensor> >;

// A "raw" tensor, `addr` is the GPU/CPU address of this tensor (depends on `device`)
// we only focus on the size, address, device of this tensor, the metadata including
// dimensions, gradients are lost in this "raw" view. But you can view its elements
// because `type` is saved.
struct raw_tensor{
    raw_tensor(torch::Tensor& tensor)
      : addr(static_cast<uint8_t*>(tensor.data_ptr())),
        size_in_bytes(tensor.nbytes()),
        dtype(tensor.scalar_type()),
        device(tensor.device())
    {}
    ~raw_tensor() = default;

    template <typename T>
    T* view(){
        return reinterpret_cast<T*>(addr);
    }

    uint8_t* addr;
    size_t size_in_bytes;
    torch::ScalarType dtype;
    torch::Device device;
};


// The "raw" view of NN parameters, where key is layer name, value is
// a raw_tensor of this layer
raw_network_t get_raw_network(torch_network_t& network_params);

// To re-build a torch::Tensor from the raw tensor and metadata, just for test
// This may be used for checkpoint restoring - unefficient because of cudaMemcpy
// return 0 for success, 1 for error
int from_raw(std::shared_ptr<raw_tensor> raw, torch::Tensor& tensor_out);
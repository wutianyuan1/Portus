#pragma once
#include "utils.h"
#include "raw_tensor.h"
#include <memory>

int init_checkpoint(std::string name, torch_network_t& network);
int checkpoint(bool async=false);
int restore();
int wait_checkpoint_done();
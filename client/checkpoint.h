/**
 * @file checkpoint.h
 * @brief Portus client checkpoint&restore interface
 * @author madoka, stevelee477
 */
#pragma once
#include "../common/utils.h"
#include "raw_tensor.h"
#include <memory>

int init_checkpoint(std::string name, torch_network_t& network, std::string ib_local_addr, std::string server_addr);
int checkpoint(bool async=false);
int restore();
int wait_checkpoint_done();
#pragma once
/**
 * @file utils.h
 * @brief common Portus utils
 * @author madoka, stevelee477
*/
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

/*
 * Convert IP address from string to sockaddr
 *
 * returns: 0 on success or 1 on error
 */
int get_addr(char *dst, struct sockaddr *addr);

/*
 * Print program run time.
 *
 * returns: 0 on success or 1 on error
 */
int print_run_time(struct timeval start, unsigned long size, int iters);

enum payload_t { RDMA_BUF_DESC, TASK_ATTRS };

struct payload_attr {
	enum payload_t data_t;
	char *payload_str;
};

int open_client_socket(const char *servername, int port);

int rdma_task_attr_flags_get_desc_str(int flags, char *desc_str, size_t desc_length);

int pack_payload_data(void *package, size_t package_size, struct payload_attr *attr);

#ifdef __cplusplus
}
#endif



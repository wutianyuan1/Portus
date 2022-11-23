/*
 * Copyright (c) 2019 Mellanox Technologies, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <netdb.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <stdlib.h>
#include <sys/file.h>
#include <unistd.h>

#include "utils.h"

int get_addr(char *dst, struct sockaddr *addr)
{
        struct addrinfo *res;
        int ret;

        ret = getaddrinfo(dst, NULL, NULL, &res);
        if (ret) {
                printf("getaddrinfo failed (%s) - invalid hostname or IP address\n", gai_strerror(ret));
                return ret;
        }

        if (res->ai_family == PF_INET)
                memcpy(addr, res->ai_addr, sizeof(struct sockaddr_in));
        else if (res->ai_family == PF_INET6)
                memcpy(addr, res->ai_addr, sizeof(struct sockaddr_in6));
        else
                ret = -1;

        freeaddrinfo(res);
        return ret;
}

int print_run_time(struct timeval start, unsigned long size, int iters)
{
    struct timeval  end;
    float           usec;
    long long       bytes;

    if (gettimeofday(&end, NULL)) {
        perror("gettimeofday");
        return 1;
    }

    usec  = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    bytes = (long long) size * iters;

    printf("%lld bytes in %.2f seconds = %.2f Mbit/sec\n",
           bytes, usec / 1000000., bytes * 8. / usec);
    printf("%d iters in %.2f seconds = %.2f usec/iter\n",
           iters, usec / 1000000., usec / iters);
    return 0;

}

/****************************************************************************************
 * Open socket connection on the client side, try to connect to the server by the given
 * IP address (servername). If success, return the connected socket file descriptor ID
 * Return value: socket fd - success, -1 - error
 ****************************************************************************************/
int open_client_socket(const char *servername,
                              int         port)
{
    struct addrinfo *res,
                    *t;
    struct addrinfo hints = {
        .ai_family   = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM
    };
    char   *service;
    int     ret_val;
    int     sockfd;

    if (asprintf(&service, "%d", port) < 0)
        return -1;

    ret_val = getaddrinfo(servername, service, &hints, &res);

    if (ret_val < 0) {
        fprintf(stderr, "FAILURE: %s for %s:%d\n", gai_strerror(ret_val), servername, port);
        free(service);
        return -1;
    }

    for (t = res; t; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd >= 0) {
            if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
                break;
            close(sockfd);
            sockfd = -1;
        }
    }

    freeaddrinfo(res);
    free(service);

    if (sockfd < 0) {
        fprintf(stderr, "FAILURE: Couldn't connect to %s:%d\n", servername, port);
        return -1;
    }

    return sockfd;
}

//====================================================================================
/*                                           t*/
#define RDMA_TASK_ATTR_DESC_STRING_LENGTH (sizeof "12345678")
/*************************************************************************************
 * Get a rdma_task_attr_flags description string representation
 *
 * The Client application should pass this description string to the
 * Server which will issue the RDMA Read/Write operation
 *
 * desc_str is input and output holding the rdma_task_attr_flags information
 * desc_length is input size in bytes of desc_str
 *
 * returns: an integer equal to the size of the char data copied into desc_str
 ************************************************************************************/
int rdma_task_attr_flags_get_desc_str(int flags, char *desc_str, size_t desc_length)
{
    if (desc_length < RDMA_TASK_ATTR_DESC_STRING_LENGTH) {
        fprintf(stderr, "desc string size (%lu) is less than required (%lu) for sending rdma_task_attr_flags data\n",
                desc_length, RDMA_TASK_ATTR_DESC_STRING_LENGTH);
        return 0;
    }
   
    sprintf(desc_str, "%08x", flags);
    
    return strlen(desc_str) + 1; /*including the terminating null character*/
}

/************************************************************************************
 * Simple package protocol which packs payload string into allocated memory.
 * Protocol consist of:
 * 		uint8_t payload_t - type of the payload data
 *  	uint16_t payload_size - strlen of the payload_str
 *  	char * payload_str - payload to pack
 * 
 * returns: an integer equal to the size of the copied into package data in bytes
 * _________________________________________________________________________________
 * 
 *            PACKAGE = {|type|size|---------payload----------|}                             
 *                         1b   2b    (size * sizeof(char))b 
 * 
 ***********************************************************************************/
int pack_payload_data(void *package, size_t package_size, struct payload_attr *attr)
{
    uint8_t data_t = attr->data_t;
    uint16_t payload_size = strlen(attr->payload_str) + 1;
    size_t req_size = sizeof(data_t) + sizeof(payload_size) + payload_size * sizeof(char) ;
    if (req_size > package_size) {
        fprintf(stderr, "package size (%lu) is less than required (%lu) for sending payload with attributes\n",
                package_size, req_size);
        return 0;
    }
    memcpy(package, &data_t, sizeof(data_t));
    memcpy(package + sizeof(data_t), &payload_size, sizeof(payload_size));
    memcpy(package + sizeof(data_t) + sizeof(payload_size), attr->payload_str, payload_size * sizeof(char));

    return req_size;
}
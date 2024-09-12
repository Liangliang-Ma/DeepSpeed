// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#ifndef _IO_OP_DESC_T_
#define _IO_OP_DESC_T_
#include <memory>
#include <queue>
#include "deepspeed_py_aio.h"


struct io_op_desc_t {
    const bool _read_op;
    torch::Tensor _buffer;
    int _fd;
    const std::string _filename;
    const long long int _num_bytes;
    torch::Tensor _cpu_buffer;
    torch::Tensor _contiguous_buffer;
    const bool _validate;

    io_op_desc_t(const bool read_op,
                 const torch::Tensor& buffer,
                 const int fd,
                 const char* filename,
                 const long long int num_bytes,
                 const bool validate);

    char* data_ptr() const;
    void fini();
};
#endif  // _IO_OP_DESC_T_
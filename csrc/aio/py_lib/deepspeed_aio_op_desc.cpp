// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_aio_op_desc.h"

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const long long int num_bytes,
                           const bool validate)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _num_bytes(num_bytes),
      _validate(validate)
{
    _cpu_buffer = (_buffer.is_cuda() || _buffer.is_xpu()
#if defined(__ENABLE_CANN__)
                   || torch_npu::utils::is_npu(_buffer)
#endif
                       )
                      ? _buffer.to(torch::kCPU).pin_memory()
                      : _buffer;
    _contiguous_buffer = _cpu_buffer.contiguous();
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::fini()
{
    if (_read_op && _buffer.is_cuda()) { _buffer.copy_(_cpu_buffer.to(torch::kCUDA)); }
    if (_read_op && _buffer.is_xpu()) { _buffer.copy_(_cpu_buffer.to(torch::kXPU)); }
#if defined(__ENABLE_CANN__)
    if (_read_op && torch_npu::utils::is_npu(_buffer)) {
        auto device = at::Device("npu:0");
        _buffer.copy_(_cpu_buffer.to(device));
    }
#endif
}
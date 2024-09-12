// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_aio_thread.h"


using namespace std;


deepspeed_aio_thread_t::deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config)
    : _tid(tid),
      _aio_config(aio_config),
      _aio_ctxt(new aio_context(aio_config._block_size, aio_config._queue_depth)),
      _time_to_exit(false)
{
}

deepspeed_aio_thread_t::~deepspeed_aio_thread_t() {}

void deepspeed_aio_thread_t::run()
{
    while (true) {
        std::shared_ptr<struct io_op_desc_t> next_io_op = nullptr;

        {
            std::unique_lock<std::mutex> lock(_work_sync._mutex);
            _work_sync._cond_var.wait(lock,
                                      [this] { return (!_work_queue.empty() || _time_to_exit); });
            if (!_work_queue.empty()) {
                next_io_op = _work_queue.front();
                _work_queue.pop();
            }
        }

        if (next_io_op) {
            // const auto base_offset = next_io_op->_num_bytes * _tid;

            // std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
            //     next_io_op->_fd, base_offset, next_io_op->_num_bytes, next_io_op->data_ptr()));

            // if (_aio_config._overlap_events) {
            //     do_aio_operation_overlap(
            //         next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
            // } else {
            //     do_aio_operation_sequential(
            //         next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
            // }
            next_io_op->run(_tid, _aio_ctxt, &_aio_config);

            {
                std::lock_guard<std::mutex> lock(_complete_sync._mutex);
                _complete_queue.push(next_io_op);
            }
            _complete_sync._cond_var.notify_one();
        }

        if (_time_to_exit) { break; }
    }
}

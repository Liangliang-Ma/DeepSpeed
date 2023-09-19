"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder


class CPUAdagradBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/xpu/adagrad/cpu_adagrad.dp.cpp',
            'csrc/xpu/adam/custom_sycl_kernel.dp.cpp'
        ]

    def include_paths(self):
        return [
            'csrc/xpu/includes',
            'csrc/xpu/adagrad',
            'csrc/includes'
        ]
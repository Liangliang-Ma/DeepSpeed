"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder


class CPUAdamBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/xpu/adam/cpu_adam.dp.cpp',
            'csrc/xpu/adam/custom_sycl_kernel.dp.cpp'
        ]

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return [
            'csrc/xpu/includes',
            'csrc/xpu/adam', 'csrc/includes'
        ]
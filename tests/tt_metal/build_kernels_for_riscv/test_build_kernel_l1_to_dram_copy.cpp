#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"




int main(int argc, char* argv[]) {

    std::string arch_name = tt::test_utils::get_env_arch_name();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","l1_to_dram_copy");

    log_info(tt::LogBuildKernels, "Compiling OP: {}", build_kernel_for_riscv_options.name);

    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/l1_to_dram_copy.cpp";

    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name);

    return 0;
}

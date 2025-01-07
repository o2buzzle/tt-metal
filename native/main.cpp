#include <cstdint>
#include <string>
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/work_split.hpp"

void run_once_cta() {
    Device* device = tt::tt_metal::CreateDevice(0);
    CommandQueue& command_queue = device->command_queue();
    auto program = tt::tt_metal::CreateProgram();

    auto cores = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores = num_cores_to_corerangeset(cores.x * cores.y, cores);

    const std::string reader_kernel_path = "/home/ubuntu/tt-metal/native/reader_kernel_256_cta.cpp";
    // Vector of 256 "1" values, use as kernel args
    std::vector<uint32_t> kernel_args(256, 1);
    for (int i = 0; i < 256; i++) {
        kernel_args[i] = i;
    }

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = kernel_args});
    EnqueueProgram(command_queue, program, true);
    Finish(command_queue);

    DumpDeviceProfileResults(device, program);

    CloseDevice(device);
}

void run_once_rta() {
    Device* device = tt::tt_metal::CreateDevice(0);
    CommandQueue& command_queue = device->command_queue();
    auto program = tt::tt_metal::CreateProgram();

    auto cores = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores = num_cores_to_corerangeset(cores.x * cores.y, cores);

    const std::string reader_kernel_path = "/home/ubuntu/tt-metal/native/reader_kernel_256_rta.cpp";
    // Vector of 256 "1" values, use as kernel args
    std::vector<uint32_t> kernel_args(256, 1);
    for (int i = 0; i < 256; i++) {
        kernel_args[i] = i;
    }

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
        });

    SetRuntimeArgs(program, reader_kernel, all_cores, kernel_args);
    EnqueueProgram(command_queue, program, true);
    Finish(command_queue);

    DumpDeviceProfileResults(device, program);

    CloseDevice(device);
}

void run_once_crta() {
    Device* device = tt::tt_metal::CreateDevice(0);
    CommandQueue& command_queue = device->command_queue();
    auto program = tt::tt_metal::CreateProgram();

    auto cores = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores = num_cores_to_corerangeset(cores.x * cores.y, cores);

    const std::string reader_kernel_path = "/home/ubuntu/tt-metal/native/reader_kernel_256_crta.cpp";
    // Vector of 256 "1" values, use as kernel args
    std::vector<uint32_t> kernel_args(256, 1);
    for (int i = 0; i < 256; i++) {
        kernel_args[i] = i;
    }

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
        });

    SetCommonRuntimeArgs(program, reader_kernel, kernel_args);
    EnqueueProgram(command_queue, program, true);
    Finish(command_queue);

    DumpDeviceProfileResults(device, program);

    CloseDevice(device);
}

int main() {
    // detail::EnablePersistentKernelCache();

    // std::cout << "CTA" << std::endl;
    // run_once_cta();

    // for (int i = 0; i < 100; i++) {
    //     run_once_cta();
    // }

    // std::cout << "RTA" << std::endl;
    // run_once_rta();

    // for (int i = 0; i < 100; i++) {
    //     run_once_rta();
    // }

    std::cout << "CRTA" << std::endl;
    run_once_crta();

    // for (int i = 0; i < 100; i++) {
    //     run_once_crta();
    // }

    return 0;
}

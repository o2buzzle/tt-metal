#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"

using namespace tt;

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

void add_defines(tt_metal::ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type){
    // TODO(AP): remove duplication
    string op_name, op_code;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; op_code = "0"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; op_code = "1"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; op_code = "2"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_code.c_str());
}


tt_metal::Program *setup_program_one(tt_metal::Device *device, const tt_xy_pair &core, uint32_t single_tile_size) {
    tt_metal::Program *program = new tt_metal::Program();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    auto binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        1, // per_core_block_cnt
        1 // per_core_block_size
    };
    tt_metal::KernelArgs eltwise_binary_args = tt_metal::KernelArgs(core, compute_kernel_args);
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        eltwise_binary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    eltwise_binary_kernel->add_define("ELTWISE_OP", "add_tiles");
    add_defines(eltwise_binary_kernel, BinaryOpType::ADD);

    return program;
}

tt_metal::Program *setup_program_two(tt_metal::Device *device, const tt_xy_pair &core, uint32_t single_tile_size) {
    tt_metal::Program *program = new tt_metal::Program();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    auto mm_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_matmul_small_block.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        1, // block_tile_dim
        1, // dst_tile_rows
        1, // dst_tile_cols
        1, // block_cnt
        1, // in0_block_tile_cnt
        1, // in1_block_tile_cnt
        1 // out_block_tile_cnt
    };
    tt_metal::KernelArgs mm_args = tt_metal::KernelArgs(core, compute_kernel_args);
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/matmul.cpp",
        core,
        mm_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    return program;
}

void write_program_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program *program,
    const tt_xy_pair &core,
    uint32_t num_tiles,
    tt_metal::Buffer &src0_dram_buffer,
    tt_metal::Buffer &src1_dram_buffer,
    tt_metal::Buffer &dst_dram_buffer) {

    auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

    for (auto dm_kernel : program->data_movement_kernels()) {
        if (dm_kernel->name() == "reader_binary" or dm_kernel->name() == "reader_matmul_small_block") {
            tt_metal::WriteRuntimeArgsToDevice(
                device, dm_kernel, core,
                {src0_dram_buffer.address(),
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                src1_dram_buffer.address(),
                (std::uint32_t)dram_src1_noc_xy.x,
                (std::uint32_t)dram_src1_noc_xy.y,
                num_tiles});
        } else if (dm_kernel->name() == "writer_unary") {
            tt_metal::WriteRuntimeArgsToDevice(
                device, dm_kernel, core,
                {dst_dram_buffer.address(),
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tiles});
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
// 1. First program runs eltwise binary on logical core {0, 0}
// 2. Host read the results from eltwise binary
// 3. Second program runs matmul, using results from step 2 as input activation
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_xy_pair core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;

        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 0;
        int dram_src1_channel_id = 1;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src0_addr, dram_src0_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src1_addr, dram_src1_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, dram_dst_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);

        tt_metal::Program *program1 = setup_program_one(device, core, single_tile_size);

        tt_metal::Program *program2 = setup_program_two(device, core, single_tile_size);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Applications
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CompileProgram(device, program1);

        // Both programs use the same CB addresses but they can be compiled one after
        // the other because they use the same data formats
        pass &= tt_metal::CompileProgram(device, program2);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Program One
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::deprecated::Tensor<bfloat16> src0_tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto src0_activations_tile_layout = convert_to_tile_layout(src0_tensor.get_values());
        auto src0_activations = pack_bfloat16_vec_into_uint32_vec(src0_activations_tile_layout);
        tt_metal::WriteToBuffer(src0_dram_buffer, src0_activations);

        tt::deprecated::Tensor<bfloat16> src1_tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::ZEROS, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto src1_activations_tile_layout = convert_to_tile_layout(src1_tensor.get_values());
        auto src1_activations = pack_bfloat16_vec_into_uint32_vec(src1_activations_tile_layout);
        tt_metal::WriteToBuffer(src1_dram_buffer, src1_activations);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program1);

        write_program_runtime_args_to_device(device, program1, core, num_tiles, src0_dram_buffer, src1_dram_buffer, dst_dram_buffer);

        pass &= tt_metal::LaunchKernels(device, program1);

        std::vector<uint32_t> intermediate_result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, intermediate_result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validatie Intermediate Result
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src0_activations == intermediate_result_vec);  // src1 is ZEROS
        if (pass) {
            log_info(LogTest, "Eltwise binary ran successfully");
        } else {
            log_error(LogTest, "Eltwise binary did not run sucessfully!");
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Program Two
        ////////////////////////////////////////////////////////////////////////////
        // Write matmul weights to DRAM
        auto identity = create_identity_matrix(32, 32, 32); //bflaot16 32x32 identity
        auto weights_tile_layout = convert_to_tile_layout(identity);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        tt_metal::WriteToBuffer(src1_dram_buffer, weights);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program2);

        write_program_runtime_args_to_device(device, program2, core, num_tiles, src0_dram_buffer, src1_dram_buffer, dst_dram_buffer);

        pass &= tt_metal::LaunchKernels(device, program2);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (intermediate_result_vec == result_vec); // src1 is identity matrix

        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}

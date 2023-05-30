#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>

#include "tt_cluster.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "tensix.h"

#include "llrt.hpp"

bool run_add_two_ints(tt_cluster *cluster, int chip_id, const CoreCoord& core) {

    uint64_t test_mailbox_addr = MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_BRISC_OFFSET;
    constexpr int INIT_VALUE = 69;
    constexpr int DONE_VALUE = 1;

    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, test_mailbox_addr);
    log_info(tt::LogVerif, "initialized test_maxilbox");

    std::vector<uint32_t> test_mailbox_init_val_check;
    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
    tt::log_assert(test_mailbox_init_val_check[0] == INIT_VALUE,
        "test_mailbox_init_val_check[0]={} != INIT_VALUE={}",
        test_mailbox_init_val_check[0],
        INIT_VALUE);
    log_info(tt::LogVerif, "checked test_mailbox is correctly initialized to value = {}", test_mailbox_init_val_check[0]);

    tt::llrt::disable_ncrisc(cluster, chip_id, core);
    tt::llrt::disable_triscs(cluster, chip_id, core);

    tt::llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_NCRISC, {core});
    tt::llrt::internal_::enable_cores(cluster, chip_id, {core});

    // Send arguments to L1
    std::uint32_t arg_a = 101;
    std::uint32_t arg_b = 202;
    log_info(tt::LogVerif, "arg_a = {}, arg_b = {}", arg_a, arg_b);
    tt::llrt::write_hex_vec_to_core(cluster, chip_id, core, {arg_a, arg_b}, BRISC_L1_ARG_BASE);

    tt::llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster);

    std::vector<uint32_t> test_mailbox_read_val = {0};
    bool brisc_done = false;
    // TODO: add time-out to this loop
    while(!brisc_done) {
        test_mailbox_read_val = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t

        TT_ASSERT(test_mailbox_read_val[0] == INIT_VALUE || test_mailbox_read_val[0] == DONE_VALUE); // ensure no corruption

        brisc_done = test_mailbox_read_val[0] == DONE_VALUE;

        tt::llrt::internal_::assert_enable_core_mailbox_is_valid_for_core(cluster, chip_id, core);
    }
    log_info(tt::LogVerif, "brisc on core {} finished", core.str());
    log_info(tt::LogVerif, "test_mailbox_read_val = {}", test_mailbox_read_val[0]);

    std::vector<uint32_t> kernel_result = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, BRISC_L1_RESULT_BASE, sizeof(uint32_t));  // read a single uint32_t
    log_info(tt::LogVerif, "kernel result = {}", kernel_result[0]);
    std::uint32_t expected_result = arg_a + arg_b;
    log_info(tt::LogVerif, "expected result = {}", expected_result);

    // TODO: if timed out return also false
    return kernel_result[0] == expected_result;
}

int main(int argc, char** argv)
{
    bool pass = true;

    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name;
    unsigned int core_r;
    unsigned int core_c;
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        std::tie(core_r, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--core-r", 8);
        std::tie(core_c, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--core-c", 2);
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        // tt::llrt::print_worker_cores(cluster);

        // the first worker core starts at (1,1)
        pass = tt::llrt::test_load_write_read_risc_binary(cluster, "built_kernels/add_two_ints/brisc/brisc.hex", 0, {core_r, core_c}, 0);
        if (pass) {
            pass = run_add_two_ints(cluster, 0, {10,2});
        }

        cluster->close_device();
        delete cluster;

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}

#include <iostream>
#include <memory>

#include "detail/tt_metal.hpp"
#include "host_api.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    auto *device = tt::tt_metal::CreateDevice(0);

    std::vector<bfloat16> data = create_random_vector_of_bfloat16_native(4096, 1, 123);
    auto buffer_config = tt::tt_metal::InterleavedBufferConfig{
        .device = device, .size = 4096, .page_size = 32, .buffer_type = tt::tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> buffer = tt::tt_metal::CreateBuffer(buffer_config);

    std::cout << "Created buffer" << std::endl;

    auto tensor =
        ttnn::Tensor(tt::tt_metal::DeviceStorage(buffer), {4096}, tt::tt_metal::DataType::BFLOAT16, ttnn::Layout::TILE);

    std::cout << "Created tensor" << std::endl;

    auto result = ttnn::composite_example_multiple_return(tensor, true, true);

    return 0;
}

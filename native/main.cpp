#include <memory>

#include "common/bfloat16.hpp"
#include "detail/tt_metal.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

int main() {
    printf("Hello, World!\n");

    auto device = tt::tt_metal::CreateDevice(0);

    std::vector<bfloat16> data = create_random_vector_of_bfloat16_native(4096, 1, 123);
    auto buffer_config = tt::tt_metal::InterleavedBufferConfig{
        .device = device, .size = 4096, .page_size = 32, .buffer_type = tt::tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> buffer = tt::tt_metal::CreateBuffer(buffer_config);

    printf("Created buffer\n");

    auto tensor =
        ttnn::Tensor(tt::tt_metal::DeviceStorage(buffer), {4096}, tt::tt_metal::DataType::BFLOAT16, ttnn::Layout::TILE);

    printf("Created a tensor!\n");
    auto result = ttnn::composite_example_multiple_return(tensor);

    std::cout << result << std::endl;
    return 0;
}

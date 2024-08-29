// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul_pybind.hpp"

namespace ttnn::operations::moreh {
void py_module(py::module& module) { moreh_matmul::bind_moreh_matmul_operation(module); }
}  // namespace ttnn::operations::moreh

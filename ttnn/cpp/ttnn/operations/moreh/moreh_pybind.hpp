// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh {
void py_module(py::module &module);
}  // namespace ttnn::operations::moreh

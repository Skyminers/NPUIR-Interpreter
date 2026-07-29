//===- NpyIO.h - Minimal .npy reader/writer ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Just enough NumPy .npy support (v1.0, little-endian, C order) to feed the
// interpreter real inputs and dump its outputs for comparison against a
// numpy/torch golden.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_NPYIO_H
#define BISHENGIR_TOOLS_INTERP_NPYIO_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bishengir {
namespace interp {

struct NpyArray {
  std::string dtype;             // e.g. "<f4"
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;     // raw little-endian element bytes
  unsigned elemBytes = 0;

  int64_t getNumElements() const {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  }
};

/// Read a .npy file. Only C-order, little-endian, non-object arrays are
/// supported; anything else is reported as an error.
llvm::Expected<NpyArray> readNpy(llvm::StringRef path);

/// Write raw element bytes as a .npy file. `dtype` must be a NumPy descriptor
/// such as "<f4".
llvm::Error writeNpy(llvm::StringRef path, llvm::StringRef dtype,
                     llvm::ArrayRef<int64_t> shape,
                     llvm::ArrayRef<uint8_t> data);

/// NumPy descriptor for an MLIR element type. bf16 and the f8 formats have no
/// portable NumPy equivalent and are described as raw unsigned integers of the
/// same width ("<u2" / "|u1"), preserving the bit pattern.
std::string getNpyDType(mlir::Type type);

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_NPYIO_H

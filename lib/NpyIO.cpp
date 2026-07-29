//===- NpyIO.cpp - Minimal .npy reader/writer -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NpyIO.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace mlir;
using namespace llvm;

namespace bishengir {
namespace interp {

static Error makeError(const Twine &msg) {
  return createStringError(inconvertibleErrorCode(), msg.str());
}

/// Element size implied by a NumPy descriptor such as "<f4".
static unsigned dtypeSize(StringRef dtype) {
  if (dtype.size() < 2)
    return 0;
  unsigned size = 0;
  if (dtype.drop_front(2).getAsInteger(10, size))
    return dtype.size() == 2 ? 1 : 0; // "|u1" style with implicit 1
  return size;
}

Expected<NpyArray> readNpy(StringRef path) {
  auto bufOr = MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!bufOr)
    return makeError("cannot open '" + path + "': " + bufOr.getError().message());
  StringRef buf = bufOr.get()->getBuffer();

  if (buf.size() < 10 || buf.substr(0, 6) != StringRef("\x93NUMPY", 6))
    return makeError("'" + path + "' is not a .npy file");

  unsigned major = static_cast<uint8_t>(buf[6]);
  size_t headerLenSize = major >= 2 ? 4 : 2;
  size_t headerStart = 8 + headerLenSize;
  if (buf.size() < headerStart)
    return makeError("'" + path + "' truncated header");

  uint32_t headerLen = 0;
  std::memcpy(&headerLen, buf.data() + 8, headerLenSize);
  if (buf.size() < headerStart + headerLen)
    return makeError("'" + path + "' truncated header");

  StringRef header = buf.substr(headerStart, headerLen);
  NpyArray result;

  // descr
  size_t pos = header.find("'descr'");
  if (pos == StringRef::npos)
    return makeError("'" + path + "' header has no 'descr'");
  size_t q1 = header.find('\'', header.find(':', pos));
  size_t q2 = header.find('\'', q1 + 1);
  if (q1 == StringRef::npos || q2 == StringRef::npos)
    return makeError("'" + path + "' malformed 'descr'");
  result.dtype = header.substr(q1 + 1, q2 - q1 - 1).str();
  if (!result.dtype.empty() && result.dtype[0] == '>')
    return makeError("'" + path + "': big-endian arrays are not supported");

  // fortran_order
  pos = header.find("'fortran_order'");
  if (pos != StringRef::npos && header.find("True", pos) < header.find(',', pos))
    return makeError("'" + path + "': Fortran-order arrays are not supported");

  // shape
  pos = header.find("'shape'");
  if (pos == StringRef::npos)
    return makeError("'" + path + "' header has no 'shape'");
  size_t open = header.find('(', pos);
  size_t close = header.find(')', open);
  if (open == StringRef::npos || close == StringRef::npos)
    return makeError("'" + path + "' malformed 'shape'");
  StringRef shapeStr = header.substr(open + 1, close - open - 1);
  SmallVector<StringRef, 4> dims;
  shapeStr.split(dims, ',');
  for (StringRef dim : dims) {
    dim = dim.trim();
    if (dim.empty())
      continue;
    int64_t v = 0;
    if (dim.getAsInteger(10, v))
      return makeError("'" + path + "' malformed shape entry '" + dim + "'");
    result.shape.push_back(v);
  }

  result.elemBytes = dtypeSize(result.dtype);
  if (result.elemBytes == 0)
    return makeError("'" + path + "': unsupported dtype '" + result.dtype + "'");

  size_t dataOffset = headerStart + headerLen;
  size_t expected =
      static_cast<size_t>(result.getNumElements()) * result.elemBytes;
  if (buf.size() - dataOffset < expected)
    return makeError("'" + path + "': data shorter than shape implies");

  result.data.assign(buf.bytes_begin() + dataOffset,
                     buf.bytes_begin() + dataOffset + expected);
  return result;
}

Error writeNpy(StringRef path, StringRef dtype, ArrayRef<int64_t> shape,
               ArrayRef<uint8_t> data) {
  std::string header;
  raw_string_ostream hs(header);
  hs << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i)
    hs << shape[i] << ((shape.size() == 1 || i + 1 < shape.size()) ? "," : "");
  hs << "), }";
  hs.flush();

  // The header (magic + version + length + dict + '\n') must be a multiple of
  // 64 bytes, per the .npy spec.
  size_t prefix = 10 + header.size() + 1;
  size_t pad = (64 - (prefix % 64)) % 64;
  header.append(pad, ' ');
  header.push_back('\n');

  std::error_code ec;
  raw_fd_ostream out(path, ec, sys::fs::OF_None);
  if (ec)
    return makeError("cannot write '" + path + "': " + ec.message());

  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t headerLen = static_cast<uint16_t>(header.size());
  out.write(reinterpret_cast<const char *>(&headerLen), 2);
  out.write(header.data(), header.size());
  out.write(reinterpret_cast<const char *>(data.data()), data.size());
  out.close();
  return Error::success();
}

std::string getNpyDType(Type type) {
  if (isa<Float32Type>(type))
    return "<f4";
  if (isa<Float64Type>(type))
    return "<f8";
  if (isa<Float16Type>(type))
    return "<f2";
  // No NumPy scalar type for bf16 / f8; hand back the raw bit pattern so the
  // caller can reinterpret with ml_dtypes if it wants to.
  if (isa<BFloat16Type>(type))
    return "<u2";
  if (isa<Float8E4M3FNType>(type) || isa<Float8E5M2Type>(type))
    return "|u1";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width <= 8)
      return "|i1";
    if (width <= 16)
      return "<i2";
    if (width <= 32)
      return "<i4";
    return "<i8";
  }
  if (isa<IndexType>(type))
    return "<i8";
  return "<i8";
}

} // namespace interp
} // namespace bishengir

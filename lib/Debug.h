//===- Debug.h - Replayable interpreter debug sessions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_INTERP_DEBUG_H
#define BISHENGIR_TOOLS_INTERP_DEBUG_H

#include "bishengir/Tools/Interp/Interpreter.h"

#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <vector>

namespace bishengir {
namespace interp {

/// Streams scheduler snapshots and memory deltas as the interpreter runs.
/// The JSONL file is intentionally a replay log rather than a live RPC
/// protocol: consumers can step both forwards and backwards deterministically.
class DebugRecorder {
public:
  DebugRecorder(Interpreter &interp, llvm::StringRef path);

  bool isOpen() const { return stream != nullptr; }
  void recordInitial();
  void recordBeforeStep(const CoreState &core, mlir::Operation *operation);
  void recordStep(const CoreState &core, mlir::Operation *executed,
                  ExecResult result);
  void recordFinal();
  void recordFinish(bool succeeded);
  void memoryWritten(llvm::ArrayRef<ByteRange> ranges);

private:
  struct ValueSnapshot {
    std::string ssa;
    std::string type;
    std::string value;
    bool bound = false;
  };

  void resolveProgramFilter();
  bool recordsCore(const CoreState &core) const;
  bool recordsArena(const Arena &arena) const;
  std::vector<ValueSnapshot> snapshotValues(const CoreState &core,
                                            mlir::ValueRange values);
  llvm::json::Value operationValue(
      mlir::Operation *operation, const CoreState *core = nullptr,
      const std::vector<ValueSnapshot> *inputOverride = nullptr);
  void recordState(llvm::StringRef reason, const CoreState *selected,
                   mlir::Operation *executed, llvm::StringRef result,
                   const std::vector<ValueSnapshot> *executedInputs = nullptr);

  Interpreter &interp;
  std::unique_ptr<llvm::raw_fd_ostream> stream;
  std::vector<std::vector<uint8_t>> previousMemory;
  std::vector<ByteRange> pendingWrites;
  mlir::AsmState::LocationMap irLocations;
  std::unique_ptr<mlir::AsmState> irAsmState;
  std::unique_ptr<mlir::AsmState> operationAsmState;
  std::string irText;
  mlir::Operation *pendingOperation = nullptr;
  unsigned pendingCoreIndex = 0;
  std::vector<ValueSnapshot> pendingInputs;
  std::optional<unsigned> programFilter;
  bool filterResolved = false;
  uint64_t sequence = 0;
};

} // namespace interp
} // namespace bishengir

#endif // BISHENGIR_TOOLS_INTERP_DEBUG_H

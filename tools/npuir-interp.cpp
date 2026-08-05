//===- npuir-interp.cpp - NPUIR interpreter driver --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs memref-form HIVM (NPUIR) on the host, both as a numerical reference
// and - more importantly - as a checker for the synchronisation the compiler
// inserted. See docs/Architecture.md.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Tools/Interp/Interpreter.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace bishengir::interp;

#ifndef NPUIR_INTERP_TOOL_NAME
#define NPUIR_INTERP_TOOL_NAME "npuir-interp"
#endif

#ifndef NPUIR_INTERP_DEBUG_DEFAULT
#define NPUIR_INTERP_DEBUG_DEFAULT ""
#endif

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input MLIR file>"),
                                         llvm::cl::init("-"));

llvm::cl::opt<std::string>
    entryName("entry",
              llvm::cl::desc("Entry function; defaults to the hacc.entry "
                             "function(s) in the module"),
              llvm::cl::init(""));

llvm::cl::list<std::string> argSpecs(
    "args", llvm::cl::CommaSeparated,
    llvm::cl::desc("Per-argument inputs, in order: <file>.npy | zeros | "
                   "poison | arange | <number>"));

llvm::cl::opt<std::string>
    outPrefix("out",
              llvm::cl::desc("Write every GM argument buffer to "
                             "<prefix>arg<N>.npy after the run"),
              llvm::cl::init(""));

llvm::cl::opt<unsigned> blockDim("block-dim",
                                 llvm::cl::desc("Number of blocks to simulate"),
                                 llvm::cl::init(1));

llvm::cl::opt<unsigned>
    subBlockNum("sub-block-num",
                llvm::cl::desc("Split AIV lanes per program (private UB per lane)"),
                llvm::cl::init(1));

llvm::cl::opt<SchedMode> schedMode(
    "sched", llvm::cl::desc("Scheduling / effect-commit model"),
    llvm::cl::values(
        clEnumValN(SchedMode::InOrder, "inorder",
                   "Immediate effects with deterministic core round-robin"),
        clEnumValN(SchedMode::Lazy, "lazy",
                   "Lazy effects with deterministic core round-robin"),
        clEnumValN(SchedMode::Fuzz, "fuzz",
                   "Lazy plus randomised core interleaving")),
    llvm::cl::init(SchedMode::Lazy));

llvm::cl::opt<uint64_t> seed("seed",
                             llvm::cl::desc("Seed for --sched=fuzz"),
                             llvm::cl::init(0));

llvm::cl::opt<uint64_t> gmSize("gm-size",
                               llvm::cl::desc("Global memory pool, in bytes"),
                               llvm::cl::init(64ull * 1024 * 1024));
llvm::cl::opt<uint64_t> ubSize("ub-size", llvm::cl::desc("UB size in bytes"),
                               llvm::cl::init(192 * 1024));
llvm::cl::opt<uint64_t> l1Size("l1-size", llvm::cl::desc("L1 size in bytes"),
                               llvm::cl::init(512 * 1024));
llvm::cl::opt<uint64_t> l0aSize("l0a-size", llvm::cl::desc("L0A size in bytes"),
                                llvm::cl::init(64 * 1024));
llvm::cl::opt<uint64_t> l0bSize("l0b-size", llvm::cl::desc("L0B size in bytes"),
                                llvm::cl::init(64 * 1024));
llvm::cl::opt<uint64_t> l0cSize("l0c-size", llvm::cl::desc("L0C size in bytes"),
                                llvm::cl::init(256 * 1024));
llvm::cl::opt<uint64_t> ssbufSize("ssbuf-size",
                                  llvm::cl::desc("SSBUF size in bytes"),
                                  llvm::cl::init(4ull * 1024 * 1024));
llvm::cl::opt<uint64_t> hostSize(
    "host-size",
    llvm::cl::desc("Scratch pool for non-GM kernel arguments, in bytes"),
    llvm::cl::init(16ull * 1024 * 1024));
llvm::cl::opt<uint64_t>
    dynGmElems("dyn-gm-elems",
               llvm::cl::desc("Element count assumed for a dynamically shaped "
                              "GM argument"),
               llvm::cl::init(4096));

llvm::cl::opt<bool>
    poison("poison",
           llvm::cl::desc("Fill freshly allocated buffers with NaN / 0xCD so "
                          "a missing flag shows up as poison"),
           llvm::cl::init(true));

llvm::cl::list<std::string>
    checks("check", llvm::cl::CommaSeparated,
           llvm::cl::desc("Checks to enable: sync, race, deadlock, oob "
                          "(default: all)"));

llvm::cl::opt<bool> checkRawPointerRaces(
    "check-raw-pointer-races",
    llvm::cl::desc("Also report races where both sides are raw llvm.load / "
                   "llvm.store flag-scratchpad traffic"),
    llvm::cl::init(false));

llvm::cl::opt<bool>
    exactLayout("exact-layout",
                llvm::cl::desc("Byte-exact fractal layout (not implemented)"),
                llvm::cl::init(false));

llvm::cl::opt<std::string>
    traceFile("trace",
              llvm::cl::desc("Write a per-op execution trace to this file"),
              llvm::cl::init(""));

llvm::cl::opt<std::string> debugFile(
    "debug-output",
    llvm::cl::desc("Write a replayable JSONL debug session to this file"),
    llvm::cl::init(NPUIR_INTERP_DEBUG_DEFAULT));

llvm::cl::opt<std::string> debugCore(
    "debug-core",
    llvm::cl::desc("Record the single-program AIC/AIV slice containing this "
                   "lane (for example, AIV#0.0)"),
    llvm::cl::init(""));

llvm::cl::opt<uint64_t>
    maxSteps("max-steps",
             llvm::cl::desc("Abort after this many interpreted ops"),
             llvm::cl::init(200ull * 1000 * 1000));

llvm::cl::opt<bool> verbose("v", llvm::cl::desc("Verbose progress output"),
                            llvm::cl::init(false));

llvm::cl::opt<bool> useTargetSizes(
    "use-target-sizes",
    llvm::cl::desc("Take UB/L1/L0 capacities from the module's "
                   "dlti.target_system_spec instead of the flags"),
    llvm::cl::init(true));

void printVersion(llvm::raw_ostream &os) {
  os << NPUIR_INTERP_TOOL_NAME << ' ' << NPUIR_INTERPRETER_VERSION << '\n';
}

/// Read the on-chip memory sizes the compiler was targeting, so the capacity
/// checks match the real device instead of the flag defaults.
void applyTargetSizes(ModuleOp module, InterpOptions &options) {
  auto spec = module->getAttr("dlti.target_system_spec");
  if (!spec)
    return;
  // The spec prints as a nest of dl_entry pairs; scanning the text keeps this
  // independent of the DLTI/HACC attribute classes.
  std::string text;
  llvm::raw_string_ostream os(text);
  spec.print(os);
  StringRef s = os.str();

  auto readEntry = [&](StringRef name, uint64_t &out) {
    size_t pos = s.find(("\"" + name + "\"").str());
    if (pos == StringRef::npos)
      return;
    size_t comma = s.find(',', pos);
    if (comma == StringRef::npos)
      return;
    StringRef rest = s.substr(comma + 1).ltrim();
    uint64_t value = 0;
    size_t end = 0;
    while (end < rest.size() && llvm::isDigit(rest[end]))
      ++end;
    if (end == 0)
      return;
    if (!rest.substr(0, end).getAsInteger(10, value) && value > 0)
      out = value;
  };
  readEntry("UB_SIZE", options.ubSize);
  readEntry("L1_SIZE", options.l1SizeBytes);
  readEntry("L0A_SIZE", options.l0aSize);
  readEntry("L0B_SIZE", options.l0bSize);
  readEntry("L0C_SIZE", options.l0cSize);
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::SetVersionPrinter(printVersion);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      NPUIR_INTERP_TOOL_NAME
      " - CPU interpreter for memref-form NPUIR (HIVM)\n\n"
      "Runs a compiled kernel on the host to check both its numerics and the "
      "synchronisation the compiler inserted. Deferred pipe effects mean a "
      "missing flag shows up as poison, a data race, or a deadlock rather "
      "than silently working.\n");

  DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::DLTIDialect,
                  mlir::func::FuncDialect,
                  mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect,
                  mlir::annotation::AnnotationDialect,
                  mlir::hacc::HACCDialect,
                  mlir::hfusion::HFusionDialect,
                  mlir::hivm::HIVMDialect,
                  mlir::scope::ScopeDialect>();

  MLIRContext context(registry);
  context.allowUnregisteredDialects(false);

  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << "error: " << errorMessage << '\n';
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "error: could not parse '" << inputFilename << "'\n";
    return 1;
  }

  InterpOptions options;
  options.entry = entryName;
  options.args.assign(argSpecs.begin(), argSpecs.end());
  options.outPrefix = outPrefix;
  options.blockDim = std::max(1u, blockDim.getValue());
  options.subBlockNum = std::max(1u, subBlockNum.getValue());
  options.sched = schedMode;
  options.seed = seed;
  options.gmSize = gmSize;
  options.ubSize = ubSize;
  options.l1SizeBytes = l1Size;
  options.l0aSize = l0aSize;
  options.l0bSize = l0bSize;
  options.l0cSize = l0cSize;
  options.ssbufSize = ssbufSize;
  options.hostSize = hostSize;
  options.dynGmElems = dynGmElems;
  options.poison = poison;
  options.checkRawPointerRaces = checkRawPointerRaces;
  options.exactLayout = exactLayout;
  options.traceFile = traceFile;
  options.debugFile = debugFile;
  options.debugCore = debugCore;
  options.maxSteps = maxSteps;
  options.verbose = verbose;

  if (!checks.empty()) {
    options.checkRace = options.checkDeadlock = options.checkOob = false;
    options.checkSync = false;
    for (const std::string &check : checks) {
      if (check == "race")
        options.checkRace = true;
      else if (check == "deadlock")
        options.checkDeadlock = true;
      else if (check == "oob")
        options.checkOob = true;
      else if (check == "sync")
        options.checkSync = true;
      else if (check == "none" || check == "off")
        ; // everything already disabled
      else {
        llvm::errs() << "error: unknown check '" << check
                     << "' (expected sync, race, deadlock or oob)\n";
        return 1;
      }
    }
  }

  // Flags win over the target spec only when the user set them explicitly.
  if (useTargetSizes) {
    InterpOptions fromTarget = options;
    applyTargetSizes(*module, fromTarget);
    if (ubSize.getNumOccurrences() == 0)
      options.ubSize = fromTarget.ubSize;
    if (l1Size.getNumOccurrences() == 0)
      options.l1SizeBytes = fromTarget.l1SizeBytes;
    if (l0aSize.getNumOccurrences() == 0)
      options.l0aSize = fromTarget.l0aSize;
    if (l0bSize.getNumOccurrences() == 0)
      options.l0bSize = fromTarget.l0bSize;
    if (l0cSize.getNumOccurrences() == 0)
      options.l0cSize = fromTarget.l0cSize;
  }

  Interpreter interpreter(*module, options);
  LogicalResult result = interpreter.run();
  // Outputs are worth dumping even after a failed run: seeing the poison is
  // often how you localise the missing flag.
  if (failed(interpreter.dumpOutputs()))
    return 1;
  return succeeded(result) ? 0 : 1;
}

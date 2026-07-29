cmake_minimum_required(VERSION 3.20.0)

project(NPUIRInterpreter LANGUAGES C CXX)

# This project is configured as an LLVM external project after AscendNPU-IR's
# `bishengir` project.  That gives us the exact MLIR and generated dialect
# targets belonging to the pinned submodule revision.
if(NOT COMMAND add_mlir_tool OR NOT TARGET MLIRHIVMDialect)
  message(FATAL_ERROR
    "NPUIR-Interpreter must be configured with build.sh, or as an LLVM "
    "external project after AscendNPU-IR (bishengir).")
endif()

set(NPUIR_INTERPRETER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NPUIR_INTERPRETER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(ASCENDNPU_IR_SOURCE_DIR
  "${NPUIR_INTERPRETER_SOURCE_DIR}/third_party/AscendNPU-IR")
set(BISHENGIR_MAIN_INCLUDE_DIR
  "${ASCENDNPU_IR_SOURCE_DIR}/bishengir/include")
set(BISHENGIR_INCLUDE_DIR "${LLVM_BINARY_DIR}/tools/bishengir/include")
set(MLIR_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/../mlir/include")
set(MLIR_INCLUDE_DIR "${LLVM_BINARY_DIR}/tools/mlir/include")

add_subdirectory(lib)
add_subdirectory(tools)

option(NPUIR_INTERPRETER_INCLUDE_TESTS "Build NPUIR interpreter tests" ON)
if(NPUIR_INTERPRETER_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()

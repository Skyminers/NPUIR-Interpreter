add_library(NPUIRInterpreter STATIC
  Debug.cpp
  DialectCompat.cpp
  Interpreter.cpp
  Memory.cpp
  NpyIO.cpp
  OpUtils.cpp
  OpsCommunity.cpp
  OpsElementwise.cpp
  OpsIndirect.cpp
  OpsMemory.cpp
  OpsMisc.cpp
  OpsShape.cpp
  OpsSync.cpp
  OpsVector.cpp
  PipeEngine.cpp
  Value.cpp
)

llvm_update_compile_flags(NPUIRInterpreter)

target_include_directories(NPUIRInterpreter
  PUBLIC
    $<BUILD_INTERFACE:${NPUIR_INTERPRETER_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BISHENGIR_MAIN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BISHENGIR_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${MLIR_MAIN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIR}>
)

target_link_libraries(NPUIRInterpreter PUBLIC
  MLIRHIVMDialect
  MLIRHACCDialect
  MLIRAnnotationDialect
  BiShengIRScopeDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRVectorDialect
)

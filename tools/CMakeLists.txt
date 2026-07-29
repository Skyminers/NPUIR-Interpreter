add_mlir_tool(npuir-interp
  npuir-interp.cpp
)

target_include_directories(npuir-interp PRIVATE
  ${NPUIR_INTERPRETER_SOURCE_DIR}/include
  ${BISHENGIR_MAIN_INCLUDE_DIR}
  ${BISHENGIR_INCLUDE_DIR}
  ${MLIR_MAIN_INCLUDE_DIR}
  ${MLIR_INCLUDE_DIR}
)

target_compile_definitions(npuir-interp PRIVATE
  NPUIR_INTERPRETER_VERSION="1.0.0"
)

target_link_libraries(npuir-interp PRIVATE
  NPUIRInterpreter
  MLIRAnnotationDialect
  MLIRHACCDialect
  MLIRHFusionDialect
  MLIRHIVMDialect
  BiShengIRScopeDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRIR
  MLIRSCFDialect
  MLIRSupport
  MLIRVectorDialect
)

mlir_check_all_link_libraries(npuir-interp)

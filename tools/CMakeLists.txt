function(configure_npuir_interp_tool target)
  target_include_directories(${target} PRIVATE
    ${NPUIR_INTERPRETER_SOURCE_DIR}/include
    ${BISHENGIR_MAIN_INCLUDE_DIR}
    ${BISHENGIR_INCLUDE_DIR}
    ${MLIR_MAIN_INCLUDE_DIR}
    ${MLIR_INCLUDE_DIR}
  )

  target_compile_definitions(${target} PRIVATE
    NPUIR_INTERPRETER_VERSION="1.0.0"
    NPUIR_INTERP_TOOL_NAME="${target}"
  )

  target_link_libraries(${target} PRIVATE
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

  mlir_check_all_link_libraries(${target})
endfunction()

add_mlir_tool(npuir-interp npuir-interp.cpp)
configure_npuir_interp_tool(npuir-interp)

add_mlir_tool(npuir-interp-debug npuir-interp.cpp)
target_compile_definitions(npuir-interp-debug PRIVATE
  NPUIR_INTERP_DEBUG_DEFAULT="npuir-debug.jsonl"
)
configure_npuir_interp_tool(npuir-interp-debug)

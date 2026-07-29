add_custom_target(NPUIRInterpreterUnitTests)

add_unittest(NPUIRInterpreterUnitTests NPUIRInterpreterTests
  PipeEngineTest.cpp
  ShadowMemoryTest.cpp
  ValueTest.cpp
)

target_include_directories(NPUIRInterpreterTests PRIVATE
  ${NPUIR_INTERPRETER_SOURCE_DIR}/include
)

target_link_libraries(NPUIRInterpreterTests PRIVATE
  NPUIRInterpreter
  MLIRIR
)

add_custom_target(check-npuir-interpreter-unit
  COMMAND $<TARGET_FILE:NPUIRInterpreterTests>
  DEPENDS NPUIRInterpreterTests
  USES_TERMINAL
)

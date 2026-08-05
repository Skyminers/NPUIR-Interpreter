configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-npuir-interpreter
  "Running the NPUIR interpreter regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS npuir-interp npuir-interp-debug bishengir-compile FileCheck count not
)

set_target_properties(check-npuir-interpreter PROPERTIES FOLDER "Tests")

add_custom_target(check-npuir-interpreter-precision
  COMMAND ${CMAKE_COMMAND} -E env
          NPUIR_INTERP=$<TARGET_FILE:npuir-interp>
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Precision/sweep_binary.py
  COMMAND ${CMAKE_COMMAND} -E env
          NPUIR_INTERP=$<TARGET_FILE:npuir-interp>
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Precision/sweep_exhaustive.py
  COMMAND ${CMAKE_COMMAND} -E env
          NPUIR_INTERP=$<TARGET_FILE:npuir-interp>
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Precision/sweep_integer.py
  COMMAND ${CMAKE_COMMAND} -E env
          NPUIR_INTERP=$<TARGET_FILE:npuir-interp>
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Precision/sweep_lowprec.py
  DEPENDS npuir-interp
  WORKING_DIRECTORY ${NPUIR_INTERPRETER_SOURCE_DIR}
  USES_TERMINAL
)

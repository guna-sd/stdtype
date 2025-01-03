llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(STDTYPE_TEST_DEPENDS
        FileCheck count not
        stdtype-opt
        stdtype-translate
        StdTypePlugin
        )

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND STDTYPE_TEST_DEPENDS StdTypePythonModules)
endif()

add_lit_testsuite(check-stdtype "Running the stdtype regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${STDTYPE_TEST_DEPENDS}
        )
set_target_properties(check-stdtype PROPERTIES FOLDER "Tests")

add_lit_testsuites(STDTYPE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${STDTYPE_TEST_DEPENDS})
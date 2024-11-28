function(add_mlir_dialect target)
    target_include_directories(${target} PRIVATE ${MLIR_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE ${LLVM_COMPILE_FLAGS})
    llvm_update_compile_flags(${target})
endfunction()

function(add_mlir_ops target)
    target_include_directories(${target} PRIVATE ${MLIR_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE ${LLVM_COMPILE_FLAGS})
    llvm_update_compile_flags(${target})
endfunction()

function(process_mlir_file input_file output_file)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND stdtype-opt ${input_file} -o ${output_file}
        DEPENDS stdtype-opt ${input_file}
        COMMENT "Processing MLIR file: ${input_file}"
    )
endfunction()

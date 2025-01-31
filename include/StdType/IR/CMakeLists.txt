set(LLVM_TARGET_DEFINITIONS StdTypes.td)
mlir_tablegen(StdTypes.h.inc -gen-typedef-decls -typedefs-dialect=stdtype)
mlir_tablegen(StdTypeDialect.h.inc -gen-typedef-decls -typedefs-dialect=stdtype)
mlir_tablegen(StdTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=stdtype)
add_public_tablegen_target(MLIRStdTypesIncGen)


# Generated File	Corresponding .td File
# MyCustomOps.h.inc	MyCustomOps.td
# MyCustomOps.cpp.inc	MyCustomOps.td
# MyCustomDialect.h.inc	MyCustomOps.td
# MyCustomDialect.cpp.inc	MyCustomOps.td
# MyCustomOpsEnums.h.inc	MyCustomOps.td
# MyCustomOpsEnums.cpp.inc	MyCustomOps.td
# MyCustomOpsAttrDefs.h.inc	MyCustomOps.td
# MyCustomOpsAttrDefs.cpp.inc	MyCustomOps.td
# MyCustomTypes.h.inc	MyCustomTypes.td
# MyCustomTypes.cpp.inc	MyCustomTypes.td
# MyCustomIntrinsicOps.h.inc	MyCustomIntrinsicOps.td
# MyCustomIntrinsicOps.cpp.inc	MyCustomIntrinsicOps.td
# MyCustomInterfaces.h.inc	MyCustomInterfaces.td
# MyCustomInterfaces.cpp.inc	MyCustomInterfaces.td
# MyCustomAttrInterfaces.h.inc	MyCustomInterfaces.td
# MyCustomAttrInterfaces.cpp.inc	MyCustomInterfaces.td
# MyCustomTypeInterfaces.h.inc	MyCustomInterfaces.td
# MyCustomTypeInterfaces.cpp.inc	MyCustomInterfaces.td
# MyCustomConversions.inc	MyCustomConversions.td
# MyCustomConversionEnumsToLLVM.inc	MyCustomConversions.td
# MyCustomConversionEnumsFromLLVM.inc	MyCustomConversions.td
# MyCustomOpFromLLVMIRConversions.inc	MyCustomConversions.td
# MyCustomIntrinsicConversions.inc	MyCustomIntrinsicConversions.td
# MyCustomIntrinsicFromLLVMIRConversions.inc	MyCustomIntrinsicConversions.td
# MyCustomConvertibleLLVMIRIntrinsics.inc	MyCustomIntrinsicConversions.td
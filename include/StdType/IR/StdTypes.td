include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

include "include/StdType/IR/StdTypeDialect.td"


//===----------------------------------------------------------------------===//
// Base Definition for StdTypes
//===----------------------------------------------------------------------===//

class StdType<string Name, string typeMnemonic, list<Trait> traits = [],
              string baseCppClass = "::mlir::Type">
  : TypeDef<StdType_Dialect, Name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = "std." # typeMnemonic;
}

//===----------------------------------------------------------------------===//
// StdIntegerType
//===----------------------------------------------------------------------===//

def StdInt : StdType<"Int", "int"> {
  let summary = "Signed Integer type with a parameterized bit width";
  let description = [{
    Syntax:
    ```
    int-type ::= `!std.int` `<` bit-width `>`
    ```
    The value of `int-type` represents an signed integer type with a parameterized bit width.
    The bit-width must be a positive integer.

    #### Example:
    ```
    !std.int<8>
    !std.int<16>
    ```
  }];
  
let parameters = (ins "unsigned":$width);
let assemblyFormat = "`<` $width `>`";
let genVerifyDecl = 1;

let builders = [ TypeBuilder<(ins "unsigned":$width), [{
    return $_get($_ctxt, width);
    }]>
  ];
}

def StdUInt : StdType<"UInt", "uint"> {
  let summary = "Unsigned integer type with a parameterized bit width";
  let description = [{
    Syntax:
    ```
    integer-type ::= `!std.uint` `<` bit-width `>`
    ```
    The value of `int-type` represents an unsigned integer type with a parameterized bit width.
    The bit-width must be a positive integer.

    #### Example:
    ```mlir
    !std.uint<8>
    !std.uint<16>
    !std.uint<32>
    !std.uint<64>
    ```
  }];
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
  let builders = [ TypeBuilder<(ins "unsigned":$width), [{
      return $_get($_ctxt, width);
    }]>
  ];
}

def StdIndex : StdType<"Index", "index"> {
  let summary = "Index type, typically used for representing sizes and indices";
  let description = [{
    Syntax:

    ```
    index-type ::= `!std.index`
    ```

    The value of `index-type` represents an index type, typically used for array sizes
    or other indexing purposes. The width of the index type may be architecture-dependent,
    and it is typically used for indexing operations within the MLIR system.

    #### Example:

    ```mlir
    !std.index
    ```
  }];
}

def StdInt2 : StdType<"Int2", "si2"> {
  let summary = "2-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si2
    ```
    Represents a 2-bit signed integer type.
    
    #### Example:
    ```
    !std.si2
    ```
  }];
}

def StdInt4 : StdType<"Int4", "si4"> {
  let summary = "4-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si4
    ```
    Represents a 4-bit signed integer type.
    
    #### Example:
    ```
    !std.si4
    ```
  }];
}

def StdInt6 : StdType<"Int6", "si6"> {
  let summary = "6-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si6
    ```
    Represents a 6-bit signed integer type.
    
    #### Example:
    ```
    !std.si6
    ```
  }];
}

def StdInt8 : StdType<"Int8", "si8"> {
  let summary = "8-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si8
    ```
    Represents a standard 8-bit signed integer type.
    
    #### Example:
    ```
    !std.si8
    ```
  }];
}

def StdInt16 : StdType<"Int16", "si16"> {
  let summary = "16-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si16
    ```
    Represents a standard 16-bit signed integer type.
    
    #### Example:
    ```
    !std.si16
    ```
  }];
}

def StdInt32 : StdType<"Int32", "si32"> {
  let summary = "32-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si32
    ```
    Represents a standard 32-bit signed integer type.
    
    #### Example:
    ```
    !std.si32
    ```
  }];
}

def StdInt64 : StdType<"Int64", "si64"> {
  let summary = "64-bit signed integer type";
  let description = [{
    Syntax:
    ```
    !std.si64
    ```
    Represents a standard 64-bit signed integer type.
    
    #### Example:
    ```
    !std.si64
    ```
  }];
}

def StdUInt2 : StdType<"UInt2", "u2"> {
  let summary = "2-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u2
    ```
    Represents a 2-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u2
    ```
  }];
}

def StdUInt4 : StdType<"UInt4", "u4"> {
  let summary = "4-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u4
    ```
    Represents a 4-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u4
    ```
  }];
}

def StdUInt6 : StdType<"UInt6", "u6"> {
  let summary = "6-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u6
    ```
    Represents a 6-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u6
    ```
  }];
}

def StdUInt8 : StdType<"UInt8", "u8"> {
  let summary = "8-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u8
    ```
    Represents a standard 8-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u8
    ```
  }];
}

def StdUInt16 : StdType<"UInt16", "u16"> {
  let summary = "16-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u16
    ```
    Represents a standard 16-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u16
    ```
  }];
}

def StdUInt32 : StdType<"UInt32", "u32"> {
  let summary = "32-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u32
    ```
    Represents a standard 32-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u32
    ```
  }];
}

def StdUInt64 : StdType<"UInt64", "u64"> {
  let summary = "64-bit unsigned integer type";
  let description = [{
    Syntax:
    ```
    !std.u64
    ```
    Represents a standard 64-bit unsigned integer type.
    
    #### Example:
    ```
    !std.u64
    ```
  }];
}

def StdI1 : StdType<"I1", "i1"> {
  let summary = "1-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i1
    ```
    Represents a 1-bit signless integer type.
    
    #### Example:
    ```
    !std.i1
    ```
  }];
}

def StdI2 : StdType<"I2", "i2"> {
  let summary = "2-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i2
    ```
    Represents a 2-bit signless integer type.
    
    #### Example:
    ```
    !std.i2
    ```
  }];
}

def StdI4 : StdType<"I4", "i4"> {
  let summary = "4-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i4
    ```
    Represents a 4-bit signless integer type.
    
    #### Example:
    ```
    !std.i4
    ```
  }];
}

def StdI6 : StdType<"I6", "i6"> {
  let summary = "6-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i6
    ```
    Represents a 6-bit signless integer type.
    
    #### Example:
    ```
    !std.i6
    ```
  }];
}

def StdI8 : StdType<"I8", "i8"> {
  let summary = "8-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i8
    ```
    Represents an 8-bit signless integer type.
    
    #### Example:
    ```
    !std.i8
    ```
  }];
}

def StdI16 : StdType<"I16", "i16"> {
  let summary = "16-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i16
    ```
    Represents a 16-bit signless integer type.
    
    #### Example:
    ```
    !std.i16
    ```
  }];
}

def StdI32 : StdType<"I32", "i32"> {
  let summary = "32-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i32
    ```
    Represents a 32-bit signless integer type.
    
    #### Example:
    ```
    !std.i32
    ```
  }];
}

def StdI64 : StdType<"I64", "i64"> {
  let summary = "64-bit signless integer type";
  let description = [{
    Syntax:
    ```
    !std.i64
    ```
    Represents a 64-bit signless integer type.
    
    #### Example:
    ```
    !std.i64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StdFloatType
//===----------------------------------------------------------------------===//

def StdFloat : StdType<"Float", "float"> {
  let summary = "Floating-point type with a parameterized precision";
  let description = [{
    Syntax:
    ```
    float-type ::= `!std.float` `<` precision `>`
    ```
    The value of `float-type` represents a floating-point type with a parameterized precision.
    The precision must be a positive integer.

    #### Example:
    ```
    !std.float<32>
    !std.float<64>
    ```
  }];
  
let parameters = (ins "unsigned":$width);
let assemblyFormat = "`<` $width `>`";
let genVerifyDecl = 1;

let builders = [
TypeBuilderWithInferredContext<(ins), [{
    return $_get($_ctxt, 32);  // Default to 32-bit float
}]>,
TypeBuilder<(ins "unsigned":$width), [{
    return $_get($_ctxt, width);
}]>
];
}

def StdF4E2M1FN : StdType<"F4E2M1FN", "f4E2M1FN"> {
  let summary = "4-bit floating point with 2-bit exponent and 1-bit mantissa";
  let description = [{
    Syntax:
    ```
    !std.f4E2M1FN
    ```
    Represents a custom 4-bit floating point type with 2 bits for the exponent and 1 bit for the mantissa.
    This is a non-standard floating point type.
    
    #### Example:
    ```
    !std.f4E2M1FN
    ```
  }];
}

def StdF6E2M3FN : StdType<"F6E2M3FN", "f6E2M3FN"> {
  let summary = "6-bit floating point with 2-bit exponent and 3-bit mantissa";
  let description = [{
    Syntax:
    ```
    !std.f6E2M3FN
    ```
    Represents a custom 6-bit floating point type with 2 bits for the exponent and 3 bits for the mantissa.
    This is a non-standard floating point type.
    
    #### Example:
    ```
    !std.f6E2M3FN
    ```
  }];
}

def StdF6E3M2FN : StdType<"F6E3M2FN", "f6E3M2FN"> {
  let summary = "6-bit floating point with 3-bit exponent and 2-bit mantissa";
  let description = [{
    Syntax:
    ```
    !std.f6E3M2FN
    ```
    Represents a custom 6-bit floating point type with 3 bits for the exponent and 2 bits for the mantissa.
    This is a non-standard floating point type.
    
    #### Example:
    ```
    !std.f6E3M2FN
    ```
  }];
}

def StdF8E3M4 : StdType<"F8E3M4", "f8E3M4"> {
  let summary = "8-bit floating point with 3-bit exponent and 4-bit mantissa";
  let description = [{
    Syntax:
    ```
    !std.f8E3M4
    ```
    Represents an 8-bit floating point type with 3 bits for the exponent and 4 bits for the mantissa.
    
    #### Example:
    ```
    !std.f8E3M4
    ```
  }];
}

def StdF8E4M3 : StdType<"F8E4M3", "f8E4M3"> {
  let summary = "8-bit floating point with 4-bit exponent and 3-bit mantissa";
  let description = [{
    Syntax:
    ```
    !std.f8E4M3
    ```
    Represents an 8-bit floating point type with 4 bits for the exponent and 3 bits for the mantissa.
    
    #### Example:
    ```
    !std.f8E4M3
    ```
  }];
}

def StdF8E4M3FN : StdType<"F8E4M3FN", "f8E4M3FN"> {
  let summary = "8-bit floating point with 4-bit exponent and 3-bit mantissa (Non-Standard)";
  let description = [{
    Syntax:
    ```
    !std.f8E4M3FN
    ```
    Represents a custom 8-bit floating point type with 4 bits for the exponent and 3 bits for the mantissa (non-standard).
    
    #### Example:
    ```
    !std.f8E4M3FN
    ```
  }];
}

def StdF8E4M3FNUZ : StdType<"F8E4M3FNUZ", "f8E4M3FNUZ"> {
  let summary = "8-bit floating point with 4-bit exponent, 3-bit mantissa, and zero-based normalization";
  let description = [{
    Syntax:
    ```
    !std.f8E4M3FNUZ
    ```
    Represents an 8-bit floating point type with 4 bits for the exponent, 3 bits for the mantissa, and zero-based normalization.
    
    #### Example:
    ```
    !std.f8E4M3FNUZ
    ```
  }];
}

def StdF8E4M3B11FNUZ : StdType<"F8E4M3B11FNUZ", "f8E4M3B11FNUZ"> {
  let summary = "8-bit floating point with 4-bit exponent, 3-bit mantissa, and 11-bit biased exponent with zero normalization";
  let description = [{
    Syntax:
    ```
    !std.f8E4M3B11FNUZ
    ```
    Represents an 8-bit floating point type with 4 bits for the exponent, 3 bits for the mantissa, and an 11-bit biased exponent with zero normalization.
    
    #### Example:
    ```
    !std.f8E4M3B11FNUZ
    ```
  }];
}

def StdF8E5M2 : StdType<"F8E5M2", "f8E5M2"> {
  let summary = "8-bit floating point with 5-bit exponent and 2-bit mantissa";
  let description = [{
    Syntax:
    ```
    !std.f8E5M2
    ```
    Represents an 8-bit floating point type with 5 bits for the exponent and 2 bits for the mantissa.
    
    #### Example:
    ```
    !std.f8E5M2
    ```
  }];
}

def StdF8E5M2FNUZ : StdType<"F8E5M2FNUZ", "f8E5M2FNUZ"> {
  let summary = "8-bit floating point with 5-bit exponent, 2-bit mantissa, and zero-based normalization";
  let description = [{
    Syntax:
    ```
    !std.f8E5M2FNUZ
    ```
    Represents an 8-bit floating point type with 5 bits for the exponent, 2 bits for the mantissa, and zero-based normalization.
    
    #### Example:
    ```
    !std.f8E5M2FNUZ
    ```
  }];
}

def StdF8E8M0FNU : StdType<"F8E8M0FNU", "f8E8M0FNU"> {
  let summary = "8-bit floating point with 8-bit exponent and no mantissa";
  let description = [{
    Syntax:
    ```
    !std.f8E8M0FNU
    ```
    Represents an 8-bit floating point type with 8 bits for the exponent and no mantissa (special case).
    
    #### Example:
    ```
    !std.f8E8M0FNU
    ```
  }];
}

def StdF16 : StdType<"F16", "f16"> {
  let summary = "16-bit floating point type";
  let description = [{
    Syntax:
    ```
    !std.f16
    ```
    Represents a standard 16-bit floating point type.
    
    #### Example:
    ```
    !std.f16
    ```
  }];
}

def StdF32 : StdType<"F32", "f32"> {
  let summary = "32-bit floating point type (standard IEEE-754)";
  let description = [{
    Syntax:
    ```
    !std.f32
    ```
    Represents a standard 32-bit floating point type as per IEEE-754 specification.
    
    #### Example:
    ```
    !std.f32
    ```
  }];
}

def StdF64 : StdType<"F64", "f64"> {
  let summary = "64-bit floating point type (standard IEEE-754)";
  let description = [{
    Syntax:
    ```
    !std.f64
    ```
    Represents a standard 64-bit floating point type as per IEEE-754 specification.
    
    #### Example:
    ```
    !std.f64
    ```
  }];
}

def StdBF16 : StdType<"BF16", "bf16"> {
  let summary = "BFloat16 floating point type";
  let description = [{
    Syntax:
    ```
    !std.bf16
    ```
    Represents the BFloat16 floating point type, typically used in machine learning for reduced precision.
    
    #### Example:
    ```
    !std.bf16
    ```
  }];
}



//===----------------------------------------------------------------------===//
// StdBoolType
//===----------------------------------------------------------------------===//

def StdBoolType : StdType<"Bool", "bool"> {
  let summary = "Boolean type";
  let description = [{
    Syntax:
    ```
    bool-type ::= `!std.bool`
    ```
    Represents a boolean type with two values: `true` and `false`.

    #### Example:
    ```
    !std.bool
    ```
  }];
  let genVerifyDecl = 1;
  let builders = [TypeBuilderWithInferredContext<(ins)>];
}

//===----------------------------------------------------------------------===//
// StdStringType
//===----------------------------------------------------------------------===//

def StdStringType : StdType<"String", "string"> {
  let summary = "String type";
  let description = [{
    Syntax:
    ```
    string-type ::= `!std.string`
    ```
    Represents a string type.

    #### Example:
    ```
    !std.string
    ```
  }];
  let genVerifyDecl = 1;
  let builders = [TypeBuilderWithInferredContext<(ins)>];
}

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

def StdComplex : StdType<"Complex", "complex"> {
  let summary = "Complex number with a parameterized element type";
  let description = [{
    Syntax:
    ```
    complex-type ::= `!std.complex` `<` type `>`
    ```
    Represents a complex number composed of real and imaginary values
    of the parameterized element type. Element must be a scalar type.

    #### Examples:
    ```
    !std.complex<f32>
    !std.complex<i32>
    ```
  }];
  
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` custom<PrettyTypes($elementType) `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// StdArrayType
//===----------------------------------------------------------------------===//

def StdArrayType : StdType<"Array", "array", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>,
    DeclareTypeInterfaceMethods<DestructurableTypeInterface>]> {
  let summary = "An array type representing a sequence of elements.";
  let description = [{
    Represents a fixed-size array of elements.
    Syntax:
    ```
    array-type ::= `!std.array` `<` numElements `x` elementType `>`
    ```

    #### Example:
    ```
    !std.array<4 x i32>
    ```
  }];
  let parameters = (ins "Type":$elementType, "uint64_t":$numElements);
  let assemblyFormat = "`<` $numElements `x` $elementType `>`";
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "uint64_t":$numElements)>
  ];
  let extraClassDeclaration = [{
  static bool isValidType(Type type);
  static LLVM::ArrayType toLLVMArrayType(MLIRContext *context, Type elementType, uint64_t numElements);
  }];
}

//===----------------------------------------------------------------------===//
// StdPointerType
//===----------------------------------------------------------------------===//

def StdPointerType : StdType<"Pointer", "ptr", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "getIndexBitwidth", "areCompatible", "verifyEntries"]>]> {
  let summary = "Pointer type with an optional address space.";
  let description = [{
    Syntax:
    ```
    pointer-type ::= `!std.ptr` `<` addressSpace? `>`
    ```

    Represents a pointer type with an optional address space.

    #### Example:
    ```
    !std.ptr
    !std.ptr<1>
    ```
  }];
  let parameters = (ins DefaultValuedParameter<"unsigned", "0">:$addressSpace);
  let assemblyFormat = "(`<` $addressSpace^ `>`)?";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins DefaultValuedParameter<"unsigned", "0">:$addressSpace), [{
      return $_get($_ctxt, addressSpace);
    }]>
  ];
  let extraClassDeclaration = [{
    static LLVM::PointerType toLLVMPointerType(MLIRContext *context, uint64_t addressSpace);
  }];
}

//===----------------------------------------------------------------------===//
// StdTensorType
//===----------------------------------------------------------------------===//

// working on it...!

// #ifndef __STDBASE
// #define __STDBASE

// include "mlir/Interfaces/InferTypeOpInterface.td"
// include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/IR/AttrTypeBase.td"
// include "mlir/IR/OpBase.td"

// def StdBool : AnyTypeOf[StdI1, I1, StdBoolType];

// def StdSignlessInt : AnyTypeOf[[SignlessIntOfWidths<[1, 2, 4, 6, 8, 16, 32, 64]>, StdI1, StdI2, StdI3, StdI4, StdI8, StdI16, StdI32, StdI64]];
// def StdSInt : AnyTypeOf[[SignedIntOfWidths<[2, 4, 6, 8, 16, 32, 64]>, StdInt2, StdInt4, StdInt6, StdInt8, StdInt16, StdInt32, StdInt64]];
// def StdUInt : AnyTypeOf[[UnsignedIntOfWidths<[2, 4, 6, 8, 16, 32, 64]>, StdUInt2, StdUInt4, StdUInt6, StdUInt8, StdUInt16, StdUInt32, StdUInt64]];
// def StdInt : AnyTypeOf<[StdSignlessInt, StdSInt, StdUInt]>;

// def StdFloat : AnyTypeOf<[F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
//                            F8E4M3FN, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2,
//                            F8E5M2FNUZ, F8E8M0FNU, F16, F32, F64, BF16, StdBF16, StdF16, StdF32, StdF64,
//                            StdF4E2M1FN, StdF6E2M3FN, StdF6E3M2FN, StdF8E3M4, StdF8E4M3,
//                            StdF8E4M3FN, StdF8E4M3FNUZ, StdF8E4M3B11FNUZ, StdF8E5M2,
//                            StdF8E5M2FNUZ, StdF8E8M0FNU]>;

// #endif
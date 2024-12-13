include "mlir/TableGen/Trait.h"

class StdType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<StdType_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

def StdArrayType : StdType<"StdArray", "array", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, ["getTypeSize"]>,
    DeclareTypeInterfaceMethods<DestructurableTypeInterface>]>  {
  let description = "An array type representing a sequence of elements.";
  let parameters = (ins "Type":$elementType, "uint64_t":$numElements);
  let assemblyFormat = [{
    `<` $numElements `x` $elementType `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "uint64_t":$numElements)>
  ];


  let extraClassDeclaration = [{
    /// Checks if the given type can be used inside an array type.
    static bool isValidElementType(Type type);ls /usr/local/bin
c
  }];
}

def StdPointerType : StdType<"StdPointer", "ptr", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "getIndexBitwidth", "areCompatible", "verifyEntries"]>]> {
  let summary = "Std pointer type";
  let description = [{
    The `!std.ptr` type is an Std pointer type. This type typically represents
    a reference to an object in memory. Pointers are optionally parameterized
    by the address space.
  }];

  let parameters = (ins DefaultValuedParameter<"unsigned", "0">:$addressSpace);
  let assemblyFormat = [{
    (`<` $addressSpace^ `>`)?
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins CArg<"unsigned", "0">:$addressSpace), [{
      return $_get($_ctxt, addressSpace);
    }]>
  ];
}
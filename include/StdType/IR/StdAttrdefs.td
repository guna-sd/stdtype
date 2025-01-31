#ifndef __STDATTRDEFS
#define __STDATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"

include "StdType/IR/StdTypeDialect.td"
include "StdType/IR/StdInterface.td"

class StdType_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<StdType_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "std." # attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Attribute Definitions
//===----------------------------------------------------------------------===//

def StdIntAttr : StdType_Attr<"Int", "int",[TypedAttrInterface]> {
  let summary = "An integer attribute.";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "Integer":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Integer &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

def StdUIntAttr : StdType_Attr<"UInt", "uint"> {
  let summary = "An unsigned integer attribute.";
  let description = [{
    An unsigned integer attribute is a literal attribute that represents an unsigned integral
    value of the specified unsigned integer type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "Integer":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Integer &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}


def StdFloatAttr : StdType_Attr<"Float", "float"> {
  let summary = "A floating-point attribute.";
  let description = [{
   A floating-point attribute is a literal attribute that represents a floating-point
   value of the specified floating-point type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "Float":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const Float &":$value), [{
      return $_get(type.getContext(), type, value);
      }]>
  ];
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

def StdNoneAttr : StdType_Attr<"None", "none"> {
  let summary = "A none attribute.";
  let description = [{
    A none attribute represents the absence of a value.
  }];
  let parameters = ();
  let builders = [
    AttrBuilderWithInferredContext<()>, [{
      return $_get(type.getContext(), type);
    }]>
  ];
  let assemblyFormat = "`none`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

def StdStringAttr : StdType_Attr<"String", "string"> {
  let summary = "A string attribute.";
  let description = [{
   A string attribute is a literal attribute that represents a string value.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "String":$value);
  let builders = [
    AttrBuilderWithInferredContext<("const String &":$value), [{
                                      return $_get(type.getContext(), type, value);
                                      }]>
  ];
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}
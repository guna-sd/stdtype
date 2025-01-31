#ifndef __STDINTERFACE
#define __STDINTERFACE


include "mlir/IR/OpBase.td"

def StdPointerElementTypeInterface 
    : TypeInterfaces<"PointerElementTypeInterface">{

    let cppNamespace = "::mlir::stdtype";

    let description = [{
    An interface for types that are allowed as elements of pointer type.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the size of the type in bytes.",
      /*retTy=*/"unsigned",
      /*methodName=*/"getSizeInBytes",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return dataLayout.getTypeSize($_type);
      }]
    >
  ];
}

#endif
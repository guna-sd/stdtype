#ifndef STDTYPE_DIALECT
#define STDTYPE_DIALECT

include "mlir/IR/DialectBase.td"

def StdType_Dialect : Dialect {
  let name = "stdtype";
  let cppNamespace = "::mlir::StdType";

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleType(Type);

    Type parseType(DialectAsmParser &p) const override;
    void printType(Type, DialectAsmPrinter &p) const override;
    
    void registerTypes();
  }];
}

#endif

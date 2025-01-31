include "mlir/IR/OpBase.td"

def StdType_Dialect : Dialect {
  let name = "stdtype";
  let cppNamespace = "::mlir::stdtype";
  let description = [{
    The stdtype Dialect is a unified solution for representing types within MLIR.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerLocationAttributes();
    void registerTypes();

  public:
    //
  }];
}

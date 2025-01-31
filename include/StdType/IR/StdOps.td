#ifndef __STDOPS
#define __STDOPS

include "mlir/IR/OpBase.td"
include "include/StdType/IR/Base.td"
include "include/StdType/IR/StdTypeDialect.td"


class StdOp<string mnemonic, list<Trait> traits = []> :
    Op<StdType_Dialect, mnemonic, traits> {
      
}

#endif
//===- StdTypeDialect.h - StdType Dialect -----------------------*- C++ -*-===//



//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_STDTYPE_IR_STDTYPEDIALECT_H
#define MLIR_DIALECT_STDTYPE_IR_STDTYPEDIALECT_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace stdtype {

class IndexType;
class IntegerType;
class FloatType;
class BoolType;
class StringType;
class VectorType;
class ArrayType;

}
}


#endif
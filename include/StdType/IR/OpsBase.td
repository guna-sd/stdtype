#ifndef __STDBASE
#define __STDBASE

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def StdBool : AnyTypeOf[StdI1, I1, StdBoolType];

def StdSignlessInt : AnyTypeOf[[SignlessIntOfWidths<[1, 2, 4, 6, 8, 16, 32, 64]>, StdI1, StdI2, StdI3, StdI4, StdI8, StdI16, StdI32, StdI64]];
def StdSInt : AnyTypeOf[[SignedIntOfWidths<[2, 4, 6, 8, 16, 32, 64]>, StdInt2, StdInt4, StdInt6, StdInt8, StdInt16, StdInt32, StdInt64]];
def StdUInt : AnyTypeOf[[UnsignedIntOfWidths<[2, 4, 6, 8, 16, 32, 64]>, StdUInt2, StdUInt4, StdUInt6, StdUInt8, StdUInt16, StdUInt32, StdUInt64]];
def StdInt : AnyTypeOf<[StdSignlessInt, StdSInt, StdUInt]>;

def StdFloat : AnyTypeOf<[F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
                           F8E4M3FN, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2,
                           F8E5M2FNUZ, F8E8M0FNU, F16, F32, F64, BF16, StdBF16, StdF16, StdF32, StdF64,
                           StdF4E2M1FN, StdF6E2M3FN, StdF6E3M2FN, StdF8E3M4, StdF8E4M3,
                           StdF8E4M3FN, StdF8E4M3FNUZ, StdF8E4M3B11FNUZ, StdF8E5M2,
                           StdF8E5M2FNUZ, StdF8E8M0FNU]>;

#endif
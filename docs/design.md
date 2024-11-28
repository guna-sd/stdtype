# stdtype Dialect Design Principles and Specs

## Overview

The **stdtype** dialect in MLIR is designed to fill a gap in the existing MLIR type system by introducing more complex and composite types. These types, such as arrays, pointers, structs, and collections, are crucial for various fields like systems programming, high-performance computing, and machine learning.

The goal of stdtype is to provide a unified solution for representing data structures within MLIR, while ensuring that operations on these types are efficient and well-integrated with MLIR's existing optimization and transformation passes.

## Key Design Principles

The primary goal of stdtype is to offer a **unified dialect** for all types, ranging from simple scalars (e.g., integers, floats, and booleans) to more complex data structures (e.g., structs, arrays, and collections). This design aims to simplify the representation and manipulation of diverse data types, reducing the need for multiple dialects and the complexity of having different representations for similar data.

## Type System Specification

### 1. **Scalar Types**

Scalar types are the most basic types, including:
- **Integer Types** (`iN`) signed, unsigned, signless integer types.
- **Floating-point Types** (`fN`) and ***IEEE*** floating-point types.
- **Boolean Type** (`bool`)
- **String Type** (`string`)


### 2. **Array and Vector Types**

Array and vector types are used to represent collections of elements of the same type, such as:
- **Fixed-size Arrays**: Arrays with a predefined size.
- **Dynamic Arrays**: Arrays whose size can change dynamically.
- **Vectors**: A special form of array optimized for SIMD operations.
- **Tensors**: Multi-dimensional arrays, critical for machine learning and scientific computing.

### 3. **Pointer and Reference Types**

Pointer and reference types allow direct manipulation of memory addresses and references to other objects. These are crucial for:
- **Low-level memory operations**
- **Efficient access patterns in systems programming**
- **Enabling operations on large datasets by referencing memory regions instead of copying data**

### 4. **Composite Types**

Composite types allow the combination of multiple data fields into a single unit. This includes:
- **Structs**: A collection of named fields with different types.
- **Unions**: A type that can hold one of several types at a time.
- **Enums**: A set of named values, often used for representing a set of discrete options.

### 5. **Collection Types**

Collection types represent higher-order data structures, such as:
- **Lists**: Ordered collections of elements.
- **Sets**: Unordered collections with no duplicate elements.
- **Maps**: Key-value pairs with unique keys.

## Operations

stdtype provides operations for manipulating the types, such as:
- **Arithmetic operations** on scalar types
- **Indexing and slicing** for arrays, vectors and tensors
- **Memory access** and dereferencing for pointers
- **Field access** for structs and unions
- **Collection Operations** insertions, deletions, and lookups for lists, sets, and maps.

### Operations on Tensors

For machine learning and scientific computing, stdtype supports tensor operations, including:
- **Matrix multiplication**
- **Element-wise operations** (e.g., addition, multiplication)
- **Reduction operations** (e.g., summing across dimensions)


## Future Extensions

### 1. **Sparse Tensors**

Support for sparse tensor formats, where data is stored efficiently by only keeping track of non-zero elements, is planned.

### 2. **Atomic Types**

Atomic types are critical for parallel computing, especially in concurrent algorithms. stdtype plans to introduce atomic types for synchronization and atomic operations on variables.

### 3. **Function Types**

The introduction of function types will allow the representation of callable objects, enabling higher-order functions and more flexible computations.
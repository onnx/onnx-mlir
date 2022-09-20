<!--- SPDX-License-Identifier: Apache-2.0 -->

# Handle ONNX Sequence Type

## ONNX Sequence Type
ONNX sequence type is a type for aggregation of values. It can be sequence of 
Tensor type, or sequence of Map type in ONNX. Currently onnx-mlir supports only sequence of tensor.
In ONNX dialect defined in onnx-mlir, the sequence type is defined as `SeqType`,
and shown as `!onnx.Seq<T>` in .mlir files. There are two access function defined for sequence type:
- Type elementType(). The type of the elements in the sequence. When the elements are 
  tensors with different shape, the type of elements has to be a super type of
  each elements. Shape inference will take care the type merging and refining. 
- int64_t length(). The number of elements in the sequence. -1 for statically unknown.

There are 4 basic sequence-related operations in ONNX:
- SequenceEmpty: create an empty sequence, namely memref<0xmemref<T>>
- SequenceInsert: add an element into the input sequence at specified position and return the result as a new sequence
- SequenceConstruct: construct a new sequence from the input elements.
- SequenceErase: remove an element at a specified position from the input sequence and return the result as a new sequence

## Lower ONNX Sequence Type to memref
Sequence type is an indexed container type to/from which element can be stored
or loaded at specified position, similar to std::vector<T>, or MemRefType in MLIR..
Due to the SSA semantics of ONNX operations, the container for sequence type has a fixed size. A sequence is created once and is not further modified.
In onnx-mlir, tensor is lowered to memref. We choose to lower ONNX SeqTye of tensor to
memref<?xmemref<*xT>>.
The outer memref in memref<?xmemref<*xT>> is a 1D memref for the 
sequence. The dim size of this memref is the length of the sequence.
The inner memref type is for the element type. It should the super type of all possible
element types, as discussed in the previous session.

The advantange is that we can make use of the memref dialect without introducing external
data structure. [reference other work]. The same optimization over MemRefType can be used
on tensor operations and sequence operations.
The store/load operation of a memref (for element) to/from a memref of memref (for the sequence) is directly supported by MLIR. The index for the sequence position will be the index
for the memref for sequence.
The following code is llvm code for memref
```
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.getelementptr %13[%12] : (!llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %11, %14 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
```
In the store, the descriptor for the element memref containing the dynamic shape
informantion and data pointer of the memref, is stored into the memref of memref.
Please note the content of the element memref is not stored.  
Correspondingly, the load will construct a memref from the memref of memref.
When an element loaded from sequence, the descriptor with the dynamic info is 
loaded from the memref of memref, while its static type is the element type
of the sequence.

The basic operations seem to work.  The sequence related operations in ONNX can be easily lowered using memref allocation, load or store.  However, there is an issue with buffer deallocation.

## Issues with deallocation
onnx-mlir relies on MLIR [Bufferization::Deallocation pass](https://mlir.llvm.org/docs/BufferDeallocationInternals/) to insert deallocation for memrefs. 
When a memref for the element is stored into a sequence, its data pointer along with shape
information is store and the stored memref is invisible in the operation graph.
This operation breaks the assumption of value based SSA assume for MLIR.
As a result, the deallocation pass
will add a deallocation for the element memref after its last visible use. 
Consequently, when this element is loaded from the sequence, the memref will 
have a dangling pointer to its data.
The source of this issue is that the data pointer for element is saved in the 
sequence. This operation breaks the basic assume of operations on "values".
Another issue is with the deallocation of memref<memref> for sequence. 
If the memref for the element were not freed by deallocation pass, there would be issue
on deallocation of the memref for sequence: deep deallocation for the elements in the
sequence is needed.

## Solution
We could extend the deallocation pass to handle the load/store of memref<memref<T>>.
When a source memref is stored into a destination memref, the source memref 
could be marked as `escaped` and then no deallocation would be added by the 
deallocation pass. Such change will involve how to add clone op with the present of 
control flow. Our current solution is based the existing deallocation pass.

### Store an element into a sequence
To avoid the deallocation of the element, we can save a copy of the element into the
sequence. To get a copy, we need to allocate a memref and then use memref.copy
to copy the value. However, deallocation pass will again add deallocation to
the newly allocated memref. To avoid this issue, we should wrap all the operations of
memref the allocation, copy and store into one krnl Op, krnlSeqStoreOp, which will be
lowered AFTER the deallocation pass.

Since the type of sequence element is a super type for all possible element, memref.cast may be
needed before the store. Here is a code example for KrnlSeqStoreOp:
```
// %seq : memref<?xmemref<?x2xf32>>
// %element : memref<3x2xf32>
// %pos : Index
      %33 = memref.alloc(%32) {alignment = 16 : i64} : memref<3x2xf32>
      memref.copy %element, %33 : memref<3x2xf32> to memref<3x2xf32>
      %34 = memref.cast %33 : memref<3x2xf32> to memref<?x2xf32>
      memref.store %34, %seq[%pos] : memref<?xmemref<?x2xf32>>
```

### Allocate a sequence
Though the basic operation of sequence allocation is just memref.alloc, we introduced
KrnlSeqAllocOp so that we can define a customized deallocation function for sequence.
We use interface in MLIR Bufferization to specify that KrnlSeqAllocOp has allocation 
traits and a customized free function, which will perform a deep deallocation for the
elements as well as the sequence itself. Currently, the KrnlSeqDeallocOp is used for the
deallocation and it will be lowered to scf and memref after deallocation pass.

### Load an element from a sequence
A memref.load could be used to load an element from a sequence. But the loaded memref may
have a life span longer than the sequence itself, and will have dangling data pointer after
the sequence has been freed.

To overcome this issue, KrnlSeqExtractOp is introduced. This Op will use memref.load to
load the element, then use allocate a new memref and copy the data, and finally return
the copied memref.  This Op is marked with allocation interface and the deallocation pass will insert deallocation for the returned memref automatically. 

### Construct a new sequence from an old sequence
The sequence ops, SequenceInsert and SequenceErase, construct a new sequence
with the elements from an old sequence. It is fine to use KrnlSeqExtractOp to get an
element from the old sequence and use KrnlSeqStoreOp to store it into the new sequence.
But there are two copy operations for the element. Since it is known that the loaded
element is only used here while the old sequence is still live, a regular memref.load 
can be used to avoid one copy. This is a simple optimization for sequence lowering.

## Example

Original .mlir code:

```
func.func @test_sequence_insert(%arg0: !onnx.Seq<tensor<?x4x5xf32>>, %arg1:tensor<3x4x5xf32>) -> tensor<3xi64>  {
  %0 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<i64>
  %1 = "onnx.Add"(%arg1, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %6 = "onnx.SequenceInsert"(%arg0, %arg1, %2) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<3x4x5xf32>, none) -> !onnx.Seq<tensor<?x4x5xf32>>
  %4 = "onnx.SequenceAt"(%6, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> tensor<?x4x5xf32>
  %5 = "onnx.Shape"(%4) : (tensor<?x4x5xf32>) -> tensor<3xi64>
  return %5 : tensor<3xi64>
}
```

After --convert-onnx-to-krnl pass

```
  func.func @test_sequence_insert(%arg0: memref<?xmemref<?x4x5xf32>>, %arg1: memref<3x4x5xf32>) -> memref<3xi64> {

    // onnx.Add
    %1 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
    %2:3 = krnl.define_loops 3
    krnl.iterate(%2#0, %2#1, %2#2) with (%2#0 -> %arg2 = 0 to 3, %2#1 -> %arg3 = 0 to 4, %2#2 -> %arg4 = 0 to 5){
      %22:3 = krnl.get_induction_var_value(%2#0, %2#1, %2#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      ...
      %23 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %24 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %25 = arith.addf %23, %24 : f32
      krnl.store %25, %1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
    }

    // Sequence Insert
    %6 = "krnl.seqalloc"(%5) : (index) -> memref<?xmemref<?x4x5xf32>>
    %c0_8 = arith.constant 0 : index
    %7 = krnl.define_loops 1
    krnl.iterate(%7) with (%7 -> %arg2 = 0 to %4){
      %22 = krnl.get_induction_var_value(%7) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      "krnl.seqstore"(%23, %6, %4) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    %c1_9 = arith.constant 1 : index
    %8 = affine.apply #map0()[%4]
    %9 = krnl.define_loops 1
    krnl.iterate(%9) with (%9 -> %arg2 = #map0()[%4] to %4){
      %22 = krnl.get_induction_var_value(%9) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      %c1_18 = arith.constant 1 : index
      %24 = arith.addi %22, %c1_18 : index
      "krnl.seqstore"(%23, %6, %24) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    "krnl.seqstore"(%1, %6, %4) : (memref<3x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()

    // SequenceAt
    ...
    %16 = "krnl.seqextract"(%6, %15) {copy = 1 : ui1} : (memref<?xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>

    // onnx.Shape
    %17 = memref.alloc() {alignment = 16 : i64} : memref<3xi64>
    ...
    krnl.store %19, %17[%c0_16] : memref<3xi64>
    krnl.store %20, %17[%c1_17] : memref<3xi64>
    krnl.store %21, %17[%c2] : memref<3xi64>

    return %17 : memref<3xi64>
  }
```

After --buffer-deallocation pass
```
  func.func @test_sequence_insert(%arg0: memref<?xmemref<?x4x5xf32>>, %arg1: memref<3x4x5xf32>) -> memref<3xi64> {

    // onnx.Add
    %1 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
    %2:3 = krnl.define_loops 3
    krnl.iterate(%2#0, %2#1, %2#2) with (%2#0 -> %arg2 = 0 to 3, %2#1 -> %arg3 = 0 to 4, %2#2 -> %arg4 = 0 to 5){
      %22:3 = krnl.get_induction_var_value(%2#0, %2#1, %2#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      ...
      %23 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %24 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %25 = arith.addf %23, %24 : f32
      krnl.store %25, %1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
    }

    // Sequence Insert
    %6 = "krnl.seqalloc"(%5) : (index) -> memref<?xmemref<?x4x5xf32>>
    %c0_8 = arith.constant 0 : index
    %7 = krnl.define_loops 1
    krnl.iterate(%7) with (%7 -> %arg2 = 0 to %4){
      %22 = krnl.get_induction_var_value(%7) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      "krnl.seqstore"(%23, %6, %4) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    %c1_9 = arith.constant 1 : index
    %8 = affine.apply #map0()[%4]
    %9 = krnl.define_loops 1
    krnl.iterate(%9) with (%9 -> %arg2 = #map0()[%4] to %4){
      %22 = krnl.get_induction_var_value(%9) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      %c1_18 = arith.constant 1 : index
      %24 = arith.addi %22, %c1_18 : index
      "krnl.seqstore"(%23, %6, %24) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    "krnl.seqstore"(%1, %6, %4) : (memref<3x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()

    // Dealloc
    memref.dealloc %1 : memref<3x4x5xf32>

    // SequenceAt
    ...
    %16 = "krnl.seqextract"(%6, %15) {copy = 1 : ui1} : (memref<?xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>
    "krnl.seqdealloc"(%6) : (memref<?xmemref<?x4x5xf32>>) -> ()

    // onnx.Shape
      ...
    memref.dealloc %16 : memref<?x4x5xf32>
}
```

After --convert-seq-to-memref pass:
```
    // KrnlSeqStore
    %10 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
    memref.copy %1, %10 : memref<3x4x5xf32> to memref<3x4x5xf32>
    %11 = memref.cast %10 : memref<3x4x5xf32> to memref<?x4x5xf32>
    memref.store %11, %6[%4] : memref<?xmemref<?x4x5xf32>>

    // KrnlSeqExtract
    %18 = memref.load %6[%17] : memref<?xmemref<?x4x5xf32>>
    %c0_12 = arith.constant 0 : index
    %19 = memref.dim %18, %c0_12 : memref<?x4x5xf32>
    %20 = memref.alloc(%19) {alignment = 16 : i64} : memref<?x4x5xf32>
    memref.copy %18, %20 : memref<?x4x5xf32> to memref<?x4x5xf32>
 ...
    // KrnlSeqDealloc
    scf.for %arg2 = %c0_14 to %5 step %c1_15 {
      %26 = memref.load %6[%arg2] : memref<?xmemref<?x4x5xf32>>
      memref.dealloc %26 : memref<?x4x5xf32>
    }
    memref.dealloc %6 : memref<?xmemref<?x4x5xf32>>
}
```

## Discussion
### Correctness
There is no chance of dangling pointer for pointers used for sequence implementation. 
Memref is copied both when its pointer is saved into the sequence and when its pointer is
read out from the sequence. That means the pointer saved in the sequence is kind of 
"unique_pointer". These pointers will be freed only by the deallocation of the sequence.

There is no memory leak, either. The elements are freed by normal memref.dealloc and the
sequence is freed by deep deallocation for the sequence.

### Optimization
Since the tensor/memref for element is read only, there is no need to copy it in logic.
the copy is added due to two reasons:
-1. No interface to communicate with existing deallocation pass.
-2. For pointers possibly in multiple sequence.

If some analysis can guarantee the element is only in one live sequence (which is usually
the case in program), we can rewrite the ONNX sequence Ops in a different way:
- No deep deallocation for some sequences
- No copy at all when an element is moved from a sequence to another sequence
Regular memref ops, instead of KrnlSeqAllocOp or KrnlSeqStoreOp, will be used to rewrite
ONNX Sequence Ops.

Another direction of optimization is to use std::shared_ptr for memref or ohe-shot 
interface to optimize.

## ToFix
- Shape inference with control flow. The result for test case with LoopOp(test_loop13_seq.onnx) is not correct.
- Refine the output of SequenceEmpty. The ONNX op to create an empty sequence,
  SequenceEmpty, is specified to generate a sequence of unranked tensor, which is not 
  supported in onnx-mlir.
- Handle program argument or return with SeqType.

## Side question
- Impact of SSA like requirement by deallocation pass on optimization. The deallocation pass requires that `all buffer writes need to dominate all buffer reads`. Will loop fusion across ONNX Ops break this requirement?



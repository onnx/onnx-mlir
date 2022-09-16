<!--- SPDX-License-Identifier: Apache-2.0 -->

#Handle ONNX Sequence Type

## ONNX Sequence Type
ONNX sequence type is a type for aggregation of values. It can be sequence of 
tensor, or sequence of Map. Currently only sequence of tensor is supported.
In ONNX dialect defined in onnx-mlir, the sequence type is defined as SeqType,
and shown as !onnx.Seq<T> in .mlir files. There are two members for sequence type:
- Type elementType: the type of the elements in the sequence. When the elements are 
  tensors with different shape, the type of elements has to be a super type of
  the each elements. Shape inference will take care the type merging and refining. 
- int64_t length: the number of elements in the sequence. -1 for statically unknown.

## Lower ONNX Sequence Type to memref
Sequence type is an indexed container type to/from which element can be stored
or loaded at specified position, similar to std::vector<T>.
Due to the SSA semantics of ONNX operations, the container
for sequence type have a fixed size. In onnx-mlir, memref<?xmemref<*xT>>
is chosen to implement ONNX sequence type without introducing other special data
structure.[reference other work]. memref of memref is directly supported by 
MLIR. Sequence of tensor is lowered to memref of memref along with tensor is lowered memref.
The outer memref in memref<?xmemref<*xT>> is a one dimension memref for the 
sequence. The dim size of this memref matches the length of the sequence.
The inner
memref type is for the element type. It should the super type of all possible
element types, as discussed in the previous session.
 A store of a memref (for element), into a memref (for sequence),  will put the descriptor of 
the element memref into the sequence memref at the specified position. 
The descriptor contains the dynamic shape informantion and data pointer.
 Please note the content of the element memref is not stored. Since the type of
sequence element is a super type for all possible element, memref.cast may be
needed before the store. Here is a code example:
```
// %seq : memref<?xmemref<?x2xF32>>
// %element : memref<3x2xF32>
// %pos : Index
 %converted = 'memref.cast'(%element, memref<?x2xF32>)
 memref.store(%seq, %element, %pos)
```
need to verify the example
add code after lowered to llvm to understand the store.

When an element loaded from sequence, the descriptor with the dynamic info is 
loaded from the memref of memref, while its static type is the element type
of the sequence.
If an element is loaded from the sequence of unranked tensor, the resulted
tensor will be an unranked tensor, which can not be handled by onnx-mlir.
We have to avoid using sequence of unranked tensor.

## Naive way to lower sequence ops
There are 4 sequence-related operations in ONNX:
- SequenceEmpty: create an empty sequence, namely memref<0xmemref<T>>
- SequenceInsert: add an element into the input sequence at specified position and return the result as a new sequence
- SequenceConstruct: construct a new sequence from the input elements.
- SequenceErase: remove an element at a specified position from the input sequence and return the result as a new sequence

These ONNX operations can be easily lowered using memref allocation, load or store.
However, there is an issue with buffer deallocation.

## Issues with deallocation
onnx-mlir relies on MLIR [Bufferization::Deallocation pass](https://mlir.llvm.org/docs/BufferDeallocationInternals/) to insert deallocation for memrefs. 
When a memref is stored into a sequence, only its description is stored and 
it becomes invisible in the operation graph. The deallocation pass
will add a deallocation for this memref after its last visible use. 
Consequently, when the element is loaded from the sequence, the memref will 
have a dangling pointer to its data.
The source of this issue is that the data pointer for element is saved for 
sequence. This operation breaks the basic assume of operations on "values".

## Solution from deallocation pass
We can extend the deallocation pass to handle the load/store of memref<memref<T>>.
When a source memref is stored into a destination memref, the source memref 
should be marked as `escaped` and no deallocation should be added to it.
For deallocation, we can use the buildDealloc interface. We introduce a 
krnlSeqAlloc for sequence allocation. The op is marked as allocation and has
deallocation defined. The deallocation will deallocate all the elements in 
the sequence and then deallocate the sequence itself.

## Hacking solution
To avoid the deallocation problem without changing the pass, the value,
not just pointer, of the memref should be 
saved into the sequence, according to the basic semantics of ONNX ops.
The easy way to save the value is to allocate a memref and then use memref.copy
to copy the value. However, deallocation pass will again add deallocation to
the newly allocated memref. To avoid this issue, we can wrap the allocation, 
copy and store into a krnlSeqStore Op, which will be lowered after the
deallocation pass.
For deallocation the elements in the sequence, KrnlSeqFree is introduced and
inserted just before the deallocation of the sequence.
### Optimization
If the tensor is always copied and then inserted into a sequence, the program 
will be correct but costly. There are two cases that an element is inserted
into a sequence:
- case#1: The element is a result of some operation.
- case#2: The element copied from another sequence when a new sequence is constructed in an ONNX sequence operation, such as SequenceInsert, or SequenceErase.
In case#2, it seems that the descriptor from another sequence can be directly
stored into the newly created sequence without copying the content. This is true
for most of applications. But we may have problem in deallocation: the elements
may be deallocated multple times. Pointers are stored in sequences. We either
guarantee that every pointer is a unique pointer or we need to use smart pointer
to maintain the reference count for deallocation.

If a program read out
all the elements in the sequence before deallocating the sequence, we can
use KrnlSeqExtract with toCopy=0. This op is marked with `allocate` interface,
and the deallocation pass will create dealloc operation for the extracted memref. This is  the current solution and will be obsolete.


## Questions
- Impact of SSA like requirement by deallocation pass on optimization. The deallocation pass requires that `all buffer writes needs to dominate all buffer reads`. Will loop fusion across ONNX Ops break this requirement?
- Just wrap the allocate op?



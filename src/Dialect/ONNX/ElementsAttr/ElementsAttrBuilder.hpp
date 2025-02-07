/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// ElementsAttrBuilder builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ELEM_ATTR_BUILDER_H
#define ONNX_MLIR_ELEM_ATTR_BUILDER_H
#include "mlir/IR/Threading.h"

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"

#include <functional>
#include <memory>

namespace onnx_mlir {

// Builds DisposableElementsAttr instances.
// Every instance is inserted into DisposablePool which garbage collects
// unreachable instances between compiler passes.
class ElementsAttrBuilder {
public:
  // Uses disposablePool to construct DisposableElementsAttr instances
  // in the builder methods.
  ElementsAttrBuilder(DisposablePool &disposablePool);

  // Creates a DisposableElementsAttr instance backed by the data in membuf.
  // The created instance takes ownership of membuf and will release it when the
  // instance is disposed by garbage collection, unless it has shared membuf
  // with other DisposableElementsAttr instances that are longer lived.
  mlir::ElementsAttr fromMemoryBuffer(
      mlir::ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf);

  // Wraps elements in a DisposableElementsAttr if it isn't already a
  // DisposableElementsAttr, provided the underlying DisposablePool is active.
  // If elements is DenseElementsAttr the wrapper points into elements' raw
  // data, except if the element type is bool, then a deep copy is made that
  // unpacks the bits because DisposableElementsAttr doesn't bit pack bools.
  mlir::DisposableElementsAttr toDisposableElementsAttr(
      mlir::ElementsAttr elements);

  // Converts elements to DenseElementsAttr if it's DisposableElementsAttr,
  // otherwise returns elements if it's already DenseElementsAttr.
  static mlir::DenseElementsAttr toDenseElementsAttr(
      mlir::ElementsAttr elements);

  // Compares contents for equality. Argument shapes must be broadcast
  // compatible. Element types must the same.
  // Asserts if these preconditions are violated (doesn't return false as that
  // would hide whether the lhs and rhs are different or incompatible).
  //
  // TODO: Move this function to a better place, since it doesn't build
  //       anything, but it's here for now for efficient elements access.
  static bool equal(mlir::ElementsAttr lhs, mlir::ElementsAttr rhs);

  // More efficient way to test if lhs is equal to a single (broadcasted)
  // value broadcastedRhsValue. Equivalent to equal(lhs, splatRhs) where
  // splatRhs is a splat ElementsAttr with value broadcastedRhsValue.
  static bool allEqual(mlir::ElementsAttr lhs, WideNum broadcastedRhsValue);

  template <typename T>
  using Filler = std::function<void(llvm::MutableArrayRef<T>)>;

  // Constructs a DisposableElementsAttr and calls wideDataFiller to populate
  // its memory buffer.
  mlir::ElementsAttr fromWideNums(
      mlir::ShapedType type, const Filler<WideNum> &wideDataFiller);

  // Constructs a DisposableElementsAttr and calls arrayFiller to populate
  // its memory buffer.
  template <typename T>
  mlir::ElementsAttr fromArray(
      mlir::ShapedType type, const Filler<T> &arrayFiller) {
    return fromRawBytes(
        type, toBType<T>, [&arrayFiller](llvm::MutableArrayRef<char> bytes) {
          arrayFiller(castMutableArrayRef<T>(bytes));
        });
  }

  // Returns an ElementsAttr where each element is transformed
  // by running the given transformer on all the elements.
  //
  // Reuses elms' underlying data without a data copy.
  template <typename Function = WideNum (*)(WideNum)>
  mlir::ElementsAttr transform(mlir::ElementsAttr elms,
      mlir::Type transformedElementType, Function fun) {
    return doTransform(
        elms, transformedElementType, functionTransformer(std::move(fun)));
  }

  // Returns an ElementsAttr that is the result of applying a binary function
  // pairwise on the elements lhs and rhs after broadcast to combinedType.
  // Note that combiner is a plain function pointer to make it cheap and easy
  // to copy it into a transformer lambda in the returned ElementsAttr.
  //
  // Constructs new underlying data by applying the combiner, except in the
  // case where one of the arguments is splat, in that case reuses the other
  // argument's underlying data and just adds the necessary transformation
  // and broadcast.
  mlir::ElementsAttr combine(mlir::ElementsAttr lhs, mlir::ElementsAttr rhs,
      mlir::ShapedType combinedType, WideNum (*combiner)(WideNum, WideNum));

  // Returns an ElementsAttr that is the result of applying a the function
  // (cond ? lhs : rhs) element-wise after broadcast to combinedType.
  //
  // Constructs new underlying data except in the cases where either cond is
  // splat or lhs and rhs are both splat. In those case reuses the underlying
  // data of one of the elements and just adds the necessary transformation
  // and broadcast.
  mlir::ElementsAttr where(mlir::ElementsAttr cond, mlir::ElementsAttr lhs,
      mlir::ElementsAttr rhs, mlir::ShapedType combinedType);

  // Returns an ElementsAttr with the elements cast to the given newElementType
  // with default choices for rounding (true) and saturation (false).
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr castElementType(
      mlir::ElementsAttr elms, mlir::Type newElementType);

  // Returns an ElementsAttr with the elements cast to the given intElementType.
  //
  // If round==true and elms has floating point numbers type then they are
  // rounded to nearest integer, ties to even, otherwise they are truncated
  // towards zero.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr castToIntElementType(mlir::ElementsAttr elms,
      mlir::IntegerType newElementType, bool round = true);

  // Returns an ElementsAttr with the elements cast to the given fpElementType.
  //
  // If saturate==true and newElementType has +/-infinity then out of range
  // numbers are cast to +/-infinity, otherwise they are clipped to the finite
  // range.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr castToFPElementType(mlir::ElementsAttr elms,
      mlir::FloatType newElementType, bool saturate = false);

  // Returns an ElementsAttr with the values clipped to the range [min, max].
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr clip(mlir::ElementsAttr elms, WideNum min, WideNum max);

  // Returns a transposed ElementsAttr.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr transpose(
      mlir::ElementsAttr elms, llvm::ArrayRef<uint64_t> perm);

  // Returns a reshaped ElementsAttr.
  //
  // Reuses elms' underlying data without a data copy, unless the underlying
  // data is transposed in a way that requires the data to be "restrided".
  mlir::ElementsAttr reshape(
      mlir::ElementsAttr elms, llvm::ArrayRef<int64_t> reshapedShape);

  // Broadcasts like the ONNX Expand op.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr expand(
      mlir::ElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape);

  // Splits the tensor in elms along axis into sizes.size() tensors where
  // tensor[i].shape[axis] == sizes[i], and they all sum to elms.shape[axis].
  //
  // The returned tensors don't reuse elms' underlying data, unless sizes.size()
  // is 1 and elms is returned.
  std::vector<mlir::ElementsAttr> split(
      mlir::ElementsAttr elms, unsigned axis, llvm::ArrayRef<int64_t> sizes);

  // Concatenates the tensors along axis.
  mlir::ElementsAttr concat(
      llvm::ArrayRef<mlir::ElementsAttr> elms, unsigned axis);

  // Slices the tensor.
  // shape, start, steps lengths must equal the tensor rank.
  // shape and start must be non-negative.
  // Negative steps means slicing backwards.
  mlir::ElementsAttr slice(mlir::ElementsAttr elms,
      llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> starts,
      llvm::ArrayRef<int64_t> steps);

  // Pads the tensor.
  // 'pads' length must equal two times the tensor rank and all
  // entries must be non-negative.
  mlir::ElementsAttr pad(
      mlir::ElementsAttr elms, llvm::ArrayRef<int64_t> pads, WideNum padValue);

  // Gathers a tensor of the values from an input tensor given by a tensor of
  // indices, along the specified axis.
  // Follows the specification of the onnx Gather operation.
  mlir::ElementsAttr gather(
      mlir::ElementsAttr input, mlir::ElementsAttr indices, unsigned axis);

  // Returns copy of input with updates from a tensor of update values at the
  // index positions given by a tensor of indices.
  // Follows the specification of the onnx ScatterND operation.
  mlir::ElementsAttr scatterND(mlir::ElementsAttr input,
      mlir::ElementsAttr indices, mlir::ElementsAttr updates);

  // Assumptions: elms is non-empty, reducer is associative and commutative.
  mlir::ElementsAttr reduce(mlir::ElementsAttr elms,
      llvm::ArrayRef<unsigned> axes, bool keepdims,
      WideNum (*reducer)(WideNum, WideNum));

  // Returns the matrix product like numpy.matmul.
  mlir::ElementsAttr matMul(mlir::ElementsAttr lhs, mlir::ElementsAttr rhs);

  // Returns tensor of given type and shape with [start + i * delta ...]
  // for i in [0, type.getNumElements()) in row-major order.
  mlir::ElementsAttr range(mlir::ShapedType, WideNum start, WideNum delta);

  // Returns indices of non-zero elements like numpy.nonzero,
  // but for scalar input produces output shape [0, N] instead of [1, N],
  // which is different from Numpy's behavior.
  mlir::ElementsAttr nonZero(mlir::ElementsAttr elms);

private:
  struct ElementsProperties;

  static ElementsProperties getElementsProperties(mlir::ElementsAttr elements);

  static ArrayBuffer<WideNum> getWideNumsAndStrides(
      mlir::ElementsAttr elms, llvm::SmallVectorImpl<int64_t> &strides) {
    return getWideNumsAndExpandedStrides(
        elms, elms.getShapedType().getShape(), strides);
  }

  static ArrayBuffer<WideNum> getWideNumsAndExpandedStrides(
      mlir::ElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape,
      llvm::SmallVectorImpl<int64_t> &expandedStrides);

  // A transformer mutates elements.
  using Transformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;

  // Constructs a transformer that changes every element to the result of
  // applying the given function to the element.
  template <typename Function = WideNum (*)(WideNum)>
  inline Transformer functionTransformer(Function fun) {
    mlir::MLIRContext *ctx = disposablePool.getContext();
    return [fun = std::move(fun), ctx](
               llvm::MutableArrayRef<WideNum> data) -> void {
      auto fetchBatch = [&](size_t threadNumber, bool parallel) {
        // retrun all data without spliting for sequential execution.
        if (!parallel)
          return llvm::make_range(data.begin(), data.end());
        // Each thread fetches the same data size. The leftovers are set in the
        // threads with small thread number.
        size_t tileSize = floor(data.size() / ctx->getNumThreads());
        size_t leftovers = data.size() % ctx->getNumThreads();
        int beginOffset;
        if (threadNumber < leftovers) {
          // for the first few threads, it is as if the block size is larger
          // by 1.
          tileSize++;
          beginOffset = threadNumber * tileSize;
        } else {
          // for the last threads, its as we shift the start by leftovers.
          beginOffset = threadNumber * tileSize + leftovers;
        }
        int endOffset = beginOffset + tileSize;
        return llvm::make_range(
            data.begin() + beginOffset, data.begin() + endOffset);
      };

      auto work = [&](size_t threadNumber, bool parallel = true) {
        auto tile = fetchBatch(threadNumber, parallel);
        for (WideNum &n : tile)
          n = fun(n);
      };
      // Using 'parallelFor()' introduces large overhead.
      // To avoid this overhead, call work() directry if input size is less than
      // `minCount`.
      constexpr size_t minCount = 1000;
      if (data.size() < minCount)
        work(0, /*parallel*/ false);
      else
        parallelFor(ctx, 0, ctx->getNumThreads(), work);
    };
  }

  mlir::ElementsAttr doTransform(mlir::ElementsAttr elms,
      mlir::Type transformedElementType, Transformer transformer);

  mlir::ElementsAttr expandAndTransform(mlir::ElementsAttr elms,
      mlir::ShapedType expandedTransformedType, Transformer transformer);

  mlir::ElementsAttr fromRawBytes(mlir::ShapedType type, BType bufferBType,
      const Filler<char> &bytesFiller);

  mlir::ElementsAttr createWithDefaultStrides(mlir::ShapedType type,
      BType bufferBType, std::unique_ptr<llvm::MemoryBuffer> membuf);

  // Create a DisposableElementsAttr and put it in disposablePool.
  mlir::ElementsAttr create(mlir::ShapedType type, BType bufferBType,
      llvm::ArrayRef<int64_t> strides,
      const std::shared_ptr<llvm::MemoryBuffer> &buffer,
      Transformer transformer = nullptr);

  DisposablePool &disposablePool;
};

} // namespace onnx_mlir
#endif

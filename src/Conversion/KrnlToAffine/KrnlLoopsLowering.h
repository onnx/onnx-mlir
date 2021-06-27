#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"

namespace onnx_mlir {

// Since Krnl Dialect allows optimizations to be specified in the form of
// recipes without being applied, some IR block may exist under Krnl loops
// corresponding to loops that will be materialized only after relevant
// optimization recipes are applied; these Krnl loops serve as anchors for IR
// placement as we progressively apply optimization recipes, creating new
// concrete loops that will correspond to these optimized loop references.
// Whenever a concrete loop gets materialized and is referred to by Krnl loop
// reference %loop_ref, we will need to maintain the relative positioning of IR
// block and their parent loop operations; we do so by moving IR blocks while
// Krnl Dialect lowering proceeds.
//
// Consider the following example, where we specify the recipe for a
// 2-dimensional tiled loop, and insert memory allocation/deallocation aimed to
// set up and clean up per-tile temporary buffer:
//
// %ii, %ij = krnl.define_loops 2
// %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop,
//     !krnl.loop, !krnl.loop
// krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
//   %alloc = alloc() : memref<10 x f32>
//   krnl.iterate(%il, %jl) with () {
//     %foo = addi %i, %j : index
//   }
//   dealloc %alloc : memref<10 x f32>
//  }
//
// The temporary buffer allocation/deallocation are placed within loops that
// have yet to be materialized because loop tiling and loop permutation are only
// specified as recipes without actually being applied at Krnl Dialect level.
// Therefore as we proceed to lower Krnl Dialect, there will be no place for
// these (blocks of) operations to exist until the corresponding concrete outer
// loops emerge, as a result of optimizations being applied. Upon materializing
// such a loop, we will move these (blocks of) operations to the corresponding
// regions in the newly created loops.
//
// We use LoopBody mover to:
// - register, for each Krnl loop reference, blocks of operations
//   that should be contained directly beneath the corresponding concrete loops
//   as the moving plan in the beginning of the Krnl Dialect lowering.
// - subsequently, when the concrete loops corresponding to the Krnl loop
//   reference is materialized, IR blocks will be moved to appropriate locations
//   based on information recorded as moving plan.
//
// Thus, for the above IR, the following moving plan will be registered:
// - For %ib, %jb, the list of operation nested directly under is:
//    - alloc() operation,
//    - materialized loops corresponding to %il, %jl,
//    - dealloc() operation.
// - For %il, %jl, the list of operations nested directly under is:
//    - addi operation.
//
// Subsequently, lowering will start with affine ops materialized corresponding
// to the reference to un-optimized loops:
//
// affine.for %i = 0 to 10 {
//   affine.for %j = 0 to 20 {
//     %foo = addi %i, %j : index
//   }
// }
//
// Since the tiling has not taken place yet, tile coordinate iteration loops
// have not been materialized, therefore the alloc and dealloc operations do not
// fit in the IR presently yet. Instead, they will be placed within a
// krnl.movable op region, to indicate that their positioning is subject to
// change.
//
// krnl.movable {
//   %alloc = alloc() : memref<10 x f32>;
// }
// krnl.movable {
//   dealloc %alloc : memref<10 x f32>
// }
//
// As we lower the optimization recipes, outer loops will eventually manifest as
// affine loops. When the destination loops emerge, content within the
// krnl.movable op will be transferred to appropriate locations, too, resulting
// in the following final lowered IR:
//
// affine.for ib = 0 to 10 step 5 {
//   affine.for jb = 0 to 20 step 4 {
//     %alloc = alloc() : memref<10xf32>
//     affine.for %il = ... {
//       affine.for %jl = ... {
//         %foo = addi %il, %jl : index
//       }
//     }
//     dealloc %alloc : memref<10xf32>
//   }
// }
//
// As specified by the high-level Krnl Dialect.
class LoopBodyMover {
public:
  /*!
   * Represents either:
   * - a list of operations to be moved, or
   * - a particular set of loop nests expected in the destination loop body.
   *     This is helpful because we're only adjusting the relative positioning
   *     of IR blocks with respect to the concrete loops as we lowering the Krnl
   *     Dialect by applying the optimization recipes. Therefore, clearly
   *     moving IR blocks alone is sufficient to achieve our goal, and recording
   *     the position of expected loop nests in the destination loop body simply
   *     helps determine the correct relative position of IR blocks with respect
   *     to inner loops.
   */
  struct Movable {
    llvm::Optional<mlir::KrnlMovableOp> movableOp;
    llvm::Optional<llvm::SmallVector<mlir::Value, 4>> loopsToSkip;

    explicit Movable(mlir::KrnlMovableOp op) : movableOp(op) {}
    explicit Movable(mlir::KrnlIterateOp op) {
      auto operandRange = op->getOperands();
      loopsToSkip = llvm::SmallVector<mlir::Value, 4>(operandRange.begin(),
          operandRange.begin() + op.getNumOptimizedLoops());
    }
  };

  /*!
   * Register in our moving plan that content in the movable op should be moved
   * under the concrete loops corresponding to loop.
   * @param movable IR blocks enclosed in krnl.movable op to move around.
   * @param loop The Krnl Loop referring to the concrete loop sourrounding the
   * content of the movable op in the lowered IR.
   */
  void toMoveUnder(const Movable &movable, mlir::KrnlIterateOp loop);

  /*!
   * Signal that the concrete loop corresponding to loopRef has been
   * materialized, and therefore we can transfer operations to its loop body as
   * specified by moving plan.
   * @param loopRef Krnl loop ref corresponding to the concrete loop being
   * materialized.
   * @param loopRefToOp A dictionary keeping track of the correspondence between
   * Krnl loop references and concrete loops.
   * @param erase whether to erase entries in the moving plan corresponding to
   * this action.
   */
  void moveOne(mlir::Value loopRef,
      llvm::SmallDenseMap<mlir::Value, mlir::AffineForOp, 4> &loopRefToOp,
      bool erase = true);

  void moveAll(
      llvm::SmallDenseMap<mlir::Value, mlir::AffineForOp, 4> &loopRefToOp);

private:
  llvm::DenseMap<mlir::Value, llvm::SmallVector<Movable, 4>> movingPlan;
};

LoopBodyMover preprocessKrnlLoops(
    mlir::FuncOp funcOp, mlir::OpBuilder &builder, bool debug = true);

} // namespace onnx_mlir
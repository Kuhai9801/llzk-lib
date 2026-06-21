//===- SparseAnalysis.cpp - LLZK sparse data-flow adapter -----------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SparseAnalysis.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

namespace llzk::dataflow {

AbstractSparseForwardDataFlowAnalysis::AbstractSparseForwardDataFlowAnalysis(
    mlir::DataFlowSolver &s
)
    : mlir::dataflow::AbstractSparseForwardDataFlowAnalysis(s) {}

mlir::LogicalResult AbstractSparseForwardDataFlowAnalysis::initialize(mlir::Operation *top) {
  // Match upstream sparse initialization of top-level region entry arguments.
  for (mlir::Region &region : top->getRegions()) {
    if (region.empty()) {
      continue;
    }
    for (mlir::Value argument : region.front().getArguments()) {
      setToEntryState(getLatticeElement(argument));
    }
  }

  return initializeRecursivelyInProgramOrder(top);
}

mlir::LogicalResult AbstractSparseForwardDataFlowAnalysis::visit(mlir::ProgramPoint *point) {
  if (point->isBlockStart()) {
    return mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(point);
  }

  mlir::Operation *op = point->getPrevOp();
  if (op->getNumResults() != 0) {
    return mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(point);
  }
  return visitZeroResultOperation(op);
}

mlir::LogicalResult
AbstractSparseForwardDataFlowAnalysis::initializeRecursivelyInProgramOrder(mlir::Operation *op) {
  if (mlir::failed(visitOperationDuringInitialization(op))) {
    return mlir::failure();
  }

  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      mlir::ProgramPoint *blockStart = getProgramPointBefore(&block);
      getOrCreate<mlir::dataflow::Executable>(blockStart)->blockContentSubscribe(this);
      if (mlir::failed(mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(blockStart))) {
        return mlir::failure();
      }
      for (mlir::Operation &nestedOp : block) {
        if (mlir::failed(initializeRecursivelyInProgramOrder(&nestedOp))) {
          return mlir::failure();
        }
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult
AbstractSparseForwardDataFlowAnalysis::visitOperationDuringInitialization(mlir::Operation *op) {
  if (op->getNumResults() == 0) {
    // Preserve LLZK's zero-result transfer behavior for live effect ops such as
    // constraints, assertions, and writes.
    return visitZeroResultOperation(op);
  }
  return mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(getProgramPointAfter(op));
}

bool AbstractSparseForwardDataFlowAnalysis::isOperationLive(mlir::Operation *op) {
  if (op->getBlock() == nullptr) {
    return true;
  }
  return getOrCreate<mlir::dataflow::Executable>(getProgramPointBefore(op->getBlock()))->isLive();
}

llvm::SmallVector<const AbstractSparseLattice *, 4>
AbstractSparseForwardDataFlowAnalysis::collectOperandLatticesAndSubscribe(mlir::Operation *op) {
  llvm::SmallVector<const AbstractSparseLattice *, 4> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (mlir::Value operand : op->getOperands()) {
    AbstractSparseLattice *operandLattice = getLatticeElement(operand);
    operandLattice->useDefSubscribe(this);
    operandLattices.push_back(operandLattice);
  }
  return operandLattices;
}

mlir::LogicalResult AbstractSparseForwardDataFlowAnalysis::visitZeroResultCallOperation(
    mlir::CallOpInterface call, mlir::ArrayRef<const AbstractSparseLattice *> operandLattices
) {
  mlir::ArrayRef<AbstractSparseLattice *> emptyResultLattices;

  // Preserve the external-call hook. LLZK analyses may use it for no-result
  // call side effects even when there are no result lattices to update.
  auto callable = llvm::dyn_cast_if_present<mlir::CallableOpInterface>(call.resolveCallable());
  if (!getSolverConfig().isInterprocedural() || (callable && !callable.getCallableRegion())) {
    visitExternalCallImpl(call, operandLattices, emptyResultLattices);
    return mlir::success();
  }

  // Internal zero-result calls have no result lattices, but keep a callgraph
  // dependency so later predecessor updates can revisit the call site.
  mlir::Operation *callOp = call.getOperation();
  (void)getOrCreateFor<mlir::dataflow::PredecessorState>(
      getProgramPointAfter(callOp), getProgramPointAfter(callOp)
  );
  return mlir::success();
}

mlir::LogicalResult
AbstractSparseForwardDataFlowAnalysis::visitZeroResultOperation(mlir::Operation *op) {
  if (!isOperationLive(op)) {
    return mlir::success();
  }

  // Region-branch operations are fully owned by upstream control-flow
  // propagation. For zero-result region branches, there are no parent result
  // lattices for this compatibility path to update.
  if (llvm::isa<mlir::RegionBranchOpInterface>(op)) {
    return mlir::success();
  }

  auto operandLattices = collectOperandLatticesAndSubscribe(op);

  if (auto call = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
    return visitZeroResultCallOperation(call, operandLattices);
  }

  // Invoke the typed operation transfer function with an empty result range.
  mlir::ArrayRef<AbstractSparseLattice *> emptyResultLattices;
  return visitOperationImpl(op, operandLattices, emptyResultLattices);
}

} // namespace llzk::dataflow

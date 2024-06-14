// SPDX-License-Identifier: Apache-2.0

//===--- AttrBase.td -- Base Attribute Definitions -----------*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors
//
// =============================================================================
//
// This file contains the base set of constructs for defining Attribute classes.
//
//===----------------------------------------------------------------------===//

#ifndef ATTRBASE_TD
#define ATTRBASE_TD

include "mlir/IR/AttrTypeBase.td"

// Attribute for tensor layouts.
class BaseLayoutAttr<Dialect dialect, string name, list<Trait> traits = []>
	: AttrDef<dialect, name, traits> {
  let mnemonic = "layout";
}

def LayoutAttr : Attr<
  CPred<"mlir::isa<::mlir::Attribute>($_self) ">,
        //"&& mlir::cast<Attribute>($_self).getMnemonic().equal(\"layout\")">,
  "layout attribute"
>;

#endif // ATTRBASE_TD

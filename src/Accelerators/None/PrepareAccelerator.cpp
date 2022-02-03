/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- PrepareAccelerator.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Dummy file to add accelerator passes when there are no accelerators being targeted.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/OMAccelerator.hpp"
#include "src/Support/OMOptions.hpp"
#include <iostream>

namespace mlir {
    class OMAbsentAccelerator : public OMAccelerator {
        private:
            static bool initialized;
        public:    
            OMAbsentAccelerator() {
                if (!initialized) {
                    initialized = true;
                    OMAcceleratorTargets.push_back(this);
                }

            };

            void prepareAccelerator() {
                if (acceleratorTarget == "NONE") {
                   std::cout << "No accelerator targeted" << std::endl;
                }
            };

    };

    bool OMAbsentAccelerator::initialized = false;
    static OMAccelerator absentAccelerator;

}

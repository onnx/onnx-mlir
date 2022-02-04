#include "src/Accelerators/OMAccelerator.hpp"

namespace mlir {
class OMnnpaAccelerator : public OMAccelerator {
private:
  static bool initialized;

public:
  OMnnpaAccelerator();

  void prepareAccelerator() override; 
};


} // namespace mlir
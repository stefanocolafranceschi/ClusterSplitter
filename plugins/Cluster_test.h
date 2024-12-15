#ifndef DataFormats_SiPixelClusterSoA_test_alpaka_Hits_test_h
#define DataFormats_SiPixelClusterSoA_test_alpaka_Hits_test_h

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testSiPixelClusterSoA {

  // Corrected function declaration
  void runKernels(SiPixelClustersSoAView& clusters, Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testSiPixelClusterSoA

#endif  // DataFormats_SiPixelClusterSoA_test_alpaka_Hits_test_h
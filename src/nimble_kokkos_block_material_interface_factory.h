/*
 * nimble_kokkos_block_material_interface_factory.h
 */

#ifndef SRC_NIMBLE_KOKKOS_BLOCK_MATERIAL_INTERFACE_FACTORY_H_
#define SRC_NIMBLE_KOKKOS_BLOCK_MATERIAL_INTERFACE_FACTORY_H_

#include <memory>
#include <vector>

#include "nimble_kokkos_block_material_interface.h"

namespace nimble_kokkos {

class ModelData;

class BlockMaterialInterfaceFactory {
 public:
  BlockMaterialInterfaceFactory() = default;
  virtual ~BlockMaterialInterfaceFactory() = default;

  virtual std::shared_ptr<nimble_kokkos::BlockMaterialInterface> create(const double time_n, const double time_np1,
                                                         const std::vector<nimble_kokkos::BlockData> &blocks,
                                                         nimble_kokkos::ModelData *model_data) const
  {
    return std::make_shared<nimble_kokkos::BlockMaterialInterface>(time_n, time_np1, blocks, model_data);
  }
};

}

#endif /* SRC_NIMBLE_KOKKOS_BLOCK_MATERIAL_INTERFACE_FACTORY_H_ */

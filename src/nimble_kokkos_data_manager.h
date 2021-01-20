/*
//@HEADER
// ************************************************************************
//
//                                NimbleSM
//                             Copyright 2018
//   National Technology & Engineering Solutions of Sandia, LLC (NTESS)
//
// Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government
// retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
// NO EVENT SHALL NTESS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact David Littlewood (djlittl@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef NIMBLE_KOKKOS_DATA_MANAGER_H
#define NIMBLE_KOKKOS_DATA_MANAGER_H

#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <memory>

#include "nimble_kokkos_defs.h"
#include "nimble_data_utils.h"
#include "nimble_data_manager.h"


namespace nimble {

enum class FieldID : char {
  FailFlag = -1,
  Acceleration,
  ContactForce,
  DeformationGradient,
  Displacement,
  InternalForce,
  LumpedMass,
  ReferenceCoordinate,
  Stress,
  UnrotatedStress,
  Velocity,
  Volume
};

static std::map< nimble::FieldID, std::string > fieldToLabel =
    {
        {FieldID::Acceleration, "acceleration"},
        {FieldID::ContactForce, "contact_force"},
        {FieldID::DeformationGradient, "deformation_gradient"},
        {FieldID::Displacement, "displacement"},
        {FieldID::InternalForce, "internal_force"},
        {FieldID::LumpedMass, "lumped_mass"},
        {FieldID::ReferenceCoordinate, "reference_coordinate"},
        {FieldID::Stress, "stress"},
        {FieldID::UnrotatedStress, "unrotated_stress"},
        {FieldID::Velocity, "velocity"},
        {FieldID::Volume, "volume"}
    };

static nimble::FieldID GetFieldID(const std::string &label) {
  for (const auto& ipair : fieldToLabel) {
    auto myLabel = ipair.second;
    if (myLabel == label)
      return ipair.first;
  }
  return FieldID::FailFlag;
}

}

namespace nimble_kokkos {

class ExodusOutputManager;

struct FieldIds {
  int deformation_gradient = -1;
  int stress = -1;
  int unrotated_stress = -1;

  int reference_coordinates = -1;
  int displacement = -1;
  int velocity = -1;
  int acceleration = -1;

  int lumped_mass = -1;
  int internal_force = -1;
  int contact_force = -1;
};

class ModelData : public nimble::BaseModelData
{

 public:

  using FieldIndex = int;
  using Index = int;

  int AllocateNodeData(nimble::Length length,
                       std::string label,
                       int num_objects) override
  { return -1; }

  void AllocateNodeData(nimble::Length length,
                        nimble::FieldID field,
                        int num_objects);

  void AllocateElementData(int block_id,
                           nimble::Length length,
                           nimble::FieldID field,
                           int num_objects);

protected:

  int AllocateElementData(int block_id,
                          nimble::Length length,
                          const std::string &label,
                          int num_objects);

public:

  void AllocateIntegrationPointData(int block_id,
                                    nimble::Length length,
                                    nimble::FieldID field,
                                    int num_objects,
                                    std::vector<double> initial_value = std::vector<double>());

  FieldIndex GetFieldId(const std::string& field_label) const override
  { return field_label_to_field_id_map_.at(field_label); }

  std::vector<int> GetBlockIds() const ;

  std::vector<std::string> GetScalarNodeDataLabels() const ;

  std::vector<std::string> GetVectorNodeDataLabels() const ;

  std::vector<std::string> GetSymmetricTensorIntegrationPointDataLabels(int block_id) const ;

  std::vector<std::string> GetFullTensorIntegrationPointDataLabels(int block_id) const ;

  HostScalarNodeView GetHostScalarNodeData(ModelData::FieldIndex field_id);

  HostVectorNodeView GetHostVectorNodeData(ModelData::FieldIndex field_id);

  HostSymTensorIntPtView GetHostSymTensorIntegrationPointData(int block_id,
                                                              ModelData::FieldIndex field_id,
                                                              nimble::Step step);

  HostFullTensorIntPtView GetHostFullTensorIntegrationPointData(int block_id,
                                                                ModelData::FieldIndex field_id,
                                                                nimble::Step step);

  HostScalarElemView GetHostScalarElementData(int block_id,
                                              ModelData::FieldIndex field_id);

  HostSymTensorElemView GetHostSymTensorElementData(int block_id,
                                                    ModelData::FieldIndex field_id);

  HostFullTensorElemView GetHostFullTensorElementData(int block_id,
                                                      ModelData::FieldIndex field_id);

  DeviceScalarNodeView GetDeviceScalarNodeData(ModelData::FieldIndex field_id);

  DeviceVectorNodeView GetDeviceVectorNodeData(ModelData::FieldIndex field_id);

  DeviceSymTensorIntPtView GetDeviceSymTensorIntegrationPointData(int block_id,
                                                                  nimble::FieldID field,
                                                                  nimble::Step step);

  DeviceSymTensorIntPtView GetDeviceSymTensorIntegrationPointData(int block_id,
                                                                  int field_id,
                                                                  nimble::Step step);

  DeviceFullTensorIntPtView GetDeviceFullTensorIntegrationPointData(int block_id,
                                                                    nimble::FieldID field,
                                                                    nimble::Step step);

  DeviceFullTensorIntPtView GetDeviceFullTensorIntegrationPointData(int block_id,
                                                                    int field_id,
                                                                    nimble::Step step);

  DeviceScalarElemView GetDeviceScalarElementData(int block_id,
                                                  ModelData::FieldIndex field_id);

  DeviceSymTensorElemView GetDeviceSymTensorElementData(int block_id,
                                                        ModelData::FieldIndex field_id);

  DeviceFullTensorElemView GetDeviceFullTensorElementData(int block_id,
                                                          ModelData::FieldIndex field_id);

  DeviceScalarNodeGatheredView GatherScalarNodeData(ModelData::FieldIndex field_id,
                                                    int num_elements,
                                                    int num_nodes_per_element,
                                                    const DeviceElementConnectivityView& elem_conn_d,
                                                    DeviceScalarNodeGatheredView gathered_view_d);

  DeviceVectorNodeGatheredView GatherVectorNodeData(ModelData::FieldIndex field_id,
                                                    int num_elements,
                                                    int num_nodes_per_element,
                                                    const DeviceElementConnectivityView& elem_conn_d,
                                                    DeviceVectorNodeGatheredView gathered_view_d);

  void ScatterScalarNodeData(ModelData::FieldIndex field_id,
                             int num_elements,
                             int num_nodes_per_element,
                             const DeviceElementConnectivityView& elem_conn_d,
                             const DeviceScalarNodeGatheredView& gathered_view_d);

  void ScatterVectorNodeData(ModelData::FieldIndex field_id,
                             int num_elements,
                             int num_nodes_per_element,
                             const DeviceElementConnectivityView& elem_conn_d,
                             const DeviceVectorNodeGatheredView& gathered_view_d);

  void SetReferenceCoordinates(const nimble::GenesisMesh &mesh) override;

#ifndef KOKKOS_ENABLE_QTHREADS
  void ScatterScalarNodeDataUsingKokkosScatterView(ModelData::FieldIndex field_id,
                                                   int num_elements,
                                                   int num_nodes_per_element,
                                                   const DeviceElementConnectivityView& elem_conn_d,
                                                   const DeviceScalarNodeGatheredView& gathered_view_d);
#endif

  HostVectorNodeView GetHostVectorNodeData(nimble::FieldID field_id);

  DeviceVectorNodeView GetDeviceVectorNodeData(nimble::FieldID field_id);

  HostScalarElemView GetHostScalarElementData(int block_id,
                                              nimble::FieldID field);
  DeviceScalarNodeView GetDeviceScalarNodeData(nimble::FieldID field);

  HostScalarNodeView GetHostScalarNodeData(nimble::FieldID field);

protected:

  using Data = std::unique_ptr< FieldBase >;

  ModelData::FieldIndex MapToFieldIndex(const std::string& label);

  //
  //--- Protected Variables
  //

  std::map<std::string, FieldIndex> field_label_to_field_id_map_;

  std::vector< Data > host_node_data_;
  std::vector< Data > device_node_data_;

  std::map<FieldIndex, Index> field_id_to_host_node_data_index_;
  std::map<FieldIndex, Index> field_id_to_device_node_data_index_;

  std::map<int, Index> block_id_to_element_data_index_;
  std::vector< std::vector< Data > > host_element_data_;
  std::vector< std::vector< Data > > device_element_data_;
  std::vector< std::map< FieldIndex, Index> > field_id_to_host_element_data_index_;
  std::vector< std::map< FieldIndex, Index> > field_id_to_device_element_data_index_;

  std::map<int, Index> block_id_to_integration_point_data_index_;
  std::vector< std::vector< Data > > host_integration_point_data_step_n_;
  std::vector< std::vector< Data > > host_integration_point_data_step_np1_;
  std::vector< std::vector< Data > > device_integration_point_data_step_n_;
  std::vector< std::vector< Data > > device_integration_point_data_step_np1_;
  std::vector< std::map< FieldIndex, Index> > field_id_to_host_integration_point_data_index_;
  std::vector< std::map< FieldIndex, Index> > field_id_to_device_integration_point_data_index_;

  friend class nimble_kokkos::ExodusOutputManager;

public:

  DeviceVectorNodeGatheredView
  GatherVectorNodeData(nimble::FieldID field, int num_elements,
                       int num_nodes_per_element,
                       const DeviceElementConnectivityView &elem_conn_d,
                       DeviceVectorNodeGatheredView gathered_view_d);
  void
  ScatterScalarNodeData(nimble::FieldID field, int num_elements,
                        int num_nodes_per_element,
                        const DeviceElementConnectivityView &elem_conn_d,
                        const DeviceScalarNodeGatheredView &gathered_view_d);
  void
  ScatterVectorNodeData(nimble::FieldID field, int num_elements,
                        int num_nodes_per_element,
                        const DeviceElementConnectivityView &elem_conn_d,
                        const DeviceVectorNodeGatheredView &gathered_view_d);
};

class DataManager {

  public:

    DataManager() {}

    virtual ~DataManager() {}

    ModelData& GetMacroScaleData();

 protected:

    ModelData macroscale_data_;
};

} // namespace

#endif

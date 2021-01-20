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

#include "nimble_kokkos_data_manager.h"
#include <stdexcept>

#include <Kokkos_ScatterView.hpp>

namespace nimble_kokkos {

ModelData::FieldIndex ModelData::MapToFieldIndex(const std::string& label) {

  ModelData::FieldIndex myIndex = -1;
  auto it = field_label_to_field_id_map_.find(label);
  if (it == field_label_to_field_id_map_.end()) {
    myIndex = field_label_to_field_id_map_.size();
    field_label_to_field_id_map_[label] = myIndex;
  }
  else {
    myIndex = it->second;
  }

  return myIndex;
}

void ModelData::AllocateNodeData(nimble::Length length,
                                nimble::FieldID field,
                                int num_objects) {

  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = MapToFieldIndex(label);

  if (length == nimble::SCALAR) {
    // device_node_data_ is of type std::vector< std::unique_ptr< FieldBase > >
    // Here label is used for the definition of the Kokkos entity
    device_node_data_.emplace_back( new Field< FieldType::DeviceScalarNode >( label, num_objects ) );
  }
  else if (length == nimble::VECTOR) {
    // Here label is used for the definition of the Kokkos entity
    device_node_data_.emplace_back( new Field< FieldType::DeviceVectorNode >( label, num_objects ) );
  }
  else {
    throw std::logic_error("\nError:  Invalid device data length in nimble_kokkos::ModelData::AllocateNodeData().\n");
  }

  field_id_to_device_node_data_index_[myFieldIndex] = device_node_data_.size() - 1;

  FieldBase * d_field = device_node_data_.back().get();

  if (d_field->type() == FieldType::DeviceScalarNode) {
    auto field_ptr = dynamic_cast< Field< FieldType::DeviceScalarNode> * >( d_field );
    Field< FieldType::DeviceScalarNode >::View d_view = field_ptr->data();
    auto h_view = Kokkos::create_mirror_view( d_view );
    host_node_data_.emplace_back( new Field< FieldType::HostScalarNode >( h_view ) );
  }
  else if (d_field->type() == FieldType::DeviceVectorNode) {
    auto field_ptr = dynamic_cast< Field< FieldType::DeviceVectorNode> * >( d_field );
    Field< FieldType::DeviceVectorNode >::View d_view = field_ptr->data();
    auto h_view = Kokkos::create_mirror_view( d_view );
    host_node_data_.emplace_back( new Field< FieldType::HostVectorNode >( h_view ) );
  }
  else {
    throw std::logic_error("\nError:  Invalid host data length in nimble_kokkos::ModelData::AllocateNodeData().\n");
  }

  field_id_to_host_node_data_index_[myFieldIndex] = host_node_data_.size() - 1;

}


void ModelData::AllocateElementData(int block_id,
                                    nimble::Length length,
                                    nimble::FieldID field,
                                    int num_objects) {

  auto label = nimble::fieldToLabel[field];
  AllocateElementData(block_id, length, label, num_objects);

}

int ModelData::AllocateElementData(int block_id,
                                   nimble::Length length,
                                   const std::string &label,
                                   int num_objects) {

  auto myFieldIndex = MapToFieldIndex(label);

  if (block_id_to_element_data_index_.find(block_id) == block_id_to_element_data_index_.end()) {
    block_id_to_element_data_index_[block_id] = host_element_data_.size();
    host_element_data_.emplace_back(std::vector< Data >());
    device_element_data_.emplace_back(std::vector< Data >());
    field_id_to_host_element_data_index_.emplace_back(std::map< ModelData::FieldIndex, ModelData::Index>());
    field_id_to_device_element_data_index_.emplace_back(std::map<ModelData::FieldIndex, ModelData::Index>());
  }
  int block_index = block_id_to_element_data_index_.at(block_id);

  if (length == nimble::SCALAR) {
    // Here label is used for the definition of the Kokkos entity
    device_element_data_.at(block_index).emplace_back( new Field< FieldType::DeviceScalarElem >( label, num_objects ) );
  }
  else if (length == nimble::SYMMETRIC_TENSOR) {
    // Here label is used for the definition of the Kokkos entity
    device_element_data_.at(block_index).emplace_back( new Field< FieldType::DeviceSymTensorElem >( label, num_objects ) );
  }
  else if (length == nimble::FULL_TENSOR) {
    // Here label is used for the definition of the Kokkos entity
    device_element_data_.at(block_index).emplace_back( new Field< FieldType::DeviceFullTensorElem >( label, num_objects ) );
  }
  else {
    throw std::logic_error("\nError:  Invalid device data length in nimble_kokkos::ModelData::AllocateElementData().\n");
  }

  field_id_to_device_element_data_index_.at(block_index)[myFieldIndex] =
      static_cast<Index>( device_element_data_.at(block_index).size() - 1 );

  FieldBase * d_field = device_element_data_.at(block_index).back().get();

  if (d_field->type() == FieldType::DeviceScalarElem) {
    auto field_ptr = dynamic_cast< Field< FieldType::DeviceScalarElem> * >( d_field );
    Field< FieldType::DeviceScalarElem >::View d_view = field_ptr->data();
    auto h_view = Kokkos::create_mirror_view( d_view );
    host_element_data_.at(block_index).emplace_back( new Field< FieldType::HostScalarElem >( h_view ) );
  }
  else if (d_field->type() == FieldType::DeviceSymTensorElem) {
    auto field_ptr = dynamic_cast< Field< FieldType::DeviceSymTensorElem> * >( d_field );
    Field< FieldType::DeviceSymTensorElem >::View d_view = field_ptr->data();
    auto h_view = Kokkos::create_mirror_view( d_view );
    host_element_data_.at(block_index).emplace_back( new Field< FieldType::HostSymTensorElem >( h_view ) );
  }
  else if (d_field->type() == FieldType::DeviceFullTensorElem) {
    auto field_ptr = dynamic_cast< Field< FieldType::DeviceFullTensorElem> * >( d_field );
    Field< FieldType::DeviceFullTensorElem >::View d_view = field_ptr->data();
    auto h_view = Kokkos::create_mirror_view( d_view );
    host_element_data_.at(block_index).emplace_back( new Field< FieldType::HostFullTensorElem >( h_view ) );
  }
  else {
    throw std::logic_error("\nError:  Invalid host data length in nimble_kokkos::ModelData::AllocateElementData().\n");
  }

  field_id_to_host_element_data_index_.at(block_index)[myFieldIndex] =
      static_cast<Index>( host_element_data_.at(block_index).size() - 1 );

  return myFieldIndex;

}

void ModelData::AllocateIntegrationPointData(int block_id,
                                            nimble::Length length,
                                            nimble::FieldID field,
                                            int num_objects,
                                            std::vector<double> initial_value) {
  bool set_initial_value = (!initial_value.empty());
  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = MapToFieldIndex(label);

  if (block_id_to_integration_point_data_index_.find(block_id) == block_id_to_integration_point_data_index_.end()) {
    block_id_to_integration_point_data_index_[block_id] = host_integration_point_data_step_n_.size();
    host_integration_point_data_step_n_.emplace_back(std::vector< Data >());
    host_integration_point_data_step_np1_.emplace_back(std::vector< Data >());
    device_integration_point_data_step_n_.emplace_back(std::vector< Data >());
    device_integration_point_data_step_np1_.emplace_back(std::vector< Data >());
    field_id_to_host_integration_point_data_index_.emplace_back(std::map< ModelData::FieldIndex, ModelData::Index>());
    field_id_to_device_integration_point_data_index_.emplace_back(std::map< ModelData::FieldIndex, ModelData::Index>());
  }
  int block_index = block_id_to_integration_point_data_index_.at(block_id);

  if (length == nimble::SCALAR) {
    device_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::DeviceScalarNode >( label, num_objects ) );
    device_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::DeviceScalarNode >( label, num_objects ) );
  }
  else if (length == nimble::VECTOR) {
    device_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::DeviceVectorNode >( label, num_objects ) );
    device_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::DeviceVectorNode >( label, num_objects ) );
  }
  else if (length == nimble::SYMMETRIC_TENSOR) {
    device_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::DeviceSymTensorIntPt >( label, num_objects ) );
    device_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::DeviceSymTensorIntPt >( label, num_objects ) );
  }
  else if (length == nimble::FULL_TENSOR) {
    device_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::DeviceFullTensorIntPt >( label, num_objects ) );
    device_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::DeviceFullTensorIntPt >( label, num_objects ) );
  }
  else {
    throw std::logic_error("\nError:  Invalid device data length in nimble_kokkos::ModelData::AllocateIntegrationPointData().\n");
  }

  field_id_to_device_integration_point_data_index_.at(block_index)[myFieldIndex] =
      static_cast<Index>( device_integration_point_data_step_n_.at(block_index).size() - 1 );

  FieldBase * d_field_step_n = device_integration_point_data_step_n_.at(block_index).back().get();
  FieldBase * d_field_step_np1 = device_integration_point_data_step_np1_.at(block_index).back().get();

  if (d_field_step_n->type() == FieldType::DeviceScalarNode) {
    auto field_step_n = dynamic_cast< Field< FieldType::DeviceScalarNode> * >( d_field_step_n );
    Field< FieldType::DeviceScalarNode >::View d_view_step_n = field_step_n->data();
    auto h_view_step_n = Kokkos::create_mirror_view( d_view_step_n );
    host_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::HostScalarNode >( h_view_step_n ) );

    auto field_step_np1 = dynamic_cast< Field< FieldType::DeviceScalarNode> * >( d_field_step_np1 );
    Field< FieldType::DeviceScalarNode >::View d_view_step_np1 = field_step_np1->data();
    auto h_view_step_np1 = Kokkos::create_mirror_view( d_view_step_np1 );
    host_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::HostScalarNode >( h_view_step_np1 ) );

    if (set_initial_value) {
      int num_elem = h_view_step_n.extent(0);
      for (int i_elem=0 ; i_elem<num_elem ; ++i_elem) {
        h_view_step_n(i_elem) = initial_value.at(0);
        h_view_step_np1(i_elem) = initial_value.at(0);
      }
      Kokkos::deep_copy(d_view_step_n, h_view_step_n);
      Kokkos::deep_copy(d_view_step_np1, h_view_step_np1);
    }
  }
  else if (d_field_step_n->type() == FieldType::DeviceVectorNode) {
    auto field_step_n = dynamic_cast< Field< FieldType::DeviceVectorNode> * >( d_field_step_n );
    Field< FieldType::DeviceVectorNode >::View d_view_step_n = field_step_n->data();
    auto h_view_step_n = Kokkos::create_mirror_view( d_view_step_n );
    host_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::HostVectorNode >( h_view_step_n ) );

    auto field_step_np1 = dynamic_cast< Field< FieldType::DeviceVectorNode> * >( d_field_step_np1 );
    Field< FieldType::DeviceVectorNode >::View d_view_step_np1 = field_step_np1->data();
    auto h_view_step_np1 = Kokkos::create_mirror_view( d_view_step_np1 );
    host_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::HostVectorNode >( h_view_step_np1 ) );

    if (set_initial_value) {
      int num_elem = h_view_step_n.extent(0);
      int num_entries = 3;
      for (int i_elem=0 ; i_elem<num_elem ; ++i_elem) {
        for (int i_entry=0 ; i_entry<num_entries ; ++i_entry) {
          h_view_step_n(i_elem, i_entry) = initial_value.at(i_entry);
          h_view_step_np1(i_elem, i_entry) = initial_value.at(i_entry);
        }
      }
      Kokkos::deep_copy(d_view_step_n, h_view_step_n);
      Kokkos::deep_copy(d_view_step_np1, h_view_step_np1);
    }
  }
  else if (d_field_step_n->type() == FieldType::DeviceSymTensorIntPt) {
    auto field_step_n = dynamic_cast< Field< FieldType::DeviceSymTensorIntPt> * >( d_field_step_n );
    Field< FieldType::DeviceSymTensorIntPt >::View d_view_step_n = field_step_n->data();
    auto h_view_step_n = Kokkos::create_mirror_view( d_view_step_n );
    host_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::HostSymTensorIntPt >( h_view_step_n ) );

    auto field_step_np1 = dynamic_cast< Field< FieldType::DeviceSymTensorIntPt> * >( d_field_step_np1 );
    Field< FieldType::DeviceSymTensorIntPt >::View d_view_step_np1 = field_step_np1->data();
    auto h_view_step_np1 = Kokkos::create_mirror_view( d_view_step_np1 );
    host_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::HostSymTensorIntPt >( h_view_step_np1 ) );

    if (set_initial_value) {
      int num_elem = h_view_step_n.extent(0);
      int num_int_pt = h_view_step_n.extent(1);
      int num_entries = 6;
      for (int i_elem=0 ; i_elem<num_elem ; ++i_elem) {
        for (int i_int_pt=0 ; i_int_pt<num_int_pt ; ++i_int_pt) {
          for (int i_entry=0 ; i_entry<num_entries ; ++i_entry) {
            h_view_step_n(i_elem, i_int_pt, i_entry) = initial_value.at(i_entry);
            h_view_step_np1(i_elem, i_int_pt, i_entry) = initial_value.at(i_entry);
          }
        }
      }
      Kokkos::deep_copy(d_view_step_n, h_view_step_n);
      Kokkos::deep_copy(d_view_step_np1, h_view_step_np1);
    }
  }
  else if (d_field_step_n->type() == FieldType::DeviceFullTensorIntPt) {
    auto field_step_n = dynamic_cast< Field< FieldType::DeviceFullTensorIntPt> * >( d_field_step_n );
    Field< FieldType::DeviceFullTensorIntPt >::View d_view_step_n = field_step_n->data();
    auto h_view_step_n = Kokkos::create_mirror_view( d_view_step_n );
    host_integration_point_data_step_n_.at(block_index).emplace_back( new Field< FieldType::HostFullTensorIntPt >( h_view_step_n ) );

    auto field_step_np1 = dynamic_cast< Field< FieldType::DeviceFullTensorIntPt> * >( d_field_step_np1 );
    Field< FieldType::DeviceFullTensorIntPt >::View d_view_step_np1 = field_step_np1->data();
    auto h_view_step_np1 = Kokkos::create_mirror_view( d_view_step_np1 );
    host_integration_point_data_step_np1_.at(block_index).emplace_back( new Field< FieldType::HostFullTensorIntPt >( h_view_step_np1 ) );

    if (set_initial_value) {
      int num_elem = h_view_step_n.extent(0);
      int num_int_pt = h_view_step_n.extent(1);
      int num_entries = 9;
      for (int i_elem=0 ; i_elem<num_elem ; ++i_elem) {
        for (int i_int_pt=0 ; i_int_pt<num_int_pt ; ++i_int_pt) {
          for (int i_entry=0 ; i_entry<num_entries ; ++i_entry) {
            h_view_step_n(i_elem, i_int_pt, i_entry) = initial_value.at(i_entry);
            h_view_step_np1(i_elem, i_int_pt, i_entry) = initial_value.at(i_entry);
          }
        }
      }
      Kokkos::deep_copy(d_view_step_n, h_view_step_n);
      Kokkos::deep_copy(d_view_step_np1, h_view_step_np1);
    }
  }
  else {
    throw std::logic_error("\nError:  Invalid host data length in nimble_kokkos::ModelData::AllocateElementData().\n");
  }

  field_id_to_host_integration_point_data_index_.at(block_index)[myFieldIndex] =
      static_cast<Index>( host_integration_point_data_step_n_.at(block_index).size() - 1 );

}

std::vector<int> ModelData::GetBlockIds() const {
    std::vector<int> block_ids;
    for (auto const & entry : block_id_to_integration_point_data_index_) {
      block_ids.push_back(entry.first);
    }
    return block_ids;
  }

  std::vector<std::string> ModelData::GetScalarNodeDataLabels() const {
    std::vector<std::string> node_data_labels;
    for (auto const & entry : field_label_to_field_id_map_) {
      auto it = field_id_to_host_node_data_index_.find(entry.second);
      if (it == field_id_to_host_node_data_index_.end())
        continue;
      auto node_data_index = it->second;
      if (host_node_data_.at(node_data_index)->type() == FieldType::HostScalarNode)
        node_data_labels.push_back(entry.first);
    }
    return node_data_labels;
  }

  std::vector<std::string> ModelData::GetVectorNodeDataLabels() const {
    std::vector<std::string> node_data_labels;
    for (auto const & entry : field_label_to_field_id_map_) {
      auto it = field_id_to_host_node_data_index_.find(entry.second);
      if (it == field_id_to_host_node_data_index_.end())
        continue;
      auto node_data_index = it->second;
      if (host_node_data_.at(node_data_index)->type() == FieldType::HostVectorNode)
        node_data_labels.push_back(entry.first);
    }
    return node_data_labels;
  }

std::vector<std::string> ModelData::GetSymmetricTensorIntegrationPointDataLabels(int block_id) const {
  auto block_index = block_id_to_integration_point_data_index_.at(block_id);
  std::vector<std::string> ipt_data_labels;
  for (auto const & entry : field_label_to_field_id_map_) {
    auto myMap = field_id_to_device_integration_point_data_index_.at(block_index);
    auto it = myMap.find(entry.second);
    if (it == myMap.end())
      continue;
    std::string const & field_label = entry.first;
    auto ipt_data_index = it->second;
    if (device_integration_point_data_step_np1_.at(block_index).at(ipt_data_index)->type() == FieldType::DeviceSymTensorIntPt) {
      if (std::find(ipt_data_labels.begin(), ipt_data_labels.end(), field_label) == ipt_data_labels.end()) {
        ipt_data_labels.push_back(field_label);
      }
    }
  }
  return ipt_data_labels;
}

std::vector<std::string> ModelData::GetFullTensorIntegrationPointDataLabels(int block_id) const {
  auto block_index = block_id_to_integration_point_data_index_.at(block_id);
  std::vector<std::string> ipt_data_labels;
  for (auto const & entry : field_label_to_field_id_map_) {
    auto myMap = field_id_to_device_integration_point_data_index_.at(block_index);
    auto it = myMap.find(entry.second);
    if (it == myMap.end())
      continue;
    std::string const & field_label = entry.first;
    auto ipt_data_index = it->second;
    if (device_integration_point_data_step_np1_.at(block_index).at(ipt_data_index)->type() == FieldType::DeviceFullTensorIntPt) {
      if (std::find(ipt_data_labels.begin(), ipt_data_labels.end(), field_label) == ipt_data_labels.end()) {
        ipt_data_labels.push_back(field_label);
      }
    }
  }
  return ipt_data_labels;
}

HostScalarNodeView ModelData::GetHostScalarNodeData(nimble::FieldID field) {
  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = GetFieldId(label);
  return GetHostScalarNodeData(myFieldIndex);
}

HostScalarNodeView ModelData::GetHostScalarNodeData(ModelData::FieldIndex field_id) {
  auto index = field_id_to_host_node_data_index_.at(field_id);
  FieldBase* base_field_ptr = host_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostScalarNode >* >(base_field_ptr);
  return derived_field_ptr->data();
}

HostVectorNodeView ModelData::GetHostVectorNodeData(nimble::FieldID field) {
  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = GetFieldId(label);
  return GetHostVectorNodeData(myFieldIndex);
}

HostVectorNodeView ModelData::GetHostVectorNodeData(ModelData::FieldIndex field_id) {
  auto index = field_id_to_host_node_data_index_.at(field_id);
  FieldBase* base_field_ptr = host_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostVectorNode >* >(base_field_ptr);
  return derived_field_ptr->data();
}

HostScalarElemView ModelData::GetHostScalarElementData(int block_id,
                                                       nimble::FieldID field) {
  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = GetFieldId(label);
  return GetHostScalarElementData(block_id, myFieldIndex);
}

HostScalarElemView ModelData::GetHostScalarElementData(int block_id,
                                                       ModelData::FieldIndex field_id) {
  auto block_index = block_id_to_element_data_index_.at(block_id);
  auto data_index = field_id_to_host_element_data_index_.at(block_index).at(field_id);
  FieldBase* base_field_ptr(nullptr);
  base_field_ptr = host_element_data_.at(block_index).at(data_index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostScalarElem >* >(base_field_ptr);
  return derived_field_ptr->data();
}

HostSymTensorIntPtView ModelData::GetHostSymTensorIntegrationPointData(int block_id,
                                                                       ModelData::FieldIndex field_id,
                                                                       nimble::Step step) {
  int block_index = block_id_to_integration_point_data_index_.at(block_id);
  int data_index = field_id_to_host_integration_point_data_index_.at(block_index).at(field_id);
  FieldBase* base_field_ptr(nullptr);
  if (step == nimble::STEP_N) {
    base_field_ptr = host_integration_point_data_step_n_.at(block_index).at(data_index).get();
  }
  else if (step == nimble::STEP_NP1) {
    base_field_ptr = host_integration_point_data_step_np1_.at(block_index).at(data_index).get();
  }
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostSymTensorIntPt >* >(base_field_ptr);
  return derived_field_ptr->data();
}

HostFullTensorIntPtView ModelData::GetHostFullTensorIntegrationPointData(int block_id,
                                                                         ModelData::FieldIndex field_id,
                                                                         nimble::Step step) {
  int block_index = block_id_to_integration_point_data_index_.at(block_id);
  int data_index = field_id_to_host_integration_point_data_index_.at(block_index).at(field_id);
  FieldBase* base_field_ptr(nullptr);
  if (step == nimble::STEP_N) {
    base_field_ptr = host_integration_point_data_step_n_.at(block_index).at(data_index).get();
  }
  else if (step == nimble::STEP_NP1) {
    base_field_ptr = host_integration_point_data_step_np1_.at(block_index).at(data_index).get();
  }
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostFullTensorIntPt >* >(base_field_ptr);
  return derived_field_ptr->data();
}

HostSymTensorElemView ModelData::GetHostSymTensorElementData(int block_id,
                                                             ModelData::FieldIndex field_id) {
  auto block_index = block_id_to_element_data_index_.at(block_id);
  auto data_index = field_id_to_host_element_data_index_.at(block_index).at(field_id);
  auto base_field_ptr = host_element_data_.at(block_index).at(data_index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostSymTensorElem >* >(base_field_ptr);
  return derived_field_ptr->data();
}

HostFullTensorElemView ModelData::GetHostFullTensorElementData(int block_id,
                                                               ModelData::FieldIndex field_id) {
  auto block_index = block_id_to_element_data_index_.at(block_id);
  auto data_index = field_id_to_host_element_data_index_.at(block_index).at(field_id);
  auto base_field_ptr = host_element_data_.at(block_index).at(data_index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::HostFullTensorElem >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceScalarNodeView ModelData::GetDeviceScalarNodeData(nimble::FieldID field) {
  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = GetFieldId(label);
  return GetDeviceScalarNodeData(myFieldIndex);
}

DeviceScalarNodeView ModelData::GetDeviceScalarNodeData(ModelData::FieldIndex field_id) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceScalarNode >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceVectorNodeView ModelData::GetDeviceVectorNodeData(nimble::FieldID field) {
  auto label = nimble::fieldToLabel[field];
  auto myFieldIndex = GetFieldId(label);
  return GetDeviceVectorNodeData(myFieldIndex);
}

DeviceVectorNodeView ModelData::GetDeviceVectorNodeData(ModelData::FieldIndex field_id) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceVectorNode >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceSymTensorIntPtView ModelData::GetDeviceSymTensorIntegrationPointData(int block_id,
                                                                           nimble::FieldID field,
                                                                           nimble::Step step) {
  auto label = nimble::fieldToLabel[field];
  auto field_id = GetFieldId(label);
  return GetDeviceSymTensorIntegrationPointData(block_id, field_id, step);
}

DeviceSymTensorIntPtView ModelData::GetDeviceSymTensorIntegrationPointData(int block_id,
                                                                            int field_id,
                                                                             nimble::Step step) {
  auto block_index = block_id_to_integration_point_data_index_.at(block_id);
  auto data_index = field_id_to_device_integration_point_data_index_.at(block_index).at(field_id);
  FieldBase* base_field_ptr(nullptr);
  if (step == nimble::STEP_N) {
    base_field_ptr = device_integration_point_data_step_n_.at(block_index).at(data_index).get();
  }
  else if (step == nimble::STEP_NP1) {
    base_field_ptr = device_integration_point_data_step_np1_.at(block_index).at(data_index).get();
  }
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceSymTensorIntPt >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceFullTensorIntPtView ModelData::GetDeviceFullTensorIntegrationPointData(int block_id,
                                                                             nimble::FieldID field,
                                                                             nimble::Step step) {
  auto label = nimble::fieldToLabel[field];
  auto field_id = GetFieldId(label);
  return GetDeviceFullTensorIntegrationPointData(block_id, field_id, step);
}

DeviceFullTensorIntPtView ModelData::GetDeviceFullTensorIntegrationPointData(int block_id,
                                                                             int field_id,
                                                                             nimble::Step step) {
  auto block_index = block_id_to_integration_point_data_index_.at(block_id);
  auto data_index = field_id_to_device_integration_point_data_index_.at(block_index).at(field_id);
  FieldBase* base_field_ptr(nullptr);
  if (step == nimble::STEP_N) {
    base_field_ptr = device_integration_point_data_step_n_.at(block_index).at(data_index).get();
  }
  else if (step == nimble::STEP_NP1) {
    base_field_ptr = device_integration_point_data_step_np1_.at(block_index).at(data_index).get();
  }
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceFullTensorIntPt >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceScalarElemView ModelData::GetDeviceScalarElementData(int block_id,
                                                           ModelData::FieldIndex field_id) {
  auto block_index = block_id_to_element_data_index_.at(block_id);
  auto data_index = field_id_to_device_element_data_index_.at(block_index).at(field_id);
  FieldBase* base_field_ptr(nullptr);
  base_field_ptr = device_element_data_.at(block_index).at(data_index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceScalarElem >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceSymTensorElemView ModelData::GetDeviceSymTensorElementData(int block_id,
                                                                 ModelData::FieldIndex field_id) {
  auto block_index = block_id_to_element_data_index_.at(block_id);
  auto data_index = field_id_to_device_element_data_index_.at(block_index).at(field_id);
  auto base_field_ptr = device_element_data_.at(block_index).at(data_index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceSymTensorElem >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceFullTensorElemView ModelData::GetDeviceFullTensorElementData(int block_id,
                                                                   ModelData::FieldIndex field_id) {
  auto block_index = block_id_to_element_data_index_.at(block_id);
  auto data_index = field_id_to_device_element_data_index_.at(block_index).at(field_id);
  auto base_field_ptr = device_element_data_.at(block_index).at(data_index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceFullTensorElem >* >(base_field_ptr);
  return derived_field_ptr->data();
}

DeviceScalarNodeGatheredView ModelData::GatherScalarNodeData(ModelData::FieldIndex field_id,
                                                             int num_elements,
                                                             int num_nodes_per_element,
                                                             const DeviceElementConnectivityView& elem_conn_d,
                                                             DeviceScalarNodeGatheredView gathered_view_d) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceScalarNode >* >(base_field_ptr);
  auto data = derived_field_ptr->data();
  Kokkos::parallel_for("GatherScalarNodeData", num_elements, KOKKOS_LAMBDA (const int i_elem) {
    for (int i_node=0 ; i_node < num_nodes_per_element ; i_node++) {
      gathered_view_d(i_elem, i_node) = data(elem_conn_d(num_nodes_per_element*i_elem + i_node));
    }
  });
  return gathered_view_d;
}

DeviceVectorNodeGatheredView ModelData::GatherVectorNodeData(nimble::FieldID field,
                                                             int num_elements,
                                                             int num_nodes_per_element,
                                                             const DeviceElementConnectivityView& elem_conn_d,
                                                             DeviceVectorNodeGatheredView gathered_view_d) {
  auto label = nimble::fieldToLabel[field];
  auto field_id = GetFieldId(label);
  return GatherVectorNodeData(field_id, num_elements, num_nodes_per_element, elem_conn_d, gathered_view_d);
}

DeviceVectorNodeGatheredView ModelData::GatherVectorNodeData(ModelData::FieldIndex field_id,
                                                             int num_elements,
                                                             int num_nodes_per_element,
                                                             const DeviceElementConnectivityView& elem_conn_d,
                                                             DeviceVectorNodeGatheredView gathered_view_d) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceVectorNode >* >(base_field_ptr);
  auto data = derived_field_ptr->data();
  Kokkos::parallel_for("GatherVectorNodeData", num_elements, KOKKOS_LAMBDA (const int i_elem) {
    for (int i_node=0 ; i_node < num_nodes_per_element ; i_node++) {
      int node_index = elem_conn_d(num_nodes_per_element*i_elem + i_node);
      for (int i_coord=0 ; i_coord < 3 ; i_coord++) {
        gathered_view_d(i_elem, i_node, i_coord) = data(node_index, i_coord);
      }
    }
  });
  return gathered_view_d;
}

void ModelData::ScatterScalarNodeData(nimble::FieldID field,
                                      int num_elements,
                                      int num_nodes_per_element,
                                      const DeviceElementConnectivityView& elem_conn_d,
                                      const DeviceScalarNodeGatheredView& gathered_view_d) {
  auto label = nimble::fieldToLabel[field];
  auto field_id = GetFieldId(label);
  ScatterScalarNodeData(field_id, num_elements, num_nodes_per_element,
                        elem_conn_d, gathered_view_d);
}

void ModelData::ScatterScalarNodeData(ModelData::FieldIndex field_id,
                                      int num_elements,
                                      int num_nodes_per_element,
                                      const DeviceElementConnectivityView& elem_conn_d,
                                      const DeviceScalarNodeGatheredView& gathered_view_d) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceScalarNode >* >(base_field_ptr);
  Field< FieldType::DeviceScalarNode >::AtomicView data = derived_field_ptr->data();
  Kokkos::parallel_for("ScatterScalarNodeData", num_elements, KOKKOS_LAMBDA (const int i_elem) {
    for (int i_node=0 ; i_node < num_nodes_per_element ; i_node++) {
      data(elem_conn_d(num_nodes_per_element*i_elem + i_node)) += gathered_view_d(i_elem, i_node);
    }
  });
}

void ModelData::ScatterVectorNodeData(nimble::FieldID field,
                                      int num_elements,
                                      int num_nodes_per_element,
                                      const DeviceElementConnectivityView& elem_conn_d,
                                      const DeviceVectorNodeGatheredView& gathered_view_d) {
  auto label = nimble::fieldToLabel[field];
  auto field_id = GetFieldId(label);
  ScatterVectorNodeData(field_id, num_elements, num_nodes_per_element,
                        elem_conn_d, gathered_view_d);
}


void ModelData::ScatterVectorNodeData(ModelData::FieldIndex field_id,
                                      int num_elements,
                                      int num_nodes_per_element,
                                      const DeviceElementConnectivityView& elem_conn_d,
                                      const DeviceVectorNodeGatheredView& gathered_view_d) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceVectorNode >* >(base_field_ptr);
  Field< FieldType::DeviceVectorNode >::AtomicView data = derived_field_ptr->data();
  Kokkos::parallel_for("ScatterVectorNodeData", num_elements, KOKKOS_LAMBDA (const int i_elem) {
    for (int i_node=0 ; i_node < num_nodes_per_element ; i_node++) {
      int node_index = elem_conn_d(num_nodes_per_element*i_elem + i_node);
      for (int i_coord=0 ; i_coord < 3 ; i_coord++) {
        data(node_index, i_coord) += gathered_view_d(i_elem, i_node, i_coord);
      }
    }
  });
}

void ModelData::SetReferenceCoordinates(const nimble::GenesisMesh &mesh)
{
  const double *ref_coord_x = mesh.GetCoordinatesX();
  const double *ref_coord_y = mesh.GetCoordinatesY();
  const double *ref_coord_z = mesh.GetCoordinatesZ();
  auto reference_coordinate_h = GetHostVectorNodeData(nimble::FieldID::ReferenceCoordinate);
  int num_nodes = static_cast<int>(mesh.GetNumNodes());
  switch (dim_) {
  case 2:
    for (int i=0 ; i<num_nodes ; i++) {
      reference_coordinate_h(i, 0) = ref_coord_x[i];
      reference_coordinate_h(i, 1) = ref_coord_y[i];
    }
    break;
  default:
  case 3:
    for (int i=0 ; i<num_nodes ; i++) {
      reference_coordinate_h(i, 0) = ref_coord_x[i];
      reference_coordinate_h(i, 1) = ref_coord_y[i];
      reference_coordinate_h(i, 2) = ref_coord_z[i];
    }
    break;
  }
  auto reference_coordinate_d = GetDeviceVectorNodeData(nimble::FieldID::ReferenceCoordinate);
  Kokkos::deep_copy(reference_coordinate_d, reference_coordinate_h);
}

#ifndef KOKKOS_ENABLE_QTHREADS
void ModelData::ScatterScalarNodeDataUsingKokkosScatterView(ModelData::FieldIndex field_id,
                                                            int num_elements,
                                                            int num_nodes_per_element,
                                                            const DeviceElementConnectivityView& elem_conn_d,
                                                            const DeviceScalarNodeGatheredView& gathered_view_d) {
  auto index = field_id_to_device_node_data_index_.at(field_id);
  auto base_field_ptr = device_node_data_.at(index).get();
  auto derived_field_ptr = dynamic_cast< Field< FieldType::DeviceScalarNode >* >(base_field_ptr);
  auto data = derived_field_ptr->data();
  auto scatter_view = Kokkos::Experimental::create_scatter_view(data); // DJL it is a terrible idea to allocate this here
  scatter_view.reset();
  Kokkos::parallel_for("GatherVectorNodeData", num_elements, KOKKOS_LAMBDA (const int i_elem) {
    auto scattered_access = scatter_view.access();
    for (int i_node=0 ; i_node < num_nodes_per_element ; i_node++) {
      scattered_access(elem_conn_d(num_nodes_per_element*i_elem + i_node)) += gathered_view_d(i_elem, i_node);
    }
  });
  Kokkos::Experimental::contribute(data, scatter_view);
}
#endif

  ModelData& DataManager::GetMacroScaleData() {
    return macroscale_data_;
  }

} // namespace nimble

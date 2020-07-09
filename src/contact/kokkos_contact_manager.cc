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

#include "kokkos_contact_manager.h"

#ifdef NIMBLE_HAVE_KOKKOS

namespace nimble {

void
KokkosContactManager::BoundingBox(double& x_min,
                            double& x_max,
                            double& y_min,
                            double& y_max,
                            double& z_min,
                            double& z_max) const {

  double big = std::numeric_limits<double>::max();

  nimble_kokkos::DeviceScalarNodeView contact_bounding_box_d("contact_bounding_box_d", 6);
  nimble_kokkos::HostScalarNodeView contact_bounding_box_h("contact_bounding_box_h", 6);
  contact_bounding_box_h(0) = big;       // x_min
  contact_bounding_box_h(1) = -1.0*big;  // x_max
  contact_bounding_box_h(2) = big;       // y_min
  contact_bounding_box_h(3) = -1.0*big;  // y_max
  contact_bounding_box_h(4) = big;       // z_min
  contact_bounding_box_h(5) = -1.0*big;  // z_max
  Kokkos::deep_copy(contact_bounding_box_d, contact_bounding_box_h);

  nimble_kokkos::DeviceScalarNodeView coord_d = coord_d_;
  int contact_vector_size = coord_d.extent(0) / 3;

  Kokkos::parallel_for("Contact Bounding Box",
                       contact_vector_size,
                       KOKKOS_LAMBDA(const int i) {
                         double x = coord_d(3*i);
                         double y = coord_d(3*i+1);
                         double z = coord_d(3*i+2);
                         Kokkos::atomic_min_fetch(&contact_bounding_box_d(0), x);
                         Kokkos::atomic_max_fetch(&contact_bounding_box_d(1), x);
                         Kokkos::atomic_min_fetch(&contact_bounding_box_d(2), y);
                         Kokkos::atomic_max_fetch(&contact_bounding_box_d(3), y);
                         Kokkos::atomic_min_fetch(&contact_bounding_box_d(4), z);
                         Kokkos::atomic_max_fetch(&contact_bounding_box_d(5), z);
                       });

  Kokkos::deep_copy(contact_bounding_box_h, contact_bounding_box_d);
  x_min = contact_bounding_box_h(0);
  x_max = contact_bounding_box_h(1);
  y_min = contact_bounding_box_h(2);
  y_max = contact_bounding_box_h(3);
  z_min = contact_bounding_box_h(4);
  z_max = contact_bounding_box_h(5);
}

void
KokkosContactManager::InitializeContactVisualization(std::string const & contact_visualization_exodus_file_name){

  // Exodus id convention for contact visualization:
  //
  // Both node and face contact entities have a unique, parallel-consistent id called contact_entity_global_id_.
  // For faces, the contact_entity_global_id_ a bit-wise combination of the global exodus id of the parent element,
  // plus the face ordinal (1-6), plus the triangle ordinal (1-4).
  // For nodes, the contact_entity_global_id_ is the exodus global node id for the node in the original FEM mesh.
  //
  // For visualization output, we need unique, parallel-consistent node ids and element ids.  For the faces, the
  // contact_entity_global_id_ is used as the element id, and the node ids are constructed here.  For nodes, the
  // contact_entity_global_id_ is used for both the node id and the element id (sphere element containing a single node).
  //
  // For the MPI bounding boxes, both the node ids and the element id are constructed here.
  //
  //   contact faces:
  //     node ids are (3 * contact_entity_global_id_ + max_contact_entity_id + 9,
  //                   3 * contact_entity_global_id_ + max_contact_entity_id + 10,
  //                   3 * contact_entity_global_id_ + max_contact_entity_id + 11)
  //     element id is contact_entity_global_id_
  //   contact nodes
  //     node id is contact_entity_global_id_
  //     element id contact_entity_global_id_
  //   mpi partition bounding box:
  //     nodes id are (3 * max_contact_entity_id + 1,
  //                   3 * max_contact_entity_id + 2,
  //                   3 * max_contact_entity_id + 3,
  //                   3 * max_contact_entity_id + 4,
  //                   3 * max_contact_entity_id + 5,
  //                   3 * max_contact_entity_id + 6,
  //                   3 * max_contact_entity_id + 7,
  //                   3 * max_contact_entity_id + 8)
  //     element id max_contact_entity_id + 1

  // determine the maximum contact entity global id over all MPI partitions
  int max_contact_entity_id = 0;
  for (int i_face=0 ; i_face<contact_faces_h_.extent(0) ; i_face++) {
    if (contact_faces_h_[i_face].contact_entity_global_id_ > max_contact_entity_id) {
      max_contact_entity_id = contact_faces_h_[i_face].contact_entity_global_id_;
    }
  }
  for (int i_node=0 ; i_node<contact_nodes_h_.extent(0) ; i_node++) {
    if (contact_nodes_h_[i_node].contact_entity_global_id_ > max_contact_entity_id) {
      max_contact_entity_id = contact_nodes_h_[i_node].contact_entity_global_id_;
    }
  }

  if (num_ranks_ > 1) {
#ifdef NIMBLE_HAVE_MPI
    int global_max_contact_entity_id = max_contact_entity_id;
    MPI_Allreduce(&max_contact_entity_id, &global_max_contact_entity_id, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    max_contact_entity_id = global_max_contact_entity_id;
#endif
  }

  std::vector<int> node_global_id;
  std::vector<double> node_x;
  std::vector<double> node_y;
  std::vector<double> node_z;
  std::vector<int> elem_global_id;
  std::vector<int> block_ids;
  std::map<int, std::string> block_names;
  std::map<int, std::vector<int> > block_elem_global_ids;
  std::map<int, int> block_num_nodes_per_elem;
  std::map<int, std::vector<int> > block_elem_connectivity;

  // first block contains the contact faces
  int block_id = 1;
  block_ids.push_back(block_id);
  block_names[block_id] = "contact_faces";
  block_elem_global_ids[block_id] = std::vector<int>();
  block_num_nodes_per_elem[block_id] = 3;
  block_elem_connectivity[block_id] = std::vector<int>();

  int node_index(0);
  for (int i_face=0 ; i_face<contact_faces_h_.extent(0) ; i_face++) {
    ContactEntity const & face = contact_faces_h_(i_face);
    int contact_entity_global_id = face.contact_entity_global_id_;
    node_global_id.push_back(3 * contact_entity_global_id + max_contact_entity_id + 9);
    node_x.push_back(face.coord_1_x_);
    node_y.push_back(face.coord_1_y_);
    node_z.push_back(face.coord_1_z_);
    block_elem_connectivity[block_id].push_back(node_index++);
    node_global_id.push_back(3 * contact_entity_global_id + max_contact_entity_id + 10);
    node_x.push_back(face.coord_2_x_);
    node_y.push_back(face.coord_2_y_);
    node_z.push_back(face.coord_2_z_);
    block_elem_connectivity[block_id].push_back(node_index++);
    node_global_id.push_back(3 * contact_entity_global_id + max_contact_entity_id + 11);
    node_x.push_back(face.coord_3_x_);
    node_y.push_back(face.coord_3_y_);
    node_z.push_back(face.coord_3_z_);
    block_elem_connectivity[block_id].push_back(node_index++);
    elem_global_id.push_back(contact_entity_global_id);
  }

  // second block contains the contact nodes
  block_id = 2;
  block_ids.push_back(block_id);
  block_names[block_id] = "contact_nodes";
  block_elem_global_ids[block_id] = std::vector<int>();
  block_num_nodes_per_elem[block_id] = 1;
  block_elem_connectivity[block_id] = std::vector<int>();

  for (int i_node=0 ; i_node<contact_nodes_h_.extent(0) ; i_node++) {
    ContactEntity const & node = contact_nodes_h_(i_node);
    int contact_entity_global_id = node.contact_entity_global_id_;
    node_global_id.push_back(contact_entity_global_id);
    node_x.push_back(node.coord_1_x_);
    node_y.push_back(node.coord_1_y_);
    node_z.push_back(node.coord_1_z_);
    block_elem_connectivity[block_id].push_back(node_index++);
    elem_global_id.push_back(contact_entity_global_id);
  }

  // third block is the bounding box for this mpi rank
  /**
  block_id = 3;
  block_ids.push_back(block_id);
  block_names[block_id] = "contact_mpi_rank_bounding_box";
  block_elem_global_ids[block_id] = std::vector<int>();
  block_num_nodes_per_elem[block_id] = 8;
  block_elem_connectivity[block_id] = std::vector<int>();

  double x_min, x_max, y_min, y_max, z_min, z_max;
  BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);
  node_global_id.push_back(3 * max_contact_entity_id + 1);
  node_x.push_back(x_min); node_y.push_back(y_min); node_z.push_back(z_max);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 2);
  node_x.push_back(x_max); node_y.push_back(y_min); node_z.push_back(z_max);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 3);
  node_x.push_back(x_max); node_y.push_back(y_min); node_z.push_back(z_min);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 4);
  node_x.push_back(x_min); node_y.push_back(y_min); node_z.push_back(z_min);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 5);
  node_x.push_back(x_min); node_y.push_back(y_max); node_z.push_back(z_max);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 6);
  node_x.push_back(x_max); node_y.push_back(y_max); node_z.push_back(z_max);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 7);
  node_x.push_back(x_max); node_y.push_back(y_max); node_z.push_back(z_min);
  block_elem_connectivity[block_id].push_back(node_index++);
  node_global_id.push_back(3 * max_contact_entity_id + 8);
  node_x.push_back(x_min); node_y.push_back(y_max); node_z.push_back(z_min);
  block_elem_connectivity[block_id].push_back(node_index++);
  elem_global_id.push_back(max_contact_entity_id + 1);

  // store the model coordinate bounding box
  contact_visualization_model_coord_bounding_box_[0] = x_min;
  contact_visualization_model_coord_bounding_box_[1] = x_max;
  contact_visualization_model_coord_bounding_box_[2] = y_min;
  contact_visualization_model_coord_bounding_box_[3] = y_max;
  contact_visualization_model_coord_bounding_box_[4] = z_min;
  contact_visualization_model_coord_bounding_box_[5] = z_max;
  **/
  genesis_mesh_for_contact_visualization_.Initialize("contact_visualization",
                                                     node_global_id,
                                                     node_x,
                                                     node_y,
                                                     node_z,
                                                     elem_global_id,
                                                     block_ids,
                                                     block_names,
                                                     block_elem_global_ids,
                                                     block_num_nodes_per_elem,
                                                     block_elem_connectivity);

  exodus_output_for_contact_visualization_.Initialize(contact_visualization_exodus_file_name,
                                                      genesis_mesh_for_contact_visualization_);

  std::vector<std::string> global_data_labels;
  std::vector<std::string> node_data_labels_for_output;
  node_data_labels_for_output.push_back("displacement_x");
  node_data_labels_for_output.push_back("displacement_y");
  node_data_labels_for_output.push_back("displacement_z");
  std::map<int, std::vector<std::string> > elem_data_labels_for_output;
  std::map<int, std::vector<std::string> > derived_elem_data_labels;
  for (auto & block_id_ : block_ids) {
    elem_data_labels_for_output[block_id_] = std::vector<std::string>();
    derived_elem_data_labels[block_id_] = std::vector<std::string>();
  }
  exodus_output_for_contact_visualization_.InitializeDatabase(genesis_mesh_for_contact_visualization_,
                                                              global_data_labels,
                                                              node_data_labels_for_output,
                                                              elem_data_labels_for_output,
                                                              derived_elem_data_labels);
}

void
KokkosContactManager::ContactVisualizationWriteStep(double time_current) {

  // copy contact entities from host to device
  Kokkos::deep_copy(contact_nodes_h_, contact_nodes_d_);
  Kokkos::deep_copy(contact_faces_h_, contact_faces_d_);

  std::vector<double> global_data;
  std::vector< std::vector<double> > node_data_for_output(3);
  std::map<int, std::vector<std::string> > elem_data_labels_for_output;
  std::map<int, std::vector< std::vector<double> > > elem_data_for_output;
  std::map<int, std::vector<std::string> > derived_elem_data_labels;
  std::map<int, std::vector< std::vector<double> > > derived_elem_data;

  std::vector<int> const & block_ids = genesis_mesh_for_contact_visualization_.GetBlockIds();
  for (auto & block_id : block_ids) {
    elem_data_labels_for_output[block_id] = std::vector<std::string>();
    derived_elem_data_labels[block_id] = std::vector<std::string>();
  }

  // node_data_for_output contains displacement_x, displacement_y, displacement_z
  int num_nodes = genesis_mesh_for_contact_visualization_.GetNumNodes();
  node_data_for_output[0].resize(num_nodes);
  node_data_for_output[1].resize(num_nodes);
  node_data_for_output[2].resize(num_nodes);
  const double * model_coord_x = genesis_mesh_for_contact_visualization_.GetCoordinatesX();
  const double * model_coord_y = genesis_mesh_for_contact_visualization_.GetCoordinatesY();
  const double * model_coord_z = genesis_mesh_for_contact_visualization_.GetCoordinatesZ();

  int node_index(0);
  for (int i_face=0 ; i_face<contact_faces_h_.extent(0) ; i_face++) {
    ContactEntity const & face = contact_faces_h_[i_face];
    node_data_for_output[0][node_index] = face.coord_1_x_ - model_coord_x[node_index];
    node_data_for_output[1][node_index] = face.coord_1_y_ - model_coord_y[node_index];
    node_data_for_output[2][node_index] = face.coord_1_z_ - model_coord_z[node_index];
    node_index += 1;
    node_data_for_output[0][node_index] = face.coord_2_x_ - model_coord_x[node_index];
    node_data_for_output[1][node_index] = face.coord_2_y_ - model_coord_y[node_index];
    node_data_for_output[2][node_index] = face.coord_2_z_ - model_coord_z[node_index];
    node_index += 1;
    node_data_for_output[0][node_index] = face.coord_3_x_ - model_coord_x[node_index];
    node_data_for_output[1][node_index] = face.coord_3_y_ - model_coord_y[node_index];
    node_data_for_output[2][node_index] = face.coord_3_z_ - model_coord_z[node_index];
    node_index += 1;
  }
  for (int i_node=0 ; i_node<contact_nodes_h_.extent(0) ; i_node++) {
    ContactEntity const & node = contact_nodes_h_(i_node);
    node_data_for_output[0][node_index] = node.coord_1_x_ - model_coord_x[node_index];
    node_data_for_output[1][node_index] = node.coord_1_y_ - model_coord_y[node_index];
    node_data_for_output[2][node_index] = node.coord_1_z_ - model_coord_z[node_index];
    node_index += 1;
  }
  /*
  double x_min_model_coord = contact_visualization_model_coord_bounding_box_[0];
  double x_max_model_coord = contact_visualization_model_coord_bounding_box_[1];
  double y_min_model_coord = contact_visualization_model_coord_bounding_box_[2];
  double y_max_model_coord = contact_visualization_model_coord_bounding_box_[3];
  double z_min_model_coord = contact_visualization_model_coord_bounding_box_[4];
  double z_max_model_coord = contact_visualization_model_coord_bounding_box_[5];
  double x_min, x_max, y_min, y_max, z_min, z_max;
  BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);
  // node_x.push_back(x_min); node_y.push_back(y_min); node_z.push_back(z_max);
  node_data_for_output[0][node_index] = x_min - x_min_model_coord;
  node_data_for_output[1][node_index] = y_min - y_min_model_coord;
  node_data_for_output[2][node_index] = z_min - z_min_model_coord;
  node_index += 1;
  // node_x.push_back(x_max); node_y.push_back(y_min); node_z.push_back(z_max);
  node_data_for_output[0][node_index] = x_max - x_max_model_coord;
  node_data_for_output[1][node_index] = y_min - y_min_model_coord;
  node_data_for_output[2][node_index] = z_max - z_max_model_coord;
  node_index += 1;
  // node_x.push_back(x_max); node_y.push_back(y_min); node_z.push_back(z_min);
  node_data_for_output[0][node_index] = x_max - x_max_model_coord;
  node_data_for_output[1][node_index] = y_min - y_min_model_coord;
  node_data_for_output[2][node_index] = z_min - z_min_model_coord;
  node_index += 1;
  // node_x.push_back(x_min); node_y.push_back(y_min); node_z.push_back(z_min);
  node_data_for_output[0][node_index] = x_min - x_min_model_coord;
  node_data_for_output[1][node_index] = y_min - y_min_model_coord;
  node_data_for_output[2][node_index] = z_min - z_min_model_coord;
  node_index += 1;
  // node_x.push_back(x_min); node_y.push_back(y_max); node_z.push_back(z_max);
  node_data_for_output[0][node_index] = x_min - x_min_model_coord;
  node_data_for_output[1][node_index] = y_max - y_max_model_coord;
  node_data_for_output[2][node_index] = z_max - z_max_model_coord;
  node_index += 1;
  // node_x.push_back(x_max); node_y.push_back(y_max); node_z.push_back(z_max);
  node_data_for_output[0][node_index] = x_max - x_max_model_coord;
  node_data_for_output[1][node_index] = y_max - y_max_model_coord;
  node_data_for_output[2][node_index] = z_max - z_max_model_coord;
  node_index += 1;
  // node_x.push_back(x_max); node_y.push_back(y_max); node_z.push_back(z_min);
  node_data_for_output[0][node_index] = x_max - x_max_model_coord;
  node_data_for_output[1][node_index] = y_max - y_max_model_coord;
  node_data_for_output[2][node_index] = z_min - z_min_model_coord;
  node_index += 1;
  // node_x.push_back(x_min); node_y.push_back(y_max); node_z.push_back(z_min);
  node_data_for_output[0][node_index] = x_min - x_min_model_coord;
  node_data_for_output[1][node_index] = y_max - y_max_model_coord;
  node_data_for_output[2][node_index] = z_min - z_min_model_coord;
  node_index += 1;
  */
  exodus_output_for_contact_visualization_.WriteStep(time_current,
                                                     global_data,
                                                     node_data_for_output,
                                                     elem_data_labels_for_output,
                                                     elem_data_for_output,
                                                     derived_elem_data_labels,
                                                     derived_elem_data);
}

void KokkosContactManager::InitializeContactData(GenesisMesh const & mesh,
                                                 std::vector< std::vector<int> > const & master_skin_faces,
                                                 std::vector<int> const & master_skin_entity_ids,
                                                 std::vector<int> const & slave_node_ids,
                                                 std::vector<int> const & slave_node_entity_ids,
                                                 std::map<int, double> const & slave_node_char_lens
) {

  ContactManager::InitializeContactData(mesh, master_skin_faces, master_skin_entity_ids,
                                        slave_node_ids, slave_node_entity_ids,
                                        slave_node_char_lens);

  const double* coord_x = mesh.GetCoordinatesX();
  const double* coord_y = mesh.GetCoordinatesY();
  const double* coord_z = mesh.GetCoordinatesZ();

  const auto array_len = 3*node_ids_.size();

  nimble_kokkos::HostIntegerArrayView node_ids_h("contact_node_ids_h", node_ids_.size());
  for (unsigned int i_node=0 ; i_node<node_ids_.size() ; i_node++) {
    node_ids_h[i_node] = node_ids_[i_node];
  }

  nimble_kokkos::HostScalarNodeView model_coord_h("contact_model_coord_h", array_len);
  for (unsigned int i_node=0 ; i_node<node_ids_.size() ; i_node++) {
    model_coord_h[3*i_node]   = coord_x[node_ids_[i_node]];
    model_coord_h[3*i_node+1] = coord_y[node_ids_[i_node]];
    model_coord_h[3*i_node+2] = coord_z[node_ids_[i_node]];
  }

  Kokkos::resize(node_ids_d_, node_ids_.size());
  Kokkos::resize(model_coord_d_, array_len);
  Kokkos::resize(coord_d_, array_len);
  Kokkos::resize(force_d_, array_len);

  Kokkos::deep_copy(node_ids_d_, node_ids_h);
  Kokkos::deep_copy(model_coord_d_, model_coord_h);
  Kokkos::deep_copy(coord_d_, model_coord_h);
  Kokkos::deep_copy(force_d_, 0.0);

  Kokkos::resize(contact_nodes_h_, slave_node_ids.size());
  Kokkos::resize(contact_faces_h_, 4*master_skin_faces.size());
  CreateContactNodesAndFaces< nimble_kokkos::HostContactEntityArrayView >(
       master_skin_faces, master_skin_entity_ids,
       slave_node_ids, slave_node_entity_ids, slave_node_char_lens,
      contact_nodes_h_, contact_faces_h_
      );

  Kokkos::resize(contact_nodes_d_, slave_node_ids.size());
  Kokkos::resize(contact_faces_d_, 4*master_skin_faces.size());
  Kokkos::deep_copy(contact_nodes_d_, contact_nodes_h_);
  Kokkos::deep_copy(contact_faces_d_, contact_faces_h_);

}

void KokkosContactManager::ComputeContactForce(int step, bool debug_output) {

  if (penalty_parameter_ <= 0.0) {
    throw std::logic_error("\nError in ComputeContactForce(), invalid penalty_parameter.\n");
  }

  double background_grid_cell_size = BoundingBoxAverageCharacteristicLengthOverAllRanks();
  // std::cout << "DEBUGGING background_grid_cell_size " << background_grid_cell_size << std::endl;
  // DJL PARALLEL CONTACT  processCollision(coord_.data(), coord_.size(), background_grid_cell_size);

  // kokkos_view::extent(), accessor kokkos_view::(i, 0) x coordinte of ith entry in list

  // ANTONIO, HERE IS THE COMMENTED OUT LINE
  // std::vector<int> exchange_members = getExchangeMembers(coord_d_,
  //                                                        background_grid_cell_size,
  //                                                        MPI_COMM_WORLD,
  //                                                        0,
  //                                                        1,
  //                                                        2);

  // DJL
  // entities are stored in contact_nodes_d, contact_nodes_h
  // 1) box box search
  // 2) node-face projection
  // 3) culling
  // 4) enforcement

  contact_interface->ComputeContact(contact_nodes_d_, contact_faces_d_, force_d_);

}

} // namespace nimble

#endif


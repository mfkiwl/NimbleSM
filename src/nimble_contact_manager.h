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

#ifndef NIMBLE_CONTACT_H
#define NIMBLE_CONTACT_H

#include <vector>
#include <map>
#include <memory>
#include <float.h>
#include <cmath>

#include "nimble_contact_entity.h"
#include "nimble_contact_interface.h"
#include "nimble_genesis_mesh.h"
#include "nimble_exodus_output.h"
#include "nimble.mpi.utils.h"

#ifdef NIMBLE_HAVE_BVH
  #include <bvh/kdop.hpp>
  #include <bvh/tree.hpp>
  #include <bvh/patch.hpp>
  #include <bvh/perf/instrument.hpp>
#ifdef BVH_ENABLE_VT
  #include <bvh/vt/collection.hpp>
  #include <bvh/vt/helpers.hpp>
  #include <bvh/vt/collision_world.hpp>
#endif
#endif

namespace nimble {

class ContactInterface;

  class ContactManager {

  public:

    typedef enum ProjectionType {
      UNKNOWN=0,
      NODE_OR_EDGE=1,
      FACE=2
    } PROJECTION_TYPE;

    explicit ContactManager( std::shared_ptr<ContactInterface> interface );

    virtual ~ContactManager() {}

    bool ContactEnabled() { return contact_enabled_; }

    void SkinBlocks(GenesisMesh const & mesh,
                    std::vector<int> const & block_ids,
                    int entity_id_offset,
                    std::vector< std::vector<int> > & skin_faces,
                    std::vector<int> & entity_ids);

    void RemoveInternalSkinFaces(GenesisMesh const & mesh,
                                 std::vector< std::vector<int> >& faces,
                                 std::vector<int>& entity_ids);

    void SetPenaltyParameter(double penalty_parameter) {
      penalty_parameter_ = penalty_parameter;
      contact_interface->SetUpPenaltyEnforcement(penalty_parameter_);
    }

    void CreateContactEntities(GenesisMesh const & mesh,
                               std::vector<int> const & master_block_ids,
                               std::vector<int> const & slave_block_ids) {
      nimble::MPIContainer mpi_container;
      CreateContactEntities(mesh, mpi_container, master_block_ids, slave_block_ids);
    }


    void CreateContactEntities(GenesisMesh const & mesh,
                               nimble::MPIContainer & mpi_container,
                               std::vector<int> const & master_block_ids,
                               std::vector<int> const & slave_block_ids);

    template <typename ArgT>
    void CreateContactNodesAndFaces(std::vector< std::vector<int> > const & master_skin_faces,
                                    std::vector<int> const & master_skin_entity_ids,
                                    std::vector<int> const & slave_node_ids,
                                    std::vector<int> const & slave_skin_entity_ids,
                                    std::map<int, double> const & slave_node_char_lens,
                                    ArgT& contact_nodes,
                                    ArgT& contact_faces) const ;

    virtual void BoundingBox(double& x_min,
                     double& x_max,
                     double& y_min,
                     double& y_max,
                     double& z_min,
                     double& z_max) const;

    double BoundingBoxAverageCharacteristicLengthOverAllRanks() const ;

    void ApplyDisplacements(const double * const displacement) {
      for (unsigned int i_node=0; i_node<node_ids_.size() ; i_node++) {
        int node_id = node_ids_[i_node];
        for (int i=0 ; i<3 ; i++) {
          coord_[3*i_node + i] = model_coord_[3*i_node + i] + displacement[3*node_id + i];
        }
      }
      for (unsigned int i_face=0 ; i_face<contact_faces_.size() ; i_face++) {
        contact_faces_[i_face].SetCoordinates(coord_.data());
      }
      for (unsigned int i_node=0 ; i_node<contact_nodes_.size() ; i_node++) {
        contact_nodes_[i_node].SetCoordinates(coord_.data());
      }
    }

    void GetForces(double * const contact_force) {
      for (unsigned int i_node=0; i_node<node_ids_.size() ; i_node++) {
        int node_id = node_ids_[i_node];
        for (int i=0 ; i<3 ; i++) {
          contact_force[3*node_id + i] = force_[3*i_node + i];
        }
      }
    }

    virtual void ComputeContactForce(int step, bool debug_output);

    void BruteForceBoxIntersectionSearch(std::vector<ContactEntity> const & nodes,
                                         std::vector<ContactEntity> const & triangles);

    void ClosestPointProjection(std::vector<ContactEntity> const & nodes,
                                std::vector<ContactEntity> const & triangles,
                                std::vector<ContactEntity::vertex>& closest_points,
                                std::vector<PROJECTION_TYPE>& projection_types);

    virtual void InitializeContactVisualization(std::string const & contact_visualization_exodus_file_name);

    virtual void ContactVisualizationWriteStep(double time_current);

  protected:

    virtual void InitializeContactData(GenesisMesh const & mesh,
                                       std::vector< std::vector<int> > const & master_skin_faces,
                                       std::vector<int> const & master_skin_entity_ids,
                                       std::vector<int> const & slave_node_ids,
                                       std::vector<int> const & slave_node_entity_ids,
                                       std::map<int, double> const & slave_node_char_lens)
    {
      contact_nodes_.resize(slave_node_ids.size());
      contact_faces_.resize(4*master_skin_faces.size());
      CreateContactNodesAndFaces< std::vector<ContactEntity> >(master_skin_faces, master_skin_entity_ids,
                                                             slave_node_ids, slave_node_entity_ids, slave_node_char_lens,
                                                             contact_nodes_, contact_faces_);
    }

    void InitializeContactVisualizationImpl(std::string const & contact_visualization_exodus_file_name,
                                            nimble::GenesisMesh &mesh,
                                            nimble::ExodusOutput &out,
                                            ContactEntity *faces, std::size_t nfaces,
                                            ContactEntity *nodes, std::size_t nnodes);

    void WriteVisualizationData( double t, nimble::GenesisMesh &mesh,
                                 nimble::ExodusOutput &out,
        ContactEntity *faces, std::size_t nfaces,
        ContactEntity *nodes, std::size_t nnodes );

    //--- Variables

    int rank_ = 0;
    int num_ranks_ = 1;

    bool contact_enabled_ = false;
    double penalty_parameter_;

    std::vector<int> node_ids_;
    std::vector<double> model_coord_;
    std::vector<double> coord_;
    std::vector<double> force_;
    std::vector<ContactEntity> contact_faces_;
    std::vector<ContactEntity> contact_nodes_;

    double contact_visualization_model_coord_bounding_box_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    nimble::GenesisMesh genesis_mesh_for_contact_visualization_;
    nimble::ExodusOutput exodus_output_for_contact_visualization_;

    std::shared_ptr<ContactInterface> contact_interface;

};


//-------------------------------------------
//--- Implementation of template functions
//-------------------------------------------


template <typename ArgT>
void ContactManager::CreateContactNodesAndFaces(std::vector< std::vector<int> > const & master_skin_faces,
                                                std::vector<int> const & master_skin_entity_ids,
                                                std::vector<int> const & slave_node_ids,
                                                std::vector<int> const & slave_node_entity_ids,
                                                std::map<int, double> const & slave_node_char_lens,
                                                ArgT& contact_nodes,
                                                ArgT& contact_faces) const {

  int index = 0;

  // convert master faces to trangular facets
  for (unsigned int i_face=0 ; i_face < master_skin_faces.size() ; i_face++) {

    auto face = master_skin_faces[i_face];

    int num_nodes_in_face = static_cast<int>(face.size());
    if (num_nodes_in_face != 4) {
      throw std::logic_error("\nError in ContactManager::CreateContactNodesAndFaces(), invalid number of face nodes.\n");
    }

    // determine a characteristic length based on max edge length
    double max_edge_length = std::numeric_limits<double>::lowest();
    for (int i=0 ; i<num_nodes_in_face ; ++i) {
      int node_id_1 = face[i];
      int node_id_2 = face[0];
      if (i+1 < num_nodes_in_face) {
        node_id_2 = face[i+1];
      }
      double edge_length = sqrt( (coord_[3*node_id_2  ] - coord_[3*node_id_1  ])*(coord_[3*node_id_2  ] - coord_[3*node_id_1  ]) +
                                 (coord_[3*node_id_2+1] - coord_[3*node_id_1+1])*(coord_[3*node_id_2+1] - coord_[3*node_id_1+1]) +
                                 (coord_[3*node_id_2+2] - coord_[3*node_id_1+2])*(coord_[3*node_id_2+2] - coord_[3*node_id_1+2]) );
      if (edge_length > max_edge_length) {
        max_edge_length = edge_length;
      }
    }
    double characteristic_length = max_edge_length;

    // create a node at the barycenter of the face
    double fictitious_node[3] = {0.0, 0.0, 0.0};
    for (int i=0 ; i<num_nodes_in_face ; ++i) {
      int node_id = face[i];
      for (int j=0 ; j<3 ; j++) {
        fictitious_node[j] += coord_[3*node_id+j];
      }
    }
    for (int j=0 ; j<3 ; j++) {
      fictitious_node[j] /= num_nodes_in_face;
    }

    // Create a map for transfering displacements and contact forces from the nodes on the
    // triangle patch to the contact manager data structures.  There is a 1-to-1 transfer for the two real nodes,
    // and for the fictitious node the mapping applies an equal fraction of the displacement/force
    // at the fictitious node to each for four real nodes in the original mesh face
    int node_ids_for_fictitious_node[4];
    for (int i=0 ; i<4 ; i++){
      node_ids_for_fictitious_node[i] = face[i];
    }

    double model_coord[9];
    int node_id_1, node_id_2, entity_id;

    // triangle node_0, node_1, fictitious_node
    node_id_1 = face[0];
    node_id_2 = face[1];
    for (int i=0 ; i<3 ; ++i) {
      model_coord[i] = coord_[3*node_id_1+i];
      model_coord[3+i] = coord_[3*node_id_2+i];
    }
    model_coord[6] = fictitious_node[0];
    model_coord[7] = fictitious_node[1];
    model_coord[8] = fictitious_node[2];
    entity_id = master_skin_entity_ids[i_face];
    entity_id |= 0; // triangle ordinal
    contact_faces[index++] = ContactEntity(ContactEntity::TRIANGLE,
                                           entity_id,
                                           model_coord,
                                           characteristic_length,
                                           node_id_1,
                                           node_id_2,
                                           node_ids_for_fictitious_node);

    // triangle node_1, node_2, fictitious_node
    node_id_1 = face[1];
    node_id_2 = face[2];
    for (int i=0 ; i<3 ; ++i) {
      model_coord[i] = coord_[3*node_id_1+i];
      model_coord[3+i] = coord_[3*node_id_2+i];
    }
    model_coord[6] = fictitious_node[0];
    model_coord[7] = fictitious_node[1];
    model_coord[8] = fictitious_node[2];
    entity_id = master_skin_entity_ids[i_face];
    entity_id |= 1; // triangle ordinal
    contact_faces[index++] = ContactEntity(ContactEntity::TRIANGLE,
                                           entity_id,
                                           model_coord,
                                           characteristic_length,
                                           node_id_1,
                                           node_id_2,
                                           node_ids_for_fictitious_node);

    // triangle node_2, node_3, fictitious_node
    node_id_1 = face[2];
    node_id_2 = face[3];
    for (int i=0 ; i<3 ; ++i) {
      model_coord[i] = coord_[3*node_id_1+i];
      model_coord[3+i] = coord_[3*node_id_2+i];
    }
    model_coord[6] = fictitious_node[0];
    model_coord[7] = fictitious_node[1];
    model_coord[8] = fictitious_node[2];
    entity_id = master_skin_entity_ids[i_face];
    entity_id |= 2; // triangle ordinal
    contact_faces[index++] = ContactEntity(ContactEntity::TRIANGLE,
                                           entity_id,
                                           model_coord,
                                           characteristic_length,
                                           node_id_1,
                                           node_id_2,
                                           node_ids_for_fictitious_node);

    // triangle node_3, node_0, fictitious_node
    node_id_1 = face[3];
    node_id_2 = face[0];
    for (int i=0 ; i<3 ; ++i) {
      model_coord[i] = coord_[3*node_id_1+i];
      model_coord[3+i] = coord_[3*node_id_2+i];
    }
    model_coord[6] = fictitious_node[0];
    model_coord[7] = fictitious_node[1];
    model_coord[8] = fictitious_node[2];
    entity_id = master_skin_entity_ids[i_face];
    entity_id |= 3; // triangle ordinal
    contact_faces[index++] = ContactEntity(ContactEntity::TRIANGLE,
                                           entity_id,
                                           model_coord,
                                           characteristic_length,
                                           node_id_1,
                                           node_id_2,
                                           node_ids_for_fictitious_node);
  }

  // Slave node entities
  for (unsigned int i_node=0 ; i_node < slave_node_ids.size() ; ++i_node) {
    int node_id = slave_node_ids.at(i_node);
    int entity_id = slave_node_entity_ids.at(i_node);
    double characteristic_length = slave_node_char_lens.at(node_id);
    double model_coord[3];
    for (int i=0 ; i<3 ; ++i) {
      model_coord[i] = coord_[3*node_id+i];
    }
    contact_nodes[i_node] = ContactEntity(ContactEntity::NODE,
                                          entity_id,
                                          model_coord,
                                          characteristic_length,
                                          node_id);
  }
}

} // namespace nimble

#endif // NIMBLE_MATERIAL_H

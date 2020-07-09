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

#ifndef NIMBLE_KOKKOS_CONTACT_MANAGER_H
#define NIMBLE_KOKKOS_CONTACT_MANAGER_H

#include "../nimble_contact_manager.h"

#ifdef NIMBLE_HAVE_KOKKOS
#include "nimble_kokkos_defs.h"
#include "nimble_kokkos_contact_defs.h"
#endif

namespace nimble {
  class KokkosContactManager : public ContactManager
  {
  public:

    KokkosContactManager(std::shared_ptr<ContactInterface> interface)
        : ContactManager(interface)
    {
#ifdef NIMBLE_HAVE_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
      MPI_Comm_size(MPI_COMM_WORLD, &num_ranks_);
#endif
    }

#ifdef NIMBLE_HAVE_KOKKOS

  void BoundingBox(double& x_min,
                           double& x_max,
                           double& y_min,
                           double& y_max,
                           double& z_min,
                           double& z_max) const override;

  void InitializeContactVisualization(std::string const & contact_visualization_exodus_file_name) override;

  void ContactVisualizationWriteStep(double time_current) override;

  void ComputeContactForce(int step, bool debug_output) override;

  void InitializeContactData(GenesisMesh const & mesh,
                             std::vector< std::vector<int> > const & master_skin_faces,
                             std::vector<int> const & master_skin_entity_ids,
                             std::vector<int> const & slave_node_ids,
                             std::vector<int> const & slave_node_entity_ids,
                             std::map<int, double> const & slave_node_char_lens) override;

  void ApplyDisplacements(nimble_kokkos::DeviceVectorNodeView displacement_d) {

    int num_nodes_in_contact_manager = node_ids_d_.extent(0);
    int num_contact_node_entities = contact_nodes_d_.extent(0);
    int num_contact_face_entities = contact_faces_d_.extent(0);

    // circumvent lambda *this glitch
    nimble_kokkos::DeviceIntegerArrayView node_ids = node_ids_d_;
    nimble_kokkos::DeviceScalarNodeView model_coord = model_coord_d_;
    nimble_kokkos::DeviceScalarNodeView coord = coord_d_;
    nimble_kokkos::DeviceContactEntityArrayView contact_nodes = contact_nodes_d_;
    nimble_kokkos::DeviceContactEntityArrayView contact_faces = contact_faces_d_;

    Kokkos::parallel_for("ContactManager::ApplyDisplacements set coord_d_ vector",
                         num_nodes_in_contact_manager,
                         KOKKOS_LAMBDA(const int i) {
                           int node_id = node_ids(i);
                           coord(3*i)   = model_coord(3*i)   + displacement_d(node_id, 0);
                           coord(3*i+1) = model_coord(3*i+1) + displacement_d(node_id, 1);
                           coord(3*i+2) = model_coord(3*i+2) + displacement_d(node_id, 2);
                         });

    Kokkos::parallel_for("ContactManager::ApplyDisplacements set contact node entity displacements",
                         num_contact_node_entities,
                         KOKKOS_LAMBDA(const int i_node) {
                           contact_nodes(i_node).SetCoordinates(coord);
                         });

    Kokkos::parallel_for("ContactManager::ApplyDisplacements set contact face entity displacements",
                         num_contact_face_entities,
                         KOKKOS_LAMBDA(const int i_face) {
                           contact_faces(i_face).SetCoordinates(coord);
                         });

  }

  /// \brief Get the forces
  /// \param[in] contact_force_d
  void GetForces(nimble_kokkos::DeviceVectorNodeView contact_force_d) {

    int num_nodes_in_contact_manager = node_ids_d_.extent(0);

    // circumvent lambda *this glitch
    nimble_kokkos::DeviceIntegerArrayView node_ids = node_ids_d_;
    nimble_kokkos::DeviceScalarNodeView force = force_d_;

    Kokkos::parallel_for("ContactManager::GetForces",
                         num_nodes_in_contact_manager,
                         KOKKOS_LAMBDA(const int i) {
                           int node_id = node_ids(i);
                           contact_force_d(node_id, 0) = force(3*i);
                           contact_force_d(node_id, 1) = force(3*i+1);
                           contact_force_d(node_id, 2) = force(3*i+2);
                         });
  }

protected:

  nimble_kokkos::DeviceIntegerArrayView node_ids_d_ = nimble_kokkos::DeviceIntegerArrayView("contact node_ids_d", 1);
  nimble_kokkos::DeviceScalarNodeView model_coord_d_ = nimble_kokkos::DeviceScalarNodeView("contact model_coord_d", 1);
  nimble_kokkos::DeviceScalarNodeView coord_d_ = nimble_kokkos::DeviceScalarNodeView("contact coord_d", 1);
  nimble_kokkos::DeviceScalarNodeView force_d_ = nimble_kokkos::DeviceScalarNodeView("contact force_d", 1);

  nimble_kokkos::DeviceContactEntityArrayView contact_faces_d_ = nimble_kokkos::DeviceContactEntityArrayView("contact_faces_d", 1);
  nimble_kokkos::DeviceContactEntityArrayView contact_nodes_d_ = nimble_kokkos::DeviceContactEntityArrayView("contact_nodes_d", 1);

  // TODO remove this once enforcement is on device
  nimble_kokkos::HostContactEntityArrayView contact_faces_h_ = nimble_kokkos::HostContactEntityArrayView("contact_faces_h", 1);
  nimble_kokkos::HostContactEntityArrayView contact_nodes_h_ = nimble_kokkos::HostContactEntityArrayView("contact_nodes_h", 1);

#endif

  };
}

#endif  // NIMBLE_KOKKOS_CONTACT_MANAGER_H

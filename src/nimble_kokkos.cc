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

#include "nimble_kokkos.h"
#include "nimble.quanta.stopwatch.h"
#include "nimble_boundary_condition_manager.h"
#include "nimble_contact_interface.h"
#include "nimble_contact_manager.h"
#include "nimble_data_manager.h"
#include "nimble_exodus_output.h"
#include "nimble_exodus_output_manager.h"
#include "nimble_parser.h"
#include "nimble_timer.h"
#include "nimble_timing_utils.h"
#include "nimble_version.h"

#include "nimble_kokkos_block.h"
#include "nimble_kokkos_defs.h"
#include "nimble_kokkos_material_factory.h"
#include "nimble_kokkos_profiling.h"
#include "nimble_kokkos_block_material_interface.h"
#include "nimble_kokkos_block_material_interface_factory.h"

#include <cassert>

#ifdef NIMBLE_HAVE_ARBORX
  #ifdef NIMBLE_HAVE_MPI
    #include "contact/parallel/arborx_parallel_contact_manager.h"
  #endif
    #include "contact/serial/arborx_serial_contact_manager.h"
#endif


#include <iostream>

namespace nimble {

namespace details_kokkos {

int ExplicitTimeIntegrator(nimble::Parser & parser,
                           nimble::GenesisMesh & mesh,
                           nimble::DataManager & data_manager,
                           nimble::BoundaryConditionManager &bc_manager,
                           nimble::ExodusOutput & exodus_output,
                           std::shared_ptr<nimble::ContactInterface> contact_interface,
                           std::shared_ptr<nimble_kokkos::BlockMaterialInterfaceFactory> block_material_interface_factory,
                           int num_ranks,
                           int my_rank
);

}


NimbleKokkosInitData NimbleKokkosInitializeAndGetInput(int argc, char* argv[]) {

#ifdef NIMBLE_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

#ifdef NIMBLE_HAVE_KOKKOS
  Kokkos::initialize(argc, argv);
#endif

  NimbleKokkosInitData init_data;

#ifdef NIMBLE_HAVE_MPI
  int mpi_err;
  mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &init_data.num_mpi_ranks);
  if (mpi_err != MPI_SUCCESS) {
    throw std::logic_error("\nError:  MPI_Comm_size() returned nonzero error code.\n");
  }
  mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &init_data.my_mpi_rank);
  if (mpi_err != MPI_SUCCESS) {
    throw std::logic_error("\nError:  MPI_Comm_rank() returned nonzero error code.\n");
  }
#else
  init_data.num_mpi_ranks = 1;
  init_data.my_mpi_rank = 0;
#endif
  
  // Banner
  if (init_data.my_mpi_rank == 0) {
    std::string nimble_have_kokkos("false");
#ifdef NIMBLE_HAVE_KOKKOS
    nimble_have_kokkos = "true";
#endif
    std::string kokkos_enable_cuda("false");
#ifdef KOKKOS_ENABLE_CUDA
    kokkos_enable_cuda = "true";
#endif
    std::string kokkos_enable_cuda_uvm("false");
#ifdef KOKKOS_ENABLE_CUDA_UVM
    kokkos_enable_cuda_uvm = "true";
#endif

    std::cout << "\n-- NimbleSM_Kokkos" << std::endl;
    std::cout << "-- version " << nimble::NimbleVersion() << "\n" << std::endl;
    if (argc != 2) {
      std::cout << "Usage:  mpirun -np NP NimbleSM_Kokkos <input_deck.in>\n" << std::endl;
      Kokkos::finalize();
#ifdef NIMBLE_HAVE_MPI
      MPI_Finalize();
#endif
      exit(1);
    }
    std::cout << "NimbleSM_Kokkos initialized on " << init_data.num_mpi_ranks << " mpi rank(s)." << std::endl;

    std::cout << "\nKokkos configuration:" << std::endl;
    std::cout << "  NIMBLE_HAVE_KOKKOS               " << nimble_have_kokkos << std::endl;
#ifdef NIMBLE_HAVE_ARBORX
    std::cout << "  NIMBLE_HAVE_ARBORX               true\n";
#endif
    std::cout << "  KOKKOS_ENABLE_CUDA               " << kokkos_enable_cuda << std::endl;
    std::cout << "  KOKKOS_ENABLE_CUDA_UVM           " << kokkos_enable_cuda_uvm << std::endl;
    std::cout << "  kokkos_host_execution_space      " << typeid(nimble_kokkos::kokkos_host_execution_space).name() << std::endl;
    std::cout << "  kokkos_host_mirror_memory_space  " << typeid(nimble_kokkos::kokkos_host_mirror_memory_space).name() << std::endl;
    std::cout << "  kokkos_host                      " << typeid(nimble_kokkos::kokkos_host).name() << std::endl;
    std::cout << "  kokkos_device_execution_space    " << typeid(nimble_kokkos::kokkos_device_execution_space).name() << std::endl;
    std::cout << "  kokkos_device_memory_space       " << typeid(nimble_kokkos::kokkos_device_memory_space).name() << std::endl;
    std::cout << "  kokkos_device                    " << typeid(nimble_kokkos::kokkos_device).name() << std::endl;
    std::cout << "  kokkos_layout                    " << typeid(nimble_kokkos::kokkos_layout).name() << std::endl;
    std::cout << std::endl;
  }

  init_data.input_deck_name = std::string(argv[1]);

  return init_data;
}

int NimbleKokkosFinalize(const NimbleKokkosInitData& init_data) {
  if (init_data.my_mpi_rank == 0) {
    std::cout << "\ncomplete.\n" << std::endl;
  }

#ifdef NIMBLE_HAVE_KOKKOS
  Kokkos::finalize();
#endif

#ifdef NIMBLE_HAVE_MPI
  return MPI_Finalize();
#else
  return 0;
#endif
}

void NimbleKokkosMain(std::shared_ptr<nimble_kokkos::MaterialFactory> material_factory,
                     std::shared_ptr<nimble::ContactInterface> contact_interface,
                     std::shared_ptr<nimble_kokkos::BlockMaterialInterfaceFactory> block_material_interface_factory,
                     std::shared_ptr<nimble::Parser> parser,
                     const NimbleKokkosInitData& init_data) {
  const int num_ranks = init_data.num_mpi_ranks;
  const int my_rank = init_data.my_mpi_rank;
  const std::string& input_deck_name = init_data.input_deck_name;

  //--- Define timers
  nimble_kokkos::ProfilingTimer watch_simulation;
  watch_simulation.push_region("Parse and read mesh");
  
  parser->Initialize(init_data.input_deck_name);

  // Read the mesh
  nimble::GenesisMesh mesh;
  nimble::GenesisMesh rve_mesh;
  {
    std::string genesis_file_name = nimble::IOFileName(parser->GenesisFileName(), "g", "", my_rank, num_ranks);
    std::string rve_genesis_file_name = nimble::IOFileName(parser->RVEGenesisFileName(), "g");
    mesh.ReadFile(genesis_file_name);
    if (rve_genesis_file_name != "none") {
      rve_mesh.ReadFile(rve_genesis_file_name);
    }
  }

  watch_simulation.pop_region_and_report_time();

#ifdef NIMBLE_HAVE_ARBORX
  std::string tag = "arborx";
#else
  std::string tag = "kokkos";
#endif
  std::string output_exodus_name = nimble::IOFileName(parser->ExodusFileName(), "e", tag, my_rank, num_ranks);
  int dim = mesh.GetDim();
  int num_nodes = static_cast<int>(mesh.GetNumNodes());

  if (my_rank == 0) {
    std::cout << "\n";
    if (num_ranks == 1) {
      std::cout << " Number of Nodes = " << num_nodes << "\n";
      std::cout << " Number of Elements = " << mesh.GetNumElements() << "\n";
    }
    std::cout << " Number of Global Blocks = " << mesh.GetNumGlobalBlocks() << "\n";
    std::cout << "\n";
    std::cout << " Number of Ranks         = " << num_ranks << "\n";
#ifdef _OPENMP
    std::cout << " Number of Threads       = " << omp_get_max_threads() << "\n";
#endif
    std::cout << "\n";
  }
  watch_simulation.push_region("Model data and field allocation");

#ifdef NIMBLE_HAVE_KOKKOS
  //--- UH This is redundant for the moment
  nimble::DataManager data_manager(true);
#endif
//  auto model_data = data_manager.GetMacroScaleData();
  auto model_data = dynamic_cast<nimble_kokkos::ModelData*>(data_manager.GetMacroScaleData().get());
  model_data->SetDimension(dim);

  // Global data
  std::vector<std::string> global_data_labels;

  model_data->AllocateNodeData(nimble::SCALAR, nimble::FieldID::LumpedMass, num_nodes);
  model_data->AllocateNodeData(nimble::VECTOR, nimble::FieldID::ReferenceCoordinate, num_nodes);
  model_data->AllocateNodeData(nimble::VECTOR, nimble::FieldID::Displacement, num_nodes);
  model_data->AllocateNodeData(nimble::VECTOR, nimble::FieldID::Velocity, num_nodes);
  model_data->AllocateNodeData(nimble::VECTOR, nimble::FieldID::Acceleration, num_nodes);
  model_data->AllocateNodeData(nimble::VECTOR, nimble::FieldID::InternalForce, num_nodes);
  model_data->AllocateNodeData(nimble::VECTOR, nimble::FieldID::ContactForce, num_nodes);

//  bool store_unrotated_stress(true);
  bool store_unrotated_stress = false;

  // Initialize the blocks
  model_data->InitializeBlocks(mesh, *parser, material_factory, store_unrotated_stress);

  // Initialize the initial- and boundary-condition manager
  std::map<int, std::string> const & node_set_names = mesh.GetNodeSetNames();
  std::map<int, std::vector<int> > const & node_sets = mesh.GetNodeSets();
  std::vector<std::string> const & bc_strings = parser->GetBoundaryConditionStrings();
  std::string const & time_integration_scheme = parser->TimeIntegrationScheme();
  nimble::BoundaryConditionManager bc;
  bc.Initialize(node_set_names, node_sets, bc_strings, dim, time_integration_scheme);

  // Initialize the output file
  model_data->SpecifyOutputFields(parser->GetOutputFieldString());

  auto node_data_labels_for_output = model_data->GetNodeDataLabelsForOutput();
  auto elem_data_labels_for_output = model_data->GetElementDataLabelsForOutput();
  auto derived_elem_data_labels = model_data->GetDerivedElementDataLabelsForOutput();

  nimble::ExodusOutput exodus_output;
  exodus_output.Initialize(output_exodus_name, mesh);
  exodus_output.InitializeDatabase(mesh,
                                   global_data_labels,
                                   node_data_labels_for_output,
                                   elem_data_labels_for_output,
                                   derived_elem_data_labels);

  model_data->SetReferenceCoordinates(mesh);

  watch_simulation.pop_region_and_report_time();

  if (time_integration_scheme == "explicit") {
    details_kokkos::ExplicitTimeIntegrator(*parser, mesh, data_manager, bc,
                           exodus_output,
                           contact_interface, block_material_interface_factory,
                           num_ranks, my_rank);
  }
  else {
    throw std::runtime_error("\n Time Integration Scheme Not Implemented \n");
  }

}


namespace details_kokkos {

int ExplicitTimeIntegrator(nimble::Parser & parser,
                           nimble::GenesisMesh & mesh,
                           nimble::DataManager & data_manager,
                           nimble::BoundaryConditionManager &bc_manager,
                           nimble::ExodusOutput & exodus_output,
                           std::shared_ptr<nimble::ContactInterface> contact_interface,
                           std::shared_ptr<nimble_kokkos::BlockMaterialInterfaceFactory> block_material_interface_factory,
                           int num_ranks,
                           int my_rank
) {

  int dim = mesh.GetDim();
  int num_nodes = static_cast<int>(mesh.GetNumNodes());
  int num_blocks = static_cast<int>(mesh.GetNumBlocks());

  //--- UH Temporary solution
  auto model_data = dynamic_cast<nimble_kokkos::ModelData *>(
      data_manager.GetMacroScaleData().get());
  std::map<int, nimble_kokkos::Block> blocks = model_data->GetBlocks();
  //--- UH Temporary solution

  // Build up block data for stress computation
  std::vector<nimble_kokkos::BlockData> block_data;
  for (auto &&block_it : blocks) {
    int block_id = block_it.first;
    nimble_kokkos::Block &block = block_it.second;
    nimble::Material *material_d = block.GetDeviceMaterialModel();
    int num_elem_in_block = mesh.GetNumElementsInBlock(block_id);
    int num_integration_points_per_element =
        block.GetHostElement()->NumIntegrationPointsPerElement();
    block_data.emplace_back(block, material_d, block_id, num_elem_in_block,
                            num_integration_points_per_element);
  }

  auto lumped_mass_h =
      model_data->GetHostScalarNodeData(nimble::FieldID::LumpedMass);
  auto reference_coordinate_h =
      model_data->GetHostVectorNodeData(nimble::FieldID::ReferenceCoordinate);

  int block_index;
  std::map<int, nimble_kokkos::Block>::iterator block_it;
  nimble_kokkos::ProfilingTimer watch_simulation;
  std::vector<nimble_kokkos::DeviceVectorNodeGatheredView>
      gathered_reference_coordinate_d(
          num_blocks, nimble_kokkos::DeviceVectorNodeGatheredView(
                          "gathered_reference_coordinates", 1));

  model_data->ComputeLumpedMass(mesh, gathered_reference_coordinate_d);

  std::vector<nimble_kokkos::DeviceVectorNodeGatheredView> gathered_displacement_d(num_blocks, nimble_kokkos::DeviceVectorNodeGatheredView("gathered_displacement", 1));
  std::vector<nimble_kokkos::DeviceVectorNodeGatheredView> gathered_internal_force_d(num_blocks, nimble_kokkos::DeviceVectorNodeGatheredView("gathered_internal_force", 1));
  std::vector<nimble_kokkos::DeviceVectorNodeGatheredView> gathered_contact_force_d(num_blocks, nimble_kokkos::DeviceVectorNodeGatheredView("gathered_contact_force", 1));
  for (block_index=0, block_it=blocks.begin(); block_it!=blocks.end() ; block_index++, block_it++) {
    int block_id = block_it->first;
    int num_elem_in_block = mesh.GetNumElementsInBlock(block_id);
    Kokkos::resize(gathered_displacement_d.at(block_index), num_elem_in_block);
    Kokkos::resize(gathered_internal_force_d.at(block_index), num_elem_in_block);
    Kokkos::resize(gathered_contact_force_d.at(block_index), num_elem_in_block);
  }
  //
  auto displacement_h = model_data->GetHostVectorNodeData(nimble::FieldID::Displacement);
  auto displacement_d = model_data->GetDeviceVectorNodeData(nimble::FieldID::Displacement);
  Kokkos::deep_copy(displacement_h, (double)(0.0));

  auto velocity_h = model_data->GetHostVectorNodeData(nimble::FieldID::Velocity);
  auto velocity_d = model_data->GetDeviceVectorNodeData(nimble::FieldID::Velocity);
  Kokkos::deep_copy(velocity_h, (double)(0.0));

  auto acceleration_h = model_data->GetHostVectorNodeData(nimble::FieldID::Acceleration);
  Kokkos::deep_copy(acceleration_h, (double)(0.0));

  auto internal_force_h = model_data->GetHostVectorNodeData(nimble::FieldID::InternalForce);
  auto internal_force_d = model_data->GetDeviceVectorNodeData(nimble::FieldID::InternalForce);
  Kokkos::deep_copy(internal_force_h, (double)(0.0));

  auto contact_force_h = model_data->GetHostVectorNodeData(nimble::FieldID::ContactForce);
  auto contact_force_d = model_data->GetDeviceVectorNodeData(nimble::FieldID::ContactForce);
  Kokkos::deep_copy(contact_force_h, (double)(0.0));

  // ------
  // Initialize the MPI container
  std::vector<int> global_node_ids(num_nodes);
  int const * const global_node_ids_ptr = mesh.GetNodeGlobalIds();
  for (int n=0 ; n<num_nodes ; ++n) {
    global_node_ids[n] = global_node_ids_ptr[n];
  }
  nimble::VectorCommunicator myVectorCommunicator;
  myVectorCommunicator.Initialize(global_node_ids);

  int mpi_scalar_dimension = 1;
  std::vector<double> mpi_scalar_buffer(mpi_scalar_dimension * num_nodes);
  int mpi_vector_dimension = 3;
  // ------

  // ------
  watch_simulation.push_region("Contact setup");
#ifdef NIMBLE_HAVE_ARBORX
  #ifdef NIMBLE_HAVE_MPI
    nimble::ArborXParallelContactManager contact_manager(contact_interface);
  #else
    nimble::ArborXSerialContactManager contact_manager(contact_interface);
  #endif // NIMBLE_HAVE_MPI
#else
  nimble::ContactManager contact_manager(contact_interface);
#endif // NIMBLE_HAVE_ARBORX

  bool contact_enabled = parser.HasContact();
  bool contact_visualization = parser.ContactVisualization();
  if (contact_enabled) {
    std::vector<std::string> contact_master_block_names, contact_slave_block_names;
    double penalty_parameter;
    nimble::ParseContactCommand(parser.ContactString(),
                                contact_master_block_names,
                                contact_slave_block_names,
                                penalty_parameter);
    std::vector<int> contact_master_block_ids, contact_slave_block_ids;
    mesh.BlockNamesToOnProcessorBlockIds(contact_master_block_names,
                                         contact_master_block_ids);
    mesh.BlockNamesToOnProcessorBlockIds(contact_slave_block_names,
                                         contact_slave_block_ids);
    contact_manager.SetPenaltyParameter(penalty_parameter);
    contact_manager.CreateContactEntities(mesh,
                                          myVectorCommunicator,
                                          contact_master_block_ids,
                                          contact_slave_block_ids);
    if (contact_visualization) {
//      std::string tag = parser.GetOutputTag();
#ifdef NIMBLE_HAVE_ARBORX
      std::string tag = "arborx";
#else
      std::string tag = "kokkos";
#endif
      std::string contact_visualization_exodus_file_name = nimble::IOFileName(parser.ContactVisualizationFileName(), "e", tag, my_rank, num_ranks);
      contact_manager.InitializeContactVisualization(contact_visualization_exodus_file_name);
    }
  }
  watch_simulation.pop_region_and_report_time();
  //---------

  watch_simulation.push_region("Lumped mass gather and compute");

  // MPI vector reduction on lumped mass
  for (unsigned int i=0 ; i<num_nodes ; i++) {
    mpi_scalar_buffer[i] = lumped_mass_h(i);
  }
  myVectorCommunicator.VectorReduction(mpi_scalar_dimension, mpi_scalar_buffer.data());
  for (int i=0 ; i<num_nodes ; i++) {
    lumped_mass_h(i) = mpi_scalar_buffer[i];
  }

  watch_simulation.pop_region_and_report_time();

  //------
  watch_simulation.push_region("BC enforcement");
  double time_current(0.0), time_previous(0.0);
  double final_time = parser.FinalTime();
  double delta_time(0.0), half_delta_time(0.0);
  const int num_load_steps = parser.NumLoadSteps();
  int output_frequency = parser.OutputFrequency();

  bc_manager.ApplyInitialConditions(reference_coordinate_h, velocity_h);
  bc_manager.ApplyKinematicBC(time_current, time_previous, reference_coordinate_h,
                              displacement_h, velocity_h);
  watch_simulation.pop_region_and_report_time();
  //------

  // Output to Exodus file
  watch_simulation.push_region("Output");
  model_data->UpdateOutputFields(mesh, gathered_reference_coordinate_d,
                                 gathered_displacement_d);
  {
    auto elem_data_labels_for_output = model_data->GetElementDataLabelsForOutput();
    auto derived_elem_data_labels = model_data->GetDerivedElementDataLabelsForOutput();
    auto node_data_for_output = model_data->GetNodeDataForOutput();
    auto elem_data_for_output = model_data->GetElementDataForOutput();
    std::vector<double> global_data;
    std::map<int, std::vector< std::vector<double> > > derived_elem_data;
    exodus_output.WriteStep(time_current,
                            global_data,
                            node_data_for_output,
                            elem_data_labels_for_output,
                            elem_data_for_output,
                            derived_elem_data_labels,
                            derived_elem_data);
  }
  if (contact_visualization) {
    contact_manager.ContactVisualizationWriteStep(time_current);
  }
  watch_simulation.pop_region_and_report_time();

  //
  double total_internal_force_time = 0.0, total_contact_time = 0.0;
  double total_arborx_time = 0.0,
      total_contact_applyd = 0.0, total_contact_getf = 0.0;
  double total_vector_reduction_time = 0.0;
  double total_update_avu_time = 0.0;
  double total_exodus_write_time = 0.0;
  //
  std::map<int, std::size_t> contactInfo;
  //
  watch_simulation.push_region("Time stepping loop");
  nimble_kokkos::ProfilingTimer watch_internal;

  for (int step = 0 ; step < num_load_steps ; ++step) {

    if (my_rank == 0) {
      if (10*(step+1) % num_load_steps == 0 && step != num_load_steps - 1) {
        std::cout << "   " << static_cast<int>( 100.0 * static_cast<double>(step+1)/num_load_steps ) << "% complete" << std::endl << std::flush;
      }
      else if (step == num_load_steps - 1) {
        std::cout << "  100% complete\n" << std::endl << std::flush;
      }
    }
    bool is_output_step = (step%output_frequency == 0 || step == num_load_steps - 1);

    watch_internal.push_region("Central difference");
    time_previous = time_current;
    time_current += final_time/num_load_steps;
    delta_time = time_current - time_previous;
    half_delta_time = 0.5*delta_time;

    // V^{n+1/2} = V^{n} + (dt/2) * A^{n}
    for (int i=0 ; i<num_nodes ; ++i) {
      velocity_h(i,0) += half_delta_time * acceleration_h(i,0);
      velocity_h(i,1) += half_delta_time * acceleration_h(i,1);
      velocity_h(i,2) += half_delta_time * acceleration_h(i,2);
    }
    total_update_avu_time += watch_internal.pop_region_and_report_time();

    // Apply kinematic boundary conditions
    watch_internal.push_region("BC enforcement");
    bc_manager.ApplyKinematicBC(time_current, time_previous, reference_coordinate_h, displacement_h, velocity_h);
    watch_internal.pop_region_and_report_time();

    // U^{n+1} = U^{n} + (dt)*V^{n+1/2}
    watch_internal.push_region("Central difference");
    for (int i=0 ; i<num_nodes ; ++i) {
      displacement_h(i,0) += delta_time * velocity_h(i,0);
      displacement_h(i,1) += delta_time * velocity_h(i,1);
      displacement_h(i,2) += delta_time * velocity_h(i,2);
    }

    // Copy the current displacement and velocity value to device memory
    Kokkos::deep_copy(displacement_d, displacement_h);
    Kokkos::deep_copy(velocity_d, velocity_h);

    total_update_avu_time += watch_internal.pop_region_and_report_time();

    watch_internal.push_region("Force calculation");

    nimble_kokkos::ProfilingTimer watch_internal_details;
    Kokkos::deep_copy(internal_force_d, (double)(0.0));
    if (contact_enabled) {
      watch_internal_details.push_region("Contact");
      Kokkos::deep_copy(contact_force_d, (double)(0.0));
      watch_internal_details.pop_region_and_report_time();
    }

    // Compute element-level kinematics

    watch_internal_details.push_region("Element kinematics");
    model_data->ComputeElementKinematics(mesh, gathered_reference_coordinate_d,
                                         gathered_displacement_d,
                                         gathered_internal_force_d);
    watch_internal_details.pop_region_and_report_time();

    {
      watch_internal_details.push_region("Material stress calculation");
      auto block_material_interface = block_material_interface_factory->create(time_previous, time_current, block_data, *model_data);
      block_material_interface->ComputeStress();
      watch_internal_details.pop_region_and_report_time();
    }

    // Stress divergence
    watch_internal_details.push_region("Stress divergence calculation");
    model_data->ComputeInternalForce(mesh, gathered_reference_coordinate_d,
                                     gathered_displacement_d,
                                     gathered_internal_force_d);
    watch_internal_details.pop_region_and_report_time();
    total_internal_force_time += watch_internal.pop_region_and_report_time();

    // Perform a reduction to obtain correct values on MPI boundaries
    watch_internal.push_region("MPI reduction");
    myVectorCommunicator.VectorReduction(mpi_vector_dimension, internal_force_h);
    total_vector_reduction_time += watch_internal.pop_region_and_report_time();
    //

    // Evaluate the contact force
    if (contact_enabled) {
      watch_internal_details.push_region("Contact");
      //
      Kokkos::deep_copy(contact_force_d, (double)(0.0));
      contact_manager.ApplyDisplacements(displacement_d);
      contact_manager.ComputeContactForce(step+1, is_output_step);
      //
      contact_manager.GetForces(contact_force_d);
      Kokkos::deep_copy(contact_force_h, contact_force_d);
      total_contact_time += watch_internal_details.pop_region_and_report_time();
      // Perform a reduction to obtain correct values on MPI boundaries
      {
        watch_internal_details.push_region("MPI reduction");
        myVectorCommunicator.VectorReduction(mpi_vector_dimension, contact_force_h);
        total_vector_reduction_time +=
            watch_internal_details.pop_region_and_report_time();
      }
      //
      auto tmpNum = contact_manager.numActiveContactFaces();
      if (tmpNum)
        contactInfo.insert(std::make_pair(step, tmpNum));
      //
//      if (contact_visualization && is_output_step)
//        contact_manager.ContactVisualizationWriteStep(time_current);
    }

    // fill acceleration vector A^{n+1} = M^{-1} ( F^{n} + b^{n} )
    watch_internal.push_region("Central difference");
    for (int i=0 ; i<num_nodes ; ++i) {
      acceleration_h(i,0) = (1.0/lumped_mass_h(i)) * (internal_force_h(i,0) + contact_force_h(i,0));
      acceleration_h(i,1) = (1.0/lumped_mass_h(i)) * (internal_force_h(i,1) + contact_force_h(i,1));
      acceleration_h(i,2) = (1.0/lumped_mass_h(i)) * (internal_force_h(i,2) + contact_force_h(i,2));
    }

    // V^{n+1}   = V^{n+1/2} + (dt/2)*A^{n+1}
    for (int i=0 ; i<num_nodes ; ++i) {
      velocity_h(i,0) += half_delta_time * acceleration_h(i,0);
      velocity_h(i,1) += half_delta_time * acceleration_h(i,1);
      velocity_h(i,2) += half_delta_time * acceleration_h(i,2);
    }
    total_update_avu_time += watch_internal.pop_region_and_report_time();

    if (is_output_step) {
      //
      watch_internal.push_region("Output");

      bc_manager.ApplyKinematicBC(time_current, time_previous, reference_coordinate_h, displacement_h, velocity_h);
      Kokkos::deep_copy(displacement_d, displacement_h);
      Kokkos::deep_copy(velocity_d, velocity_h);
      //--
      model_data->UpdateOutputFields(mesh, gathered_reference_coordinate_d,
                                     gathered_displacement_d);
      auto elem_data_labels_for_output = model_data->GetElementDataLabelsForOutput();
      auto derived_elem_data_labels = model_data->GetDerivedElementDataLabelsForOutput();

      auto node_data_for_output = model_data->GetNodeDataForOutput();
      auto elem_data_for_output = model_data->GetElementDataForOutput();
      std::vector<double> glbl_data;
      std::map<int, std::vector< std::vector<double> > > drvd_elem_data;
      exodus_output.WriteStep(time_current,
                              glbl_data,
                              node_data_for_output,
                              elem_data_labels_for_output,
                              elem_data_for_output,
                              derived_elem_data_labels,
                              drvd_elem_data);
      //
      //
      if (contact_visualization) {
        contact_manager.ContactVisualizationWriteStep(time_current);
      }

      total_exodus_write_time += watch_internal.pop_region_and_report_time();
    }

    watch_internal.push_region("Copy field data new to old");
    model_data->SwapStates();
    watch_internal.pop_region_and_report_time();

  } // loop over time steps
  double total_simulation_time = watch_simulation.pop_region_and_report_time();

  for (int irank = 0; irank < num_ranks; ++irank) {
#ifdef NIMBLE_HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if ((my_rank == irank) && (!contactInfo.empty())) {
      std::cout << " Rank " << irank << " has " << contactInfo.size()
                << " contact entries "
                << "(out of " << num_load_steps << " time steps)."<< std::endl;
      std::cout.flush();
    }
#ifdef NIMBLE_HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  if (my_rank == 0 && parser.WriteTimingDataFile()) {
//    double tcontact = total_contact_applyd + total_arborx_time + total_contact_getf;
    nimble::TimingInfo timing_writer{
        num_ranks,
        nimble::quanta::stopwatch::get_microsecond_timestamp(),
        total_simulation_time,
        total_internal_force_time,
        total_contact_time,
        total_exodus_write_time,
        total_vector_reduction_time
    };
    timing_writer.BinaryWrite();
  }

  if (my_rank == 0) {
    std::cout << " Total Time Loop = " << total_simulation_time << "\n";
    std::cout << " --- Internal Forces = " << total_internal_force_time << "\n";
    if (contact_enabled) {
//      double tcontact = total_contact_applyd + total_arborx_time + total_contact_getf;
      std::cout << " --- Contact = " << total_contact_time << "\n";
//      std::cout << " --- >>> Apply displ. = " << total_contact_applyd << "\n";
//      std::cout << " --- >>> Search / Project / Enforce = " << total_arborx_time << "\n";
      auto list_timers = contact_manager.getTimers();
      for (const auto& st_pair : list_timers)
        std::cout << " --- >>> >>> " << st_pair.first << " = " << st_pair.second << "\n";
      std::cout << " --- >>> Get Forces = " << total_contact_getf << "\n";
    }
    std::cout << " --- Exodus Write = " << total_exodus_write_time << "\n";
    std::cout << " --- Update AVU = " << total_update_avu_time << "\n";
    std::cout << " --- Vector Reduction = " << total_vector_reduction_time << "\n";
  }

  return 0;
}

}


}

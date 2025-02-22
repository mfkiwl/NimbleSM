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

#include "nimble_element.h"

#include <cmath>
#include <limits>
#include <vector>

#include "nimble_view.h"
#include "nimble_utils.h"

namespace nimble {

HexElement::HexElement()
{
  // 1/sqrt(3)
  constexpr double val = 0.577350269189626;
  // integration point 1
  int_pts_[0] = -val;
  int_pts_[1] = -val;
  int_pts_[2] = -val;
  // integration point 2
  int_pts_[3] = val;
  int_pts_[4] = -val;
  int_pts_[5] = -val;
  // integration point 3
  int_pts_[6] = val;
  int_pts_[7] = val;
  int_pts_[8] = -val;
  // integration point 4
  int_pts_[9]  = -val;
  int_pts_[10] = val;
  int_pts_[11] = -val;
  // integration point 5
  int_pts_[12] = -val;
  int_pts_[13] = -val;
  int_pts_[14] = val;
  // integration point 6
  int_pts_[15] = val;
  int_pts_[16] = -val;
  int_pts_[17] = val;
  // integration point 7
  int_pts_[18] = val;
  int_pts_[19] = val;
  int_pts_[20] = val;
  // integration point 8
  int_pts_[21] = -val;
  int_pts_[22] = val;
  int_pts_[23] = val;

  for (double & int_wt : int_wts_) { int_wt = 1.0; }

  ShapeFunctionValues(int_pts_, shape_fcn_vals_);

  ShapeFunctionDerivatives(int_pts_, shape_fcn_deriv_);
}

void
HexElement::ShapeFunctionValues(const double* natural_coords, double* shape_function_values)
{
  double r, s, t;
  constexpr double c = 1.0 / 8.0;

  // Loop over the integration points
  for (int i = 0; i < num_int_pts_; i++) {
    // Natural coordinates of this integration point
    r = natural_coords[dim_ * i];
    s = natural_coords[dim_ * i + 1];
    t = natural_coords[dim_ * i + 2];

    // Value of each of the eight shape functions at this integration point
    shape_function_values[8 * i]     = c * (1.0 - r) * (1.0 - s) * (1.0 - t);
    shape_function_values[8 * i + 1] = c * (1.0 + r) * (1.0 - s) * (1.0 - t);
    shape_function_values[8 * i + 2] = c * (1.0 + r) * (1.0 + s) * (1.0 - t);
    shape_function_values[8 * i + 3] = c * (1.0 - r) * (1.0 + s) * (1.0 - t);
    shape_function_values[8 * i + 4] = c * (1.0 - r) * (1.0 - s) * (1.0 + t);
    shape_function_values[8 * i + 5] = c * (1.0 + r) * (1.0 - s) * (1.0 + t);
    shape_function_values[8 * i + 6] = c * (1.0 + r) * (1.0 + s) * (1.0 + t);
    shape_function_values[8 * i + 7] = c * (1.0 - r) * (1.0 + s) * (1.0 + t);
  }
}

void
HexElement::ShapeFunctionDerivatives(const double* natural_coords, double* shape_function_derivatives)
{
  double r, s, t;
  constexpr double c = 1.0 / 8.0;

  // Loop over the integration points
  for (int int_pt = 0; int_pt < num_int_pts_; int_pt++) {
    // Natural coordinates of this integration point
    r = natural_coords[dim_ * int_pt];
    s = natural_coords[dim_ * int_pt + 1];
    t = natural_coords[dim_ * int_pt + 2];

    // Derivative of each of the eight shape functions w.r.t. the natural
    // coordinates at this integration point shape function 1
    shape_function_derivatives[24 * int_pt]     = -c * (1.0 - s) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 1] = -c * (1.0 - r) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 2] = -c * (1.0 - r) * (1.0 - s);
    // shape function 2
    shape_function_derivatives[24 * int_pt + 3] = c * (1.0 - s) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 4] = -c * (1.0 + r) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 5] = -c * (1.0 + r) * (1.0 - s);
    // shape function 3
    shape_function_derivatives[24 * int_pt + 6] = c * (1.0 + s) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 7] = c * (1.0 + r) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 8] = -c * (1.0 + r) * (1.0 + s);
    // shape function 4
    shape_function_derivatives[24 * int_pt + 9]  = -c * (1.0 + s) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 10] = c * (1.0 - r) * (1.0 - t);
    shape_function_derivatives[24 * int_pt + 11] = -c * (1.0 - r) * (1.0 + s);
    // shape function 5
    shape_function_derivatives[24 * int_pt + 12] = -c * (1.0 - s) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 13] = -c * (1.0 - r) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 14] = c * (1.0 - r) * (1.0 - s);
    // shape function 6
    shape_function_derivatives[24 * int_pt + 15] = c * (1.0 - s) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 16] = -c * (1.0 + r) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 17] = c * (1.0 + r) * (1.0 - s);
    // shape function 7
    shape_function_derivatives[24 * int_pt + 18] = c * (1.0 + s) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 19] = c * (1.0 + r) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 20] = c * (1.0 + r) * (1.0 + s);
    // shape function 8
    shape_function_derivatives[24 * int_pt + 21] = -c * (1.0 + s) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 22] = c * (1.0 - r) * (1.0 + t);
    shape_function_derivatives[24 * int_pt + 23] = c * (1.0 - r) * (1.0 + s);
  }
}

void
HexElement::ComputeLumpedMass(double density, const double* node_reference_coords, double* lumped_mass) const
{
  double consistent_mass_matrix[][num_nodes_] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

  nimble::Viewify<2, const double> view_node_reference_coords(node_reference_coords, {num_nodes_, dim_}, {dim_, 1});
  ComputeConsistentMass_impl(density, view_node_reference_coords, consistent_mass_matrix);

  for (int i = 0; i < num_nodes_; i++) {
    lumped_mass[i] = 0.0;
    for (int j = 0; j < num_nodes_; j++) { lumped_mass[i] += consistent_mass_matrix[i][j]; }
  }
}

#ifdef NIMBLE_HAVE_KOKKOS
void
HexElement::ComputeLumpedMass(
    double                                         density,
    nimble_kokkos::DeviceVectorNodeGatheredSubView node_reference_coords,
    nimble_kokkos::DeviceScalarNodeGatheredSubView lumped_mass) const
{
  double consistent_mass_matrix[][num_nodes_] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

  ComputeConsistentMass_impl(density, node_reference_coords, consistent_mass_matrix);

  for (int i = 0; i < num_nodes_; i++) {
    lumped_mass(i) = 0.0;
    for (int j = 0; j < num_nodes_; j++) { lumped_mass(i) += consistent_mass_matrix[i][j]; }
  }
}
#endif

double
HexElement::ComputeCharacteristicLength(const double* node_coords)
{
  // TODO Implement a better algorithm for finding the minimum
  //      length across the element.

  double characteristic_length, distance_squared, min_distance_squared, nx, ny, nz, mx, my, mz;
  double x_min, x_max, y_min, y_max, z_min, z_max;

  x_max = y_max = z_max = 0.0;
  min_distance_squared = x_min = y_min = z_min = std::numeric_limits<double>::max();

  for (int n = 0; n < num_nodes_; n++) {
    nx = node_coords[3 * n];
    ny = node_coords[3 * n + 1];
    nz = node_coords[3 * n + 2];
    if (nx < x_min) { x_min = nx; }
    if (nx > x_max) { x_max = nx; }
    if (ny < y_min) { y_min = ny; }
    if (ny > y_max) { y_max = ny; }
    if (nz < z_min) { z_min = nz; }
    if (nz > z_max) { z_max = nz; }
    for (int m = n + 1; m < num_nodes_; m++) {
      mx               = node_coords[3 * m];
      my               = node_coords[3 * m + 1];
      mz               = node_coords[3 * m + 2];
      distance_squared = (nx - mx) * (nx - mx) + (ny - my) * (ny - my) + (nz - mz) * (nz - mz);
      if (distance_squared < min_distance_squared) { min_distance_squared = distance_squared; }
    }
  }
  characteristic_length = std::sqrt(min_distance_squared);

  double min_box_length = x_max - x_min;
  if (y_max - y_min < min_box_length) { min_box_length = y_max - y_min; }
  if (z_max - z_min < min_box_length) { min_box_length = z_max - z_min; }

  if (min_box_length < characteristic_length) { characteristic_length = min_box_length; }

  return characteristic_length;
}

void
HexElement::ComputeVolumeAverage(
    const double* node_current_coords,
    int           num_quantities,
    const double* int_pt_quantities,
    double&       volume,
    double*       volume_averaged_quantities) const
{
  nimble::Viewify<2, const double> node_reference_coords(node_current_coords, {num_nodes_, dim_}, {dim_, 1});

  std::vector<double> tmp_disp(num_nodes_ * dim_, 0.0);
  nimble::Viewify<2, const double> node_disp(tmp_disp.data(), {num_nodes_, dim_}, {dim_, 1});

  nimble::Viewify<2, const double> int_pt_quantities_v(int_pt_quantities, {num_int_pts_, num_quantities},
                                                       {num_quantities, 1});

  nimble::Viewify<1> vol_ave_quantity(volume_averaged_quantities, num_quantities);

  ComputeVolumeAverageQuantities_impl(node_reference_coords, node_disp, int_pt_quantities_v,
                                      vol_ave_quantity, num_quantities, volume);
}

#ifdef NIMBLE_HAVE_KOKKOS
void
HexElement::ComputeVolume(
    nimble_kokkos::DeviceVectorNodeGatheredSubView node_reference_coords,
    nimble_kokkos::DeviceVectorNodeGatheredSubView node_displacements,
    nimble_kokkos::DeviceScalarElemSingleEntryView elem_volume) const
{
  constexpr int num_quantities = 0;
  double volume = 0.0;
  nimble::Viewify<2, const double> int_pt_quantities_v(nullptr, {num_int_pts_, num_quantities},
                                                       {num_quantities, 1});
  nimble::Viewify<1> vol_ave_quantity(nullptr, num_quantities);
  ComputeVolumeAverageQuantities_impl(node_reference_coords, node_displacements,
                                      int_pt_quantities_v, vol_ave_quantity, num_quantities, volume);
  //
  // DJL cleaner to use
  // elem_volume(0) = volume;
  // but this isn't working with older versions of Kokkos
  double* v = elem_volume.data();
  *v        = volume;
}

void
HexElement::ComputeVolumeAverageFullTensor(
    nimble_kokkos::DeviceVectorNodeGatheredSubView     node_reference_coords,
    nimble_kokkos::DeviceVectorNodeGatheredSubView     node_displacements,
    nimble_kokkos::DeviceFullTensorIntPtSubView        int_pt_quantities,
    nimble_kokkos::DeviceFullTensorElemSingleEntryView vol_ave_quantity) const
{
  constexpr int num_quantities = 9;
  double volume = 0.0;
  ComputeVolumeAverageQuantities_impl(node_reference_coords, node_displacements,
                                      int_pt_quantities, vol_ave_quantity, num_quantities, volume);
}

void
HexElement::ComputeVolumeAverageSymTensor(
    nimble_kokkos::DeviceVectorNodeGatheredSubView    node_reference_coords,
    nimble_kokkos::DeviceVectorNodeGatheredSubView    node_displacements,
    nimble_kokkos::DeviceSymTensorIntPtSubView        int_pt_quantities,
    nimble_kokkos::DeviceSymTensorElemSingleEntryView vol_ave_quantity) const
{
  constexpr int num_quantities = 6;
  double volume = 0.0;
  ComputeVolumeAverageQuantities_impl(node_reference_coords, node_displacements,
                                      int_pt_quantities, vol_ave_quantity, num_quantities, volume);
}
#endif

void
HexElement::ComputeDeformationGradients(
    const double* node_reference_coords,
    const double* node_current_coords,
    double*       deformation_gradients) const
{
  nimble::Viewify<2, const double> node_reference_coords_v(node_reference_coords, {num_nodes_, dim_}, {dim_, 1});

  std::vector<double> tmp_disp(num_nodes_ * dim_, 0.0);
  for (size_t ii = 0; ii < tmp_disp.size(); ++ii)
    tmp_disp[ii] = node_current_coords[ii] - node_reference_coords[ii];
  nimble::Viewify<2, const double> node_disp(tmp_disp.data(), {num_nodes_, dim_}, {dim_, 1});

  constexpr int dim2 = dim_ * dim_;
  nimble::Viewify<2> deformation_gradients_v(deformation_gradients, {num_int_pts_, dim2}, {dim2, 1});

  ComputeDeformationGradients_impl(node_reference_coords_v, node_disp,
                                   deformation_gradients_v);
}

#ifdef NIMBLE_HAVE_KOKKOS
void
HexElement::ComputeDeformationGradients(
    nimble_kokkos::DeviceVectorNodeGatheredSubView node_reference_coords,
    nimble_kokkos::DeviceVectorNodeGatheredSubView node_displacements,
    nimble_kokkos::DeviceFullTensorIntPtSubView    deformation_gradients) const
{
  ComputeDeformationGradients_impl(node_reference_coords, node_displacements, deformation_gradients);
}
#endif

void
HexElement::ComputeTangent(const double* node_current_coords, const double* material_tangent, double* element_tangent)
{
  double jac_det;
  double cc1, cc2, cc3, sfd1, sfd2, sfd3;

  for (int i = 0; i < 24; i++) {
    for (int j = 0; j < 24; j++) { element_tangent[i * 24 + j] = 0.0; }
  }

  // \mathbf{K}_{elem} = \int_{\Omega_{o}} \mathbf{B}^{T}_{0} \mathbf{C}^{SE}
  // \mathbf{B}_{o} \Omega_{o}

  //      dN1/dX    0       0     dN2/dX    0       0     ...  dN8/dX    0 0
  //        0     dN1/dY    0       0     dN2/dY    0     ...    0     dN8/dY 0
  //        0       0     dN1/dZ    0       0     dN2/dZ  ...    0       0
  //        dN8/dZ
  // B =  dN1/dY  dN1/dX    0     dN2/dY  dN2/dX    0     ...  dN8/dY  dN8/dX 0
  //        0     dN1/dZ  dN1/dY    0     dN2/dZ  dN2/dY  ...    0     dN8/dZ
  //        dN8/dY
  //      dN1/dZ    0     dN1/dX  dN2/dZ    0     dN2/dX  ...  dN8/dZ    0
  //      dN8/dX

  double B[6][24];
  for (auto & B_i : B) {
    for (double & B_ij : B_i) { B_ij = 0.0; }
  }

  for (int int_pt = 0; int_pt < num_int_pts_; int_pt++) {
    // \mathbf{a} = \sum_{i}^{N_{node}} x_{i} \frac{\partial N_{i}
    // (\xi)}{\partial \xi}
    double a[][3]     = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    double a_inv[][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    double temp[6][24];
    for (auto & t_i : temp) {
      for (double & t_ij : t_i) { t_ij = 0.0; }
    }

    for (int n = 0; n < num_nodes_; n++) {
      cc1  = node_current_coords[3 * n];
      cc2  = node_current_coords[3 * n + 1];
      cc3  = node_current_coords[3 * n + 2];
      sfd1 = shape_fcn_deriv_[24 * int_pt + 3 * n];
      sfd2 = shape_fcn_deriv_[24 * int_pt + 3 * n + 1];
      sfd3 = shape_fcn_deriv_[24 * int_pt + 3 * n + 2];
      a[0][0] += cc1 * sfd1;
      a[0][1] += cc1 * sfd2;
      a[0][2] += cc1 * sfd3;
      a[1][0] += cc2 * sfd1;
      a[1][1] += cc2 * sfd2;
      a[1][2] += cc2 * sfd3;
      a[2][0] += cc3 * sfd1;
      a[2][1] += cc3 * sfd2;
      a[2][2] += cc3 * sfd3;
    }
    jac_det = Invert3x3(a, a_inv);

    // derivatives of shape function with respect to current coordinate
    double dN_dcc1, dN_dcc2, dN_dcc3;
    for (int n = 0; n < num_nodes_; n++) {
      sfd1            = shape_fcn_deriv_[24 * int_pt + 3 * n];
      sfd2            = shape_fcn_deriv_[24 * int_pt + 3 * n + 1];
      sfd3            = shape_fcn_deriv_[24 * int_pt + 3 * n + 2];
      dN_dcc1         = sfd1 * a_inv[0][0] + sfd2 * a_inv[1][0] + sfd3 * a_inv[2][0];
      dN_dcc2         = sfd1 * a_inv[0][1] + sfd2 * a_inv[1][1] + sfd3 * a_inv[2][1];
      dN_dcc3         = sfd1 * a_inv[0][2] + sfd2 * a_inv[1][2] + sfd3 * a_inv[2][2];
      B[0][3 * n]     = dN_dcc1;
      B[1][3 * n + 1] = dN_dcc2;
      B[2][3 * n + 2] = dN_dcc3;
      B[3][3 * n]     = dN_dcc2;
      B[3][3 * n + 1] = dN_dcc1;
      B[4][3 * n + 1] = dN_dcc3;
      B[4][3 * n + 2] = dN_dcc2;
      B[5][3 * n]     = dN_dcc3;
      B[5][3 * n + 2] = dN_dcc1;
    }

    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 24; j++) {
        for (int k = 0; k < 6; k++) { temp[i][j] += material_tangent[36 * int_pt + 6 * i + k] * B[k][j]; }
      }
    }

    for (int i = 0; i < 24; i++) {
      for (int j = 0; j < 24; j++) {
        for (int k = 0; k < 6; k++) {
          element_tangent[i * 24 + j] += B[k][i] * temp[k][j] * int_wts_[int_pt] * jac_det;
        }
      }
    }
  }
}

void
HexElement::ComputeNodalForces(const double* node_current_coords, const double* int_pt_stresses, double* node_forces)
{
  nimble::Viewify<2, const double> node_reference_coords(node_current_coords, {num_nodes_, dim_}, {dim_, 1});

  std::vector<double> tmp_disp(num_nodes_ * dim_, 0.0);
  nimble::Viewify<2, const double> node_disp(tmp_disp.data(), {num_nodes_, dim_}, {dim_, 1});

  constexpr int sym_tensor_size = 6;
  nimble::Viewify<2, const double> int_stresses_v(int_pt_stresses, {num_int_pts_, sym_tensor_size},
                                                  {sym_tensor_size, 1});

  nimble::Viewify<2> node_forces_v(node_forces, {num_nodes_, dim_}, {dim_, 1});

  ComputeNodalForces_impl(node_reference_coords, node_disp, int_stresses_v, node_forces_v);
}

}  // namespace nimble

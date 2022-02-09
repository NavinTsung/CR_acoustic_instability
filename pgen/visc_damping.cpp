//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <vector>
#include <hdf5.h>

#ifdef MPI_PARALLEL
#include <mpi.h>   // MPI_COMM_WORLD, MPI_INFO_NULL
#endif

// Athena++ headers
#include "../globals.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"

//======================================================================================
/*! \file beam.cpp
 *  \brief Beam test for the radiative transfer module
 *
 *====================================================================================*/


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================

static Real gg = 5./3.;
static Real gg1 = gg/(gg - 1.);
static Real gc = 4./3.;
static Real gc1 = gc/(gc - 1.);

void InnerBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh);

void Viscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
  const AthenaArray<Real> &bc, int is, int ie, int js, int je, int ks, int ke);

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {    
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerBoundaryMHD);
  }
  EnrollViscosityCoefficient(Viscosity);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  // Boundary perturbation
  phydro->x_left = pin->GetReal("mesh","x1min");
  phydro->x_right = pin->GetReal("mesh","x1max");
  phydro->amp_left = pin->GetOrAddReal("problem","amp_left",0.01);

  // Viscosity
  phydro->viscos_buffer = pin->GetOrAddReal("problem","viscos_buffer",0.1);


  // Initialize hydro variable
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        Real x1 = pcoord->x1v(i);
        
        // Initialize hydro variables
        phydro->u(IDN,k,j,i) = gg;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0/(gg - 1.);
        }

      } //end i
    } //end j
  } //end k
  return;
}// end ProblemGenerator

void InnerBoundaryMHD(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, 
     int ks, int ke, int ngh)
{
  Hydro *phydro = pmb->phydro;
  Real amp_left = phydro->amp_left;
  Real x_left = phydro->x_left;
  Real x_right = phydro->x_right;
  Real length = x_right - x_left;
  Real wvlen = length/10.;
  Real k_wv = 2.*PI/wvlen;
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        for (int n=0; n<(NHYDRO); ++n) {
          Real x1 = pco->x1v(is-i);
          Real cs0 = 1.;
          Real vx = amp_left*cs0*sin(k_wv*(x1 - x_left) - k_wv*cs0*time);
          Real rho = gg*(1. + vx/cs0);
          Real pg = (1. + gg*vx/cs0);
          if (n==(IDN)) {
            prim(IDN,k,j,is-i) = rho;
          } else if (n==(IVX)) {
            prim(IVX,k,j,is-i) = vx;
          } else if (n==(IVY)) {
            prim(IVY,k,j,is-i) = 0.;
          } else if (n==(IEN)) {
            prim(IEN,k,j,is-i) = pg;
          } else {
            prim(n,k,j,is-i) = prim(n,k,j,is);
          }
        }
      }
    }
  }
  return;
}// end InnerBoundaryMHD

void Viscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
  const AthenaArray<Real> &bc, int is, int ie, int js, int je, int ks, int ke)
  {
    Hydro *phydro = pmb->phydro;
    Real xmin = phydro->x_left;
    Real xmax = phydro->x_right;
    Real domain_size = xmax - xmin;
    Real xmin_vis_buffer = xmin + phydro->viscos_buffer*domain_size;
    Real xmax_vis_buffer = xmax - phydro->viscos_buffer*domain_size;

    int kl = pmb->ks, ku = pmb->ke;
    int jl = pmb->js, ju = pmb->je;
    int il = pmb->is, iu = pmb->ie;

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {

          Real x1 = pmb->pcoord->x1v(i);

          if ((x1 > xmin) && (x1 < xmin_vis_buffer)) {
            phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = 0.0;
          } else if ((x1 < xmax) && (x1 > xmax_vis_buffer)) {
            phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->nu_iso;
          }

        }
      }
    }
  } // End Viscosity

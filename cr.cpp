//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file radiation.cpp
//  \brief implementation of functions in class Radiation
//======================================================================================


#include <sstream>  // msg
#include <iostream>  // cout
#include <stdexcept> // runtime erro
#include <stdio.h>  // fopen and fwrite


// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp" 
#include "cr.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../globals.hpp"
#include "../coordinates/coordinates.hpp"
#include "integrators/cr_integrators.hpp"

// constructor, initializes data structures and parameters

// The default opacity function.

// This function also needs to set the streaming velocity
// This is needed to calculate the work term 
inline void DefaultDiff(MeshBlock *pmb, AthenaArray<Real> &u_cr, 
              AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Real dt)
{
  // set the default opacity to be a large value in the default hydro case
  CosmicRay *pcr=pmb->pcr;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-1, iu=pmb->ie+1;
  if(pmb->block_size.nx2 > 1){
    jl -= 1;
    ju += 1;
  }
  if(pmb->block_size.nx3 > 1){
    kl -= 1;
    ku += 1;
  }


  Real invlim = 1.0/pcr->vmax;


  for(int k=kl; k<=ku; ++k){
    for(int j=jl; j<=ju; ++j){
#pragma omp simd
      for(int i=il; i<=iu; ++i){

        pcr->sigma_diff(0,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(1,k,j,i) = pcr->max_opacity;
        pcr->sigma_diff(2,k,j,i) = pcr->max_opacity;  

      }
    }
  }

  // Need to calculate the rotation matrix 
  // We need this to determine the direction of rotation velocity


  // The information stored in the array
  // b_angle is
  // b_angle[0]=sin_theta_b
  // b_angle[1]=cos_theta_b
  // b_angle[2]=sin_phi_b
  // b_angle[3]=cos_phi_b
 
  if(pcr->stream_flag){
    if(MAGNETIC_FIELDS_ENABLED){
      //First, calculate B_dot_grad_Pc
      for(int k=kl; k<=ku; ++k){
        for(int j=jl; j<=ju; ++j){
        // diffusion coefficient is calculated with respect to B direction
        // Use a simple estimate of Grad Pc

      // x component
          pmb->pcoord->CenterWidth1(k,j,il-1,iu+1,pcr->cwidth);
          for(int i=il; i<=iu; ++i){
            Real distance = 0.5*(pcr->cwidth(i-1) + pcr->cwidth(i+1))
                           + pcr->cwidth(i);
            Real dprdx=(pcr->prtensor_cr(PC11,k,j,i+1) * u_cr(CRE,k,j,i+1)
                         - pcr->prtensor_cr(PC11,k,j,i-1) * u_cr(CRE,k,j,i-1));
            dprdx /= distance;
            pcr->sigma_adv(0,k,j,i) = dprdx;
          }
      // y component
          pmb->pcoord->CenterWidth2(k,j-1,il,iu,pcr->cwidth1);       
          pmb->pcoord->CenterWidth2(k,j,il,iu,pcr->cwidth);
          pmb->pcoord->CenterWidth2(k,j+1,il,iu,pcr->cwidth2);

          for(int i=il; i<=iu; ++i){
            Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                           + pcr->cwidth(i);
            Real dprdy=(pcr->prtensor_cr(PC22,k,j+1,i) * u_cr(CRE,k,j+1,i)
                             - pcr->prtensor_cr(PC22,k,j-1,i) * u_cr(CRE,k,j-1,i));
            dprdy /= distance;
            pcr->sigma_adv(1,k,j,i) = dprdy;

          } 
  // z component
          pmb->pcoord->CenterWidth3(k-1,j,il,iu,pcr->cwidth1);       
          pmb->pcoord->CenterWidth3(k,j,il,iu,pcr->cwidth);
          pmb->pcoord->CenterWidth3(k+1,j,il,iu,pcr->cwidth2);

          for(int i=il; i<=iu; ++i){
            Real distance = 0.5*(pcr->cwidth1(i) + pcr->cwidth2(i))
                            + pcr->cwidth(i);
            Real dprdz=(pcr->prtensor_cr(PC33,k+1,j,i) * u_cr(CRE,k+1,j,i)
                             - pcr->prtensor_cr(PC33,k-1,j,i) * u_cr(CRE,k-1,j,i));
            dprdz /= distance;
            pcr->sigma_adv(2,k,j,i) = dprdz;
          }       


          for(int i=il; i<=iu; ++i){
            // Now calculate the angles of B
            Real bxby = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                             bcc(IB2,k,j,i)*bcc(IB2,k,j,i));
            Real btot = sqrt(bcc(IB1,k,j,i)*bcc(IB1,k,j,i) +
                             bcc(IB2,k,j,i)*bcc(IB2,k,j,i) + 
                             bcc(IB3,k,j,i)*bcc(IB3,k,j,i));
            
            if(btot > TINY_NUMBER){
              pcr->b_angle(0,k,j,i) = bxby/btot;
              pcr->b_angle(1,k,j,i) = bcc(IB3,k,j,i)/btot;
            }else{
              pcr->b_angle(0,k,j,i) = 1.0;
              pcr->b_angle(1,k,j,i) = 0.0;
            }
            if(bxby > TINY_NUMBER){
              pcr->b_angle(2,k,j,i) = bcc(IB2,k,j,i)/bxby;
              pcr->b_angle(3,k,j,i) = bcc(IB1,k,j,i)/bxby;
            }else{
              pcr->b_angle(2,k,j,i) = 0.0;
              pcr->b_angle(3,k,j,i) = 1.0;            
            }

            Real va = sqrt(btot*btot/prim(IDN,k,j,i));
            if(va < TINY_NUMBER){
              pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            }else{
              Real b_grad_pc = bcc(IB1,k,j,i) * pcr->sigma_adv(0,k,j,i)
                             + bcc(IB2,k,j,i) * pcr->sigma_adv(1,k,j,i)
                             + bcc(IB3,k,j,i) * pcr->sigma_adv(2,k,j,i);
              pcr->sigma_adv(0,k,j,i) = fabs(b_grad_pc)/(va * (1.0 + 
                                   pcr->prtensor_cr(PC11,k,j,i)) * invlim * 
                                   u_cr(CRE,k,j,i));
            }
            pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;

          }//end i        

        }// end j
      }// end k

    }// End MHD  
    else{

      for(int k=kl; k<=ku; ++k){
        for(int j=jl; j<=ju; ++j){
  #pragma omp simd
          for(int i=il; i<=iu; ++i){

            pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
            pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;  

            pcr->v_adv(0,k,j,i) = 0.0;   
            pcr->v_adv(1,k,j,i) = 0.0;
            pcr->v_adv(2,k,j,i) = 0.0;
          }
        }
      }

    }// end MHD
  }else{
    for(int k=kl; k<=ku; ++k){
      for(int j=jl; j<=ju; ++j){
#pragma omp simd
        for(int i=il; i<=iu; ++i){

          pcr->sigma_adv(0,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(1,k,j,i) = pcr->max_opacity;
          pcr->sigma_adv(2,k,j,i) = pcr->max_opacity;  

          pcr->v_adv(0,k,j,i) = 0.0;   
          pcr->v_adv(1,k,j,i) = 0.0;
          pcr->v_adv(2,k,j,i) = 0.0;
        }
      }
    }    
  }  
}

inline void DefaultCRTensor(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  // Default values are 1/3
  CosmicRay *pcr=pmb->pcr;
  int nz1 = pmb->block_size.nx1 + 2*(NGHOST);
  int nz2 = pmb->block_size.nx2;
  if(nz2 > 1) nz2 += 2*(NGHOST);
  int nz3 = pmb->block_size.nx3;
  if(nz3 > 1) nz3 += 2*(NGHOST);
  for(int k=0; k<nz3; ++k){
    for(int j=0; j<nz2; ++j){
      for(int i=0; i<nz1; ++i){
        pcr->prtensor_cr(PC11,k,j,i) = 1.0/3.0;
        pcr->prtensor_cr(PC22,k,j,i) = 1.0/3.0;
        pcr->prtensor_cr(PC33,k,j,i) = 1.0/3.0;
        pcr->prtensor_cr(PC12,k,j,i) = 0.0;
        pcr->prtensor_cr(PC13,k,j,i) = 0.0;
        pcr->prtensor_cr(PC23,k,j,i) = 0.0;
      }
    }
  }
  
}

CosmicRay::CosmicRay(MeshBlock *pmb, ParameterInput *pin)
{
  vmax = pin->GetOrAddReal("cr","vmax",1.0);
  vlim = pin->GetOrAddReal("cr","vlim",0.9);
  max_opacity = pin->GetOrAddReal("cr","max_opacity",1.e10);
  stream_flag = pin->GetOrAddInteger("cr","vs_flag",1);  
  
  pmy_block = pmb;

  int n1z = pmy_block->block_size.nx1 + 2*(NGHOST);
  int n2z = 1, n3z = 1;
  if (pmy_block->block_size.nx2 > 1) n2z = pmy_block->block_size.nx2 + 2*(NGHOST);
  if (pmy_block->block_size.nx3 > 1) n3z = pmy_block->block_size.nx3 + 2*(NGHOST);
  

  u_cr.NewAthenaArray(NCR,n3z,n2z,n1z);
  u_cr1.NewAthenaArray(NCR,n3z,n2z,n1z);

  // Array to store boundary input
  rho_in.NewAthenaArray(n1z); 
  pg_in.NewAthenaArray(n1z);
  ec_in.NewAthenaArray(n1z);
  fc_in.NewAthenaArray(n1z);
  g_in.NewAthenaArray(n1z);
  H_in.NewAthenaArray(n1z);

  sigma_diff.NewAthenaArray(3,n3z,n2z,n1z);
  sigma_adv.NewAthenaArray(3,n3z,n2z,n1z);

  v_adv.NewAthenaArray(3,n3z,n2z,n1z);
  v_diff.NewAthenaArray(3,n3z,n2z,n1z);

  prtensor_cr.NewAthenaArray(6,n3z,n2z,n1z);

  b_grad_pc.NewAthenaArray(n3z,n2z,n1z);
  b_angle.NewAthenaArray(4,n3z,n2z,n1z);
  
  //allocate memory to store the flux
  flux[X1DIR].NewAthenaArray(NCR,n3z,n2z,n1z);
  if(n2z > 1) flux[X2DIR].NewAthenaArray(NCR,n3z,n2z,n1z);
  if(n3z > 1) flux[X3DIR].NewAthenaArray(NCR,n3z,n2z,n1z);

  cwidth.NewAthenaArray(n1z);
  cwidth1.NewAthenaArray(n1z);
  cwidth2.NewAthenaArray(n1z);
  
  // set a default opacity function
  UpdateDiff = DefaultDiff;
  UpdateCRTensor = DefaultCRTensor;

  pcrintegrator = new CRIntegrator(this, pin);

}

// destructor

CosmicRay::~CosmicRay()
{
  u_cr.DeleteAthenaArray();
  u_cr1.DeleteAthenaArray();
  sigma_diff.DeleteAthenaArray();
  sigma_adv.DeleteAthenaArray();
  prtensor_cr.DeleteAthenaArray();
  b_grad_pc.DeleteAthenaArray();
  b_angle.DeleteAthenaArray();

  v_adv.DeleteAthenaArray();
  v_diff.DeleteAthenaArray();

  cwidth.DeleteAthenaArray();
  cwidth1.DeleteAthenaArray();
  cwidth2.DeleteAthenaArray();

  // Array to store boundary input
  rho_in.DeleteAthenaArray(); 
  pg_in.DeleteAthenaArray();
  ec_in.DeleteAthenaArray();
  fc_in.DeleteAthenaArray();
  g_in.DeleteAthenaArray();
  H_in.DeleteAthenaArray();
  
  flux[X1DIR].DeleteAthenaArray();
  if(pmy_block->block_size.nx2 > 1) flux[X2DIR].DeleteAthenaArray();
  if(pmy_block->block_size.nx3 > 1) flux[X3DIR].DeleteAthenaArray();
  
  delete pcrintegrator;
  
}


//Enrol the function to update opacity

void CosmicRay::EnrollDiffFunction(CROpa_t MyDiffFunction)
{
  UpdateDiff = MyDiffFunction;
  
}

void CosmicRay::EnrollCRTensorFunction(CR_t MyTensorFunction)
{
  UpdateCRTensor = MyTensorFunction;
  
}


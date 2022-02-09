#ifndef CR_HPP
#define CR_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include <string>



class MeshBlock;
class ParameterInput;
class CRIntegrator;

//! \class CosmicRay
//  \brief CosmicRay data and functions



// Array indices for  moments
enum {CRE=0, CRF1=1, CRF2=2, CRF3=3};


class CosmicRay {
  friend class CRIntegrator;
  friend class BoundaryValues;
public:
  CosmicRay(MeshBlock *pmb, ParameterInput *pin);
//  ~CosmicRay();
    
  AthenaArray<Real> u_cr, u_cr1, u_cr2; //cosmic ray energy density and flux

  AthenaArray<Real> coarse_cr_;

  //   diffusion coefficients for both normal diffusion term, and advection term
  AthenaArray<Real> sigma_diff, sigma_adv; 

  AthenaArray<Real> v_adv; // streaming velocity
  AthenaArray<Real> v_diff; // the diffuion velocity, need to calculate the flux
    
  int refinement_idx{-1};

  AthenaArray<Real> flux[3]; // store transport flux, also need for refinement
  
  Real vmax; // the maximum velocity (effective speed of light)
  Real vlim;
  Real max_opacity;

  // Arrays to store boundary input
  Real B, kap, Tfloor, Tceil, T0, L0, epsil;
  AthenaArray<Real> rho_in, v_in, pg_in, ec_in, fc_in, g_in, H_in, Hnocool_in;  

  // Boundary perturbation 
  Real tau, amp_left, amp_right, wvlen, x_left, x_right, k_wv, viscos_buffer_left, viscos_buffer_right, cool_buffer_left, cool_buffer_right, couple_timing;

  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Fluid
  CellCenteredBoundaryVariable cr_bvar;
  
  CRIntegrator *pcrintegrator;
  
  //Function in problem generators to update opacity
  void EnrollOpacityFunction(CROpacityFunc MyOpacityFunction);

  void EnrollUserCRSource(CRSrcTermFunc my_func);
  bool cr_source_defined;

  // The function pointer for the diffusion coefficient
  CROpacityFunc UpdateOpacity;

  AthenaArray<Real> cwidth; 
  AthenaArray<Real> cwidth1;
  AthenaArray<Real> cwidth2;
  AthenaArray<Real> b_grad_pc; // array to store B\dot Grad Pc
  AthenaArray<Real> b_angle; //sin\theta,cos\theta,sin\phi,cos\phi of B direction

  int stream_flag; // flag to include streaming or not
  int src_flag; // flag to 

private:

  CRSrcTermFunc UserSourceTerm_;

};

#endif // CR_HPP

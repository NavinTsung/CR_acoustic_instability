import sys
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import ticker, cm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import scipy.integrate as integrate 
import scipy.optimize as opt
import h5py 

# Import athena data reader
sys.path.append('/Users/tsunhinnavintsung/Workspace/Codes/Athena++')
import athena_read3 as ar 

# Matplotlib default param
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['lines.markersize'] = 0.7
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Global parameters
gg = 5./3. 
gg1 = gg/(gg - 1.)
gc = 4./3. 
gc1 = gc/(gc - 1.)

def plotdefault():
  plt.rcParams.update(plt.rcParamsDefault)
  plt.rcParams['font.size'] = 12
  plt.rcParams['legend.fontsize'] = 12
  plt.rcParams['legend.loc'] = 'best'
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.markersize'] = 2.
  plt.rcParams['mathtext.fontset'] = 'stix'
  plt.rcParams['font.family'] = 'STIXGeneral'
  return

def latexify(columns=2, square=False, num_fig=0):
  """
  Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.
  Parameters
  ----------
  columns : {1, 2}
  """
  assert(columns in [1, 2])

  fig_width_pt = 240.0 if (columns == 1) else 504.0
  inches_per_pt = 1./72.27 # Convert pt to inch
  golden_mean = (np.sqrt(5.) - 1.)/2. 
  square_size = 1.
  fig_width = fig_width_pt*inches_per_pt # Width in inches
  fig_height = fig_width*golden_mean
  if square:
    fig_height = fig_width*square_size # Height in inches
  if num_fig != 0:
    fig_height = fig_width/num_fig
  fig_size = [fig_width, fig_height]

  font_size = 10 if columns == 1 else 8

  plt.rcParams['pdf.fonttype'] = 42
  plt.rcParams['ps.fonttype'] = 42
  plt.rcParams['font.size'] = font_size
  plt.rcParams['axes.labelsize'] = font_size
  plt.rcParams['axes.titlesize'] = font_size
  plt.rcParams['xtick.labelsize'] = font_size
  plt.rcParams['ytick.labelsize'] = font_size
  plt.rcParams['legend.fontsize'] = font_size
  plt.rcParams['figure.figsize'] = fig_size
  plt.rcParams['figure.titlesize'] = 12
  return 



class Power:
  def __init__(self, alpha, beta, eta, ms, psi=0., rho0=1., pg0=1., r0=1., vmax=200.):
    self.alpha = alpha # pc_h/pg_h
    self.beta = beta # 2*pg_h/b^2
    self.eta = eta # kappa/gc L c_s
    self.ms = ms # v/cs sonic mach number, must be greater than 0
    self.psi = psi 
    self.phi = (gc/(2. - gc))*(1. - psi)
    self.rho0 = rho0 
    self.pg0 = pg0 
    self.r0 = r0 
    self.vmax = vmax
    self.pc0 = alpha*pg0 
    self.b = np.sqrt(2.*pg0/beta)
    self.cs0 = np.sqrt(gg*pg0/rho0)
    self.cc0 = np.sqrt(gc*self.pc0/rho0)
    self.va0 = self.b/np.sqrt(rho0)
    self.v0 = ms*self.cs0
    self.Lc0 = r0/self.phi
    self.kappa = eta*gc*self.Lc0*self.cs0 
    self.ldiff = self.kappa/self.cs0

    if ms < 0.0:
      raise ValueError('Use profile.py instead.')
  # End of init

  def pg(self, x):
    pg0 = self.pg0
    r0 = self.r0
    phi = self.phi
    pressure = pg0*(x/r0)**(-phi)
    return pressure

  def pc(self, x):
    pc0 = self.pc0
    r0 = self.r0
    phi = self.phi
    cr_press = pc0*(x/r0)**(-phi)
    return cr_press

  def dpgdx(self, x):
    pg0 = self.pg0
    r0 = self.r0 
    phi = self.phi 
    grad = -(phi*pg0/r0)*(x/r0)**(-phi - 1.)
    return grad

  def dpcdx(self, x):
    pc0 = self.pc0
    r0 = self.r0
    phi = self.phi
    grad = -(phi*pc0/r0)*(x/r0)**(-phi - 1.)
    return grad

  def d2pcdx2(self, x):
    pc0 = self.pc0
    r0 = self.r0
    phi = self.phi
    grad2 = (phi*(phi + 1.)*pc0/r0**2)*(x/r0)**(-phi - 2.)
    return grad2 

  def powerprofile(self, xmin, xmax, grids, block=64, nghost=2, athena=False, read_in_file=None):
    kappa = self.kappa 
    rho0 = self.rho0
    b = self.b
    va0 = self.va0
    v0 = self.v0
    r0 = self.r0
    if read_in_file != None:
      read_in = True

    func = lambda x, vpva: (kappa*self.d2pcdx2(x) - vpva*self.dpcdx(x))/(gc*self.pc(x))

    dx = (xmax - xmin)/grids 
    x_eval = np.zeros(grids + 2*nghost)
    x_eval[0] = xmin - (nghost - 0.5)*dx 
    for i in np.arange(np.size(x_eval) - 1):
      x_eval[i+1] = x_eval[i] + dx 
    index = np.argmin(np.abs(x_eval - r0))
    index = index if (x_eval[index] > r0) else index + 1

    sol_front = integrate.solve_ivp(func, (r0, x_eval[-1]), [v0 + va0], t_eval=x_eval[index:])
    sol_back = integrate.solve_ivp(func, (r0, x_eval[0]), [v0 + va0], t_eval=x_eval[0:index][::-1])
    vpva_sol = np.append(sol_back.y[0][::-1], sol_front.y[0])

    # Find va from vpva_sol
    va_sol = np.zeros(np.size(vpva_sol)) 
    for i, vp_va in enumerate(vpva_sol):
      va_sol[i] = 0.5*(-va0/v0 + np.sqrt((va0/v0)**2 + 4.*vp_va/v0))*va0

    rho_sol = (b/va_sol)**2
    v_sol = vpva_sol - va_sol 
    pg_sol = self.pg(x_eval)
    pc_sol = self.pc(x_eval)
    fc_sol = gc1*pc_sol*(v_sol + va_sol) - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    dvdx_sol = -v_sol*(vpva_sol*self.dpcdx(x_eval) - kappa*self.d2pcdx2(x_eval))/(gc*pc_sol*(v_sol + 0.5*va_sol))
    g_sol = (rho0*v0*dvdx_sol + self.dpgdx(x_eval) + self.dpcdx(x_eval))/rho_sol
    avg_g = np.mean(g_sol)
    t_ff = np.sqrt(2.*self.Lc0/np.abs(avg_g))
    H_sol = (v_sol*self.dpgdx(x_eval) + gg*pg_sol*dvdx_sol + (gg - 1.)*va_sol*self.dpcdx(x_eval))/(gg - 1.)

    if read_in:
      data = ar.athdf(read_in_file)
      rho_sol[nghost:-nghost] = data['rho'][0, 0, :]
      v_sol[nghost:-nghost] = data['vel1'][0, 0, :]
      pg_sol[nghost:-nghost] = data['press'][0, 0, :]
      pc_sol[nghost:-nghost] = data['Ec'][0, 0, :]*(gc - 1.)
      fc_sol[nghost:-nghost] = data['Fc1'][0, 0, :]*self.vmax

    # Save data
    self.dx = x_eval[1] - x_eval[0]
    self.xmin = xmin
    self.xmax = xmax
    self.x_sol = x_eval
    self.va_sol = va_sol 
    self.rho_sol = rho_sol
    self.v_sol = v_sol
    self.dvdx_sol = dvdx_sol
    self.pg_sol = pg_sol
    self.pc_sol = pc_sol
    self.fc_sol = fc_sol 
    self.g_sol = g_sol
    self.avg_g = avg_g
    self.t_ff = t_ff
    self.H_sol = H_sol

    if athena:
      assert grids%block == 0, 'grids should be integral multiple of block'
      num_block = grids//block
      size_block = block
      block_id = np.arange(num_block)

      x_input = np.zeros((num_block, block+2*nghost))
      rho_input = np.zeros((num_block, block+2*nghost))
      v_input = np.zeros((num_block, block+2*nghost))
      pg_input = np.zeros((num_block, block+2*nghost))
      pc_input = np.zeros((num_block, block+2*nghost))
      fc_input = np.zeros((num_block, block+2*nghost))
      g_input = np.zeros((num_block, block+2*nghost))
      H_input = np.zeros(((num_block, block+2*nghost)))

      for i in np.arange(num_block):
        # Fill the active zones
        x_input[i, nghost:(nghost+block)] = self.x_sol[i*block+nghost:(i+1)*block+nghost]
        rho_input[i, nghost:(nghost+block)] = self.rho_sol[i*block+nghost:(i+1)*block+nghost] 
        v_input[i, nghost:(nghost+block)] = self.v_sol[i*block+nghost:(i+1)*block+nghost]
        pg_input[i, nghost:(nghost+block)] = self.pg_sol[i*block+nghost:(i+1)*block+nghost]
        pc_input[i, nghost:(nghost+block)] = self.pc_sol[i*block+nghost:(i+1)*block+nghost]
        fc_input[i, nghost:(nghost+block)] = self.fc_sol[i*block+nghost:(i+1)*block+nghost]
        g_input[i, nghost:(nghost+block)] = self.g_sol[i*block+nghost:(i+1)*block+nghost]
        H_input[i, nghost:(nghost+block)] = self.H_sol[i*block+nghost:(i+1)*block+nghost]

        # Fill the ghost zones
        x_input[i, 0:nghost] = self.x_sol[0:nghost]
        x_input[i, -nghost:] = self.x_sol[-nghost:]
        rho_input[i, 0:nghost] = self.rho_sol[0:nghost]
        rho_input[i, -nghost:] = self.rho_sol[-nghost:]
        pg_input[i, 0:nghost] = self.pg_sol[0:nghost]
        pg_input[i, -nghost:] = self.pg_sol[-nghost:]
        pc_input[i, 0:nghost] = self.pc_sol[0:nghost]
        pc_input[i, -nghost:] = self.pc_sol[-nghost:]
        fc_input[i, 0:nghost] = self.fc_sol[0:nghost]
        fc_input[i, -nghost:] = self.fc_sol[-nghost:]

      # Save data for athena input
      self.x_input = x_input 
      self.rho_input = rho_input 
      self.v_input = v_input
      self.pg_input = pg_input
      self.pc_input = pc_input 
      self.fc_input = fc_input 
      self.g_input = g_input
      self.H_input = H_input

    return

  def units(self, n_cgs, p_cgs, r_cgs):
    # n0 in cm^-3, pg0 in erg cm^-3, r0 in kpc
    kpc = 3.086e21 # cm
    kms = 1.e5
    mp = 1.667e-24 # g
    miu = 0.62 # relative atomic weight
    kb = 1.38e-16
    muG = 1.e-6
    myr = 1.e6*86400*365
    # First convert to cgs, operate then back to astro units
    # Basic units
    self.n_cgs = n_cgs # cm^-3
    self.rho_cgs = miu*n_cgs*mp # g cm^-3
    self.p_cgs = p_cgs # erg cm^-3
    self.r_cgs = r_cgs # kpc
    self.T_cgs = p_cgs/(n_cgs*kb) # K
    self.v_cgs = np.sqrt(p_cgs/self.rho_cgs)/kms # km s^-1
    self.t_cgs = self.r_cgs*kpc/(self.v_cgs*kms)/myr # Myr
    self.g_cgs = self.v_cgs*kms/(self.t_cgs*myr) # cm s^-1

    self.rho0_cgs = self.rho0*self.rho_cgs 
    self.pg0_cgs = self.pg0*self.p_cgs
    self.pc0_cgs = self.pc0*self.p_cgs 
    self.r0_cgs = self.r0*self.r_cgs 
    self.T0_cgs = self.T_cgs*self.pg0/self.rho0 
    self.cs0_cgs = self.cs0*self.v_cgs 
    self.b_cgs = np.sqrt(2.*self.pg0_cgs/self.beta)/muG 
    self.va0_cgs = self.b_cgs*muG/np.sqrt(self.rho_cgs)/kms
    self.v0_cgs = self.ms*self.cs0_cgs
    self.Lc0_cgs = self.Lc0*self.r_cgs 
    self.kappa_cgs = self.eta*gc*self.Lc0_cgs*kpc*self.cs0_cgs*kms
    self.avg_g_cgs = self.avg_g*self.g_cgs
    self.t_ff_cgs = self.t_ff*self.t_cgs 
    return 

# End class
##########################################

# Problem
alpha = 1.
beta = 1.
eta = 0.01
ms = 0.08
psi = 0.5

rho0 = 1. 
pg0 = 1. 
r0 = 1.

vmax = 200.
data_file = '../sims/power.out1.10000.athdf'

prob = Power(alpha, beta, eta, ms, psi, rho0, pg0, r0, vmax)
prob.powerprofile(r0, 10.*r0, 4096, block=64, nghost=2, athena=True, read_in_file=data_file)

# Real units
n0_cgs = 0.01 # cm^-2 
pg0_cgs = 5.e-13 # erg cm^-3 
r0_cgs = 10. # kpc

prob.units(n0_cgs, pg0_cgs, r0_cgs)

# Plot profile
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(prob.x_sol/prob.r0, prob.pg_sol, label='$P_g$')
ax1.plot(prob.x_sol/prob.r0, prob.pc_sol, label='$P_c$')
ax1.plot(prob.x_sol/prob.r0, prob.rho_sol, label='$\\rho$')
ax1.plot(prob.x_sol/prob.r0, prob.v_sol, label='$v$')
ax1.set_xlabel('$x/r_0$')
ax1.set_ylabel('$P$, $\\rho$, $v$')
ax1.legend(loc='center left', frameon=False)
ax1.margins(x=0)
fig.tight_layout()
fig.savefig('./power.png', dpi=300)
plt.show()

plt.close('all')

# Compare terms in momentum equation
dpdx = (prob.rho0*prob.v0*prob.dvdx_sol + prob.dpgdx(prob.x_sol) + prob.dpcdx(prob.x_sol))
rhog = prob.rho_sol*prob.g_sol
dpvdx = prob.v_sol*prob.dpgdx(prob.x_sol) + gg*prob.pg_sol*prob.dvdx_sol 
heat = (-prob.va_sol*prob.dpcdx(prob.x_sol) + prob.H_sol)*(gg - 1.)

fig2 = plt.figure()
fig3 = plt.figure()
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)

ax2.plot(prob.x_sol/prob.r0, dpdx - rhog, label='mom_leftover')
ax3.plot(prob.x_sol/prob.r0, dpvdx - heat, label='eng_leftover')

ax2.margins(x=0)
ax2.legend(frameon=False)
ax2.set_xlabel('$x/r_0$')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax3.margins(x=0)
ax3.legend(frameon=False)
ax3.set_xlabel('$x/r_0$')
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())

fig2.tight_layout()
fig3.tight_layout()
plt.show()

plt.close('all')

# Save athena data
with h5py.File('./stop.hdf5', 'w') as fp:
  dset = fp.create_dataset('x', data=prob.x_input)
  dset = fp.create_dataset('rho', data=prob.rho_input)
  dset = fp.create_dataset('v', data=prob.v_input)
  dset = fp.create_dataset('pg', data=prob.pg_input)
  dset = fp.create_dataset('ec', data=prob.pc_input/(gc - 1.))
  dset = fp.create_dataset('fc', data=prob.fc_input)
  dset = fp.create_dataset('g', data=prob.g_input)
  dset = fp.create_dataset('H', data=prob.H_input)
  fp.attrs.create('B', prob.b)
  fp.attrs.create('kappa', prob.kappa)
  fp.attrs.create('ldiff', prob.ldiff)


#################################
# For publication


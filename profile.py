import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import ticker, cm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import scipy.integrate as integrate 
import scipy.optimize as opt
import h5py 

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
  def __init__(self, alpha, beta, eta, psi=0., rho0=1., pg0=1., r0=1.):
    self.alpha = alpha # pc_h/pg_h
    self.beta = beta # 2*pg_h/b^2
    self.eta = eta # kappa/gc L c_s
    self.psi = psi 
    self.phi = (gc/(2. - gc))*(1. - psi)
    self.rho0 = rho0 
    self.pg0 = pg0 
    self.r0 = r0 
    self.pc0 = alpha*pg0 
    self.b = np.sqrt(2.*pg0/beta)
    self.cs0 = np.sqrt(gg*pg0/rho0)
    self.cc0 = np.sqrt(gc*self.pc0/rho0)
    self.va0 = self.b/np.sqrt(rho0)
    self.Lc0 = r0/self.phi
    self.kappa = eta*gc*self.Lc0*self.cs0 
    self.ldiff = self.kappa/self.cs0
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

  def rho_st(self, x):
    rho0 = self.rho0 
    pc0 = self.pc0 
    rho = rho0*(self.pc(x)/pc0)**(2./gc)
    return rho

  def powerprofile(self, xmin, xmax, grids, block=64, nghost=2, athena=False, stream_only=False):
    kappa = self.kappa 
    b = self.b
    va0 = self.va0
    r0 = self.r0

    func = lambda x, va: (kappa*self.d2pcdx2(x) - va*self.dpcdx(x))/(gc*self.pc(x))

    dx = (xmax - xmin)/grids 
    x_eval = np.zeros(grids + 2*nghost)
    x_eval[0] = xmin - (nghost - 0.5)*dx 
    for i in np.arange(np.size(x_eval) - 1):
      x_eval[i+1] = x_eval[i] + dx 
    index = np.argmin(np.abs(x_eval - r0))
    index = index if (x_eval[index] > r0) else index + 1

    sol_front = integrate.solve_ivp(func, (r0, x_eval[-1]), [va0], t_eval=x_eval[index:])
    sol_back = integrate.solve_ivp(func, (r0, x_eval[0]), [va0], t_eval=x_eval[0:index][::-1])
    va_sol = np.append(sol_back.y[0][::-1], sol_front.y[0])

    rho_sol = (b/va_sol)**2
    pg_sol = self.pg(x_eval)
    pc_sol = self.pc(x_eval)
    fc_sol = gc1*pc_sol*va_sol - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    g_sol = (self.dpgdx(x_eval) + self.dpcdx(x_eval))/rho_sol
    H_sol = va_sol*self.dpcdx(x_eval)

    # Save data
    self.dx = x_eval[1] - x_eval[0]
    self.xmin = xmin
    self.xmax = xmax
    self.x_sol = x_eval
    self.va_sol = va_sol 
    self.rho_sol = rho_sol
    self.pg_sol = pg_sol
    self.pc_sol = pc_sol
    self.fc_sol = fc_sol 
    self.g_sol = g_sol
    self.H_sol = H_sol

    # For streaming only, analytic solution is possible
    self.stream_only = stream_only
    self.rho_str = self.rho_st(x_eval)
    self.va_str = self.b/np.sqrt(self.rho_str)
    self.fc_str = gc1*pc_sol*self.va_str
    # self.fc_str = gc1*pc_sol*self.va_str - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    self.g_str = (self.dpgdx(x_eval) + self.dpcdx(x_eval))/self.rho_str 
    self.H_str = self.va_str*self.dpcdx(x_eval)

    if athena:
      assert grids%block == 0, 'grids should be integral multiple of block'
      num_block = grids//block
      size_block = block
      block_id = np.arange(num_block)

      x_input = np.zeros((num_block, block+2*nghost))
      rho_input = np.zeros((num_block, block+2*nghost))
      pg_input = np.zeros((num_block, block+2*nghost))
      pc_input = np.zeros((num_block, block+2*nghost))
      fc_input = np.zeros((num_block, block+2*nghost))
      g_input = np.zeros((num_block, block+2*nghost))
      H_input = np.zeros(((num_block, block+2*nghost)))

      for i in np.arange(num_block):
        # Fill the active zones
        if stream_only:
          x_input[i, nghost:(nghost+block)] = self.x_sol[i*block+nghost:(i+1)*block+nghost]
          rho_input[i, nghost:(nghost+block)] = self.rho_str[i*block+nghost:(i+1)*block+nghost] 
          pg_input[i, nghost:(nghost+block)] = self.pg_sol[i*block+nghost:(i+1)*block+nghost]
          pc_input[i, nghost:(nghost+block)] = self.pc_sol[i*block+nghost:(i+1)*block+nghost]
          fc_input[i, nghost:(nghost+block)] = self.fc_str[i*block+nghost:(i+1)*block+nghost]
          g_input[i, nghost:(nghost+block)] = self.g_str[i*block+nghost:(i+1)*block+nghost]
          H_input[i, nghost:(nghost+block)] = self.H_str[i*block+nghost:(i+1)*block+nghost]
        else:
          x_input[i, nghost:(nghost+block)] = self.x_sol[i*block+nghost:(i+1)*block+nghost]
          rho_input[i, nghost:(nghost+block)] = self.rho_sol[i*block+nghost:(i+1)*block+nghost] 
          pg_input[i, nghost:(nghost+block)] = self.pg_sol[i*block+nghost:(i+1)*block+nghost]
          pc_input[i, nghost:(nghost+block)] = self.pc_sol[i*block+nghost:(i+1)*block+nghost]
          fc_input[i, nghost:(nghost+block)] = self.fc_sol[i*block+nghost:(i+1)*block+nghost]
          g_input[i, nghost:(nghost+block)] = self.g_sol[i*block+nghost:(i+1)*block+nghost]
          H_input[i, nghost:(nghost+block)] = self.H_sol[i*block+nghost:(i+1)*block+nghost]

        # Fill the ghost zones
        if stream_only:
          x_input[i, 0:nghost] = self.x_sol[0:nghost]
          x_input[i, -nghost:] = self.x_sol[-nghost:]
          rho_input[i, 0:nghost] = self.rho_str[0:nghost]
          rho_input[i, -nghost:] = self.rho_str[-nghost:]
          pg_input[i, 0:nghost] = self.pg_sol[0:nghost]
          pg_input[i, -nghost:] = self.pg_sol[-nghost:]
          pc_input[i, 0:nghost] = self.pc_sol[0:nghost]
          pc_input[i, -nghost:] = self.pc_sol[-nghost:]
          fc_input[i, 0:nghost] = self.fc_str[0:nghost]
          fc_input[i, -nghost:] = self.fc_str[-nghost:]
        else:
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
      self.pg_input = pg_input
      self.pc_input = pc_input 
      self.fc_input = fc_input 
      self.g_input = g_input
      self.H_input = H_input

    return

# End class
##########################################

# Problem
alpha = 1.
beta = 1.
eta = 1.e-2
psi = 0.

rho0 = 1. 
pg0 = 1. 
r0 = 1.

prob = Power(alpha, beta, eta, psi, rho0, pg0, r0)
prob.powerprofile(r0, 2.*r0, 16384, block=256, nghost=2, athena=True, stream_only=False)

# Plot profile
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(prob.x_sol/prob.r0, prob.pg_sol, label='$P_g$')
ax1.plot(prob.x_sol/prob.r0, prob.pc_sol, label='$P_c$')
ax1.plot(prob.x_sol/prob.r0, prob.rho_sol, label='$\\rho$')
ax1.set_xlabel('$x/r_0$')
ax1.set_ylabel('$P$, $\\rho$')
ax1.legend(loc='center left', frameon=False)
ax1.margins(x=0)
fig.tight_layout()
fig.savefig('./power.png', dpi=300)
plt.show()

plt.close('all')

# Compare analytic profile with real profile
dfcdx = np.diff(prob.fc_sol)/prob.dx 
src = prob.va_sol*prob.dpcdx(prob.x_sol)
dfcdx_str = np.diff(prob.fc_str)/prob.dx 
src_str = prob.va_str*prob.dpcdx(prob.x_sol)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(prob.x_sol[:-1]/prob.r0, dfcdx, 'r', label='num dfcdx')
ax1.plot(prob.x_sol[:-1]/prob.r0, dfcdx_str, 'b', label='str dfcdx')
ax1.plot(prob.x_sol/prob.r0, src, 'r--', label='num src')
ax1.plot(prob.x_sol/prob.r0, src_str, 'b--', label='str src')

ax1.margins(x=0)
ax1.legend(frameon=False)
ax1.set_xlabel('$x/r_0$')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

fig1.tight_layout()
plt.show()

plt.close('all')

# Compare terms in momentum equation
dpdx = (prob.dpgdx(prob.x_sol) + prob.dpcdx(prob.x_sol))
rhog = prob.rho_sol*prob.g_sol

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(prob.x_sol/prob.r0, dpdx - rhog, label='mom_leftover')
# ax2.plot(prob.x_sol/prob.r0, rhog, label='rhog')

ax2.margins(x=0)
ax2.legend(frameon=False)
ax2.set_xlabel('$x/r_0$')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

fig2.tight_layout()
plt.show()

plt.close('all')

# Save athena data
with h5py.File('./power.hdf5', 'w') as fp:
  dset = fp.create_dataset('x', data=prob.x_input)
  dset = fp.create_dataset('rho', data=prob.rho_input)
  dset = fp.create_dataset('pg', data=prob.pg_input)
  dset = fp.create_dataset('ec', data=prob.pc_input/(gc - 1.))
  dset = fp.create_dataset('fc', data=prob.fc_input)
  dset = fp.create_dataset('g', data=prob.g_input)
  dset = fp.create_dataset('H', data=prob.H_input)
  fp.attrs.create('B', prob.b)
  fp.attrs.create('kappa', prob.kappa)
  fp.attrs.create('ldiff', prob.ldiff)

if prob.stream_only:
  with h5py.File('./power_analysis.hdf5', 'w') as fp:
    dset = fp.create_dataset('x', data=prob.x_sol)
    dset = fp.create_dataset('rho', data=prob.rho_str)
    dset = fp.create_dataset('pg', data=prob.pg_sol)
    dset = fp.create_dataset('pc', data=prob.pc_sol)
    dset = fp.create_dataset('dpcdx', data=prob.dpcdx(prob.x_sol))
    dset = fp.create_dataset('g', data=prob.g_str)
    dset = fp.create_dataset('H', data=prob.H_str)
    fp.attrs.create('B', prob.b)
    fp.attrs.create('kappa', prob.kappa)
else:
  with h5py.File('./power_analysis.hdf5', 'w') as fp:
    dset = fp.create_dataset('x', data=prob.x_sol)
    dset = fp.create_dataset('rho', data=prob.rho_sol)
    dset = fp.create_dataset('pg', data=prob.pg_sol)
    dset = fp.create_dataset('pc', data=prob.pc_sol)
    dset = fp.create_dataset('dpcdx', data=prob.dpcdx(prob.x_sol))
    dset = fp.create_dataset('g', data=prob.g_sol)
    dset = fp.create_dataset('H', data=prob.H_sol)
    fp.attrs.create('B', prob.b)
    fp.attrs.create('kappa', prob.kappa)



#################################
# For publication

# # Amount of CR needed
# num_q = 200
# num_T = 200
# q = np.logspace(0.01, 2, num_q)
# chi_T = np.logspace(0.01, 2, num_T)
# alpha = np.zeros((num_q, num_T))

# for i, qq in enumerate(q):
#   for j, cT in enumerate(chi_T):
#     if (cT > 1.01*qq):
#       alpha[i, j] = qq**(gc/2.)*(1. - 1./qq)/(cT**(gc/2.) - qq**(gc/2.))
#     else:
#       alpha[i, j] = -1

# alpha = np.ma.masked_where(alpha <= 0, alpha)

# # latexify(columns=1, square=False)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)

# cont_alpha = ax1.contourf(q, chi_T, np.transpose(alpha), locator=ticker.LogLocator())
# fig1.colorbar(cont_alpha, ax=ax1, label='$\\alpha$')
# ax1.plot(q, q, 'k--')

# ax1.set_xlabel('$q$')
# ax1.set_ylabel('$\\chi_{T}$')
# ax1.set_xscale('log')
# ax1.set_yscale('log')

# fig1.tight_layout()
# fig1.savefig('./cr_content.png', dpi=300)
# plt.show()

# plt.close('all')
# # plotdefault()




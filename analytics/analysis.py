# Import installed packages
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import matplotlib.legend as lg
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy import interpolate
import h5py

# Matplotlib default param
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 2.0
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

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

# Global parameters
gc = 4./3.
gg = 5./3.
gc1 = gc/(gc - 1.)
gg1 = gg/(gg - 1.)
big_number = 1e20
noise_level = 1.e-16
big_level = 1e200



class Acous1d:
  def __init__(self, equil, forward=True):
    # equil consists of [rho, pg, pc, kappa, b, dpcdx, cool_eff, cool_index]
    self.forward = forward
    self.rho = equil['rho']
    self.pg = equil['pg']
    self.pc = equil['pc']
    self.b = equil['b']
    self.kappa = equil['kappa']
    self.dpcdx = equil['dpcdx']
    self.dcsdx = equil['dcsdx']
    self.cs = np.sqrt(gg*self.pg/self.rho)
    self.cc = np.sqrt(gc*self.pc/self.rho)
    self.vs = -np.sign(self.dpcdx)*self.b/np.sqrt(self.rho)
    self.ldiff = self.kappa/self.cs
    self.Lc = -self.pc/self.dpcdx if self.dpcdx != 0 else big_number
    self.Lcs = -self.cs/self.dcsdx if self.dcsdx != 0 else big_number
  # End initialization

  def growthrate(self):
    rho = self.rho 
    cs = self.cs 
    cc = self.cc 
    vs = self.vs 
    kappa = self.kappa 
    Lc = self.Lc 
    Lcs = self.Lcs 

    if self.forward:
      acous1 = (1. + (gg - 1.)*vs/cs)*(1. - 0.5*vs/cs)
      acous2 = (kappa/(gc*Lc*cs))*(1. + 0.5*(gg - 1.)*vs/cs) 
      acous = -0.5*(cc**2/kappa)*(acous1 + acous2) + 0.5*cs/Lcs
    else:
      acous1 = (1. - (gg - 1.)*vs/cs)*(1. + 0.5*vs/cs)
      acous2 = (kappa/(gc*Lc*cs))*(1. - 0.5*(gg - 1.)*vs/cs)
      acous = -0.5*(cc**2/kappa)*(acous1 - acous2) - 0.5*cs/Lcs

    return acous/(cc**2/kappa)

# End class Acous1d



class AcousProfile:
  def __init__(self, profile, forward=True):
    # profile consists of arrays of [x, rho, pg, pc, kappa, b, dpcdx]
    # cooling consists of arrays of [cool_temp, cool_eff, cool_index]
    self.forward = forward
    self.profile = profile 
    self.x = profile['x']
  # End initialization

  def plot(self):
    x = self.x
    growth = np.zeros(np.size(x))
    growth_normalized = np.zeros(np.size(x))
    sound_speed = np.zeros(np.size(x))

    for i, value in enumerate(x):
      equi = dict.fromkeys(['rho', 'pg', 'pc', 'kappa', 'b', 'dpcdx'])

      for key in self.profile.keys():
        if (key != 'x'):
          equi[key] = self.profile[key][i]

      acous = Acous1d(equi, self.forward)
      sound_speed[i] = acous.cs
      growth[i] = acous.growthrate()*(acous.cc**2/acous.kappa)
      growth_normalized[i] = acous.growthrate()

    # Save growth data
    self.sound_speed = sound_speed
    self.growth = growth

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(self.x, growth_normalized)

    ax1.margins(x=0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$\\Gamma/c_c^2/\\kappa$')

    fig.tight_layout()
    return fig

  def amplitude(self, amp0=1.e-5, log=True):
    x = self.x 
    dx = x[1] - x[0] # Uniform grid
    rho = self.profile['rho']
    growth = self.growth 
    sound_speed = self.sound_speed 
    amp = np.zeros(np.size(x))
    amp_genuine = np.zeros(np.size(x))

    # Integrate
    if self.forward:
      amp[0] = amp0
      amp_genuine[0] = amp0
      for i in np.arange(1, np.size(x)):
        amp[i] = amp0*np.exp(0.5*np.log(rho[0]/rho[i]) + np.sum(growth[:(i+1)]*dx/sound_speed[:(i+1)]))
        amp_genuine[i] = amp0*np.exp(np.sum(growth[:(i+1)]*dx/sound_speed[:(i+1)]))
        if (amp[i] < noise_level):
          amp[i] = amp[i-1]
        if (amp_genuine[i] < noise_level):
          amp_genuine[i] = amp_genuine[i-1]
        if (amp[i] > big_level):
          amp[i] = 10*big_level
        if (amp_genuine[i] > big_level):
          amp_genuine[i] = 10*big_level
    else:
      amp[-1] = amp0 
      amp_genuine[-1] = amp0
      for i in np.arange(1, np.size(x)):
        amp[-(i+1)] = amp0*np.exp(0.5*np.log(rho[-1]/rho[-(i+1)]) + np.sum(growth[::-1][:(i+1)]*dx/sound_speed[::-1][:(i+1)]))
        amp_genuine[-(i+1)] = amp0*np.exp(np.sum(growth[::-1][:(i+1)]*dx/sound_speed[::-1][:(i+1)]))
        if (amp[-(i+1)] < noise_level):
          amp[-(i+1)] = amp[-i]
        if (amp_genuine[-(i+1)] < noise_level):
          amp_genuine[-(i+1)] = amp_genuine[-i]
        if (amp[-(i+1)] > big_level):
          amp[-(i+1)] = 10*big_level
        if (amp_genuine[-(i+1)] > big_level):
          amp_genuine[-(i+1)] = 10*big_level
    # Save data
    self.amp = amp 

    # Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    if log:
      ax1.semilogy(self.x, np.ma.masked_where(amp > big_level, amp), label='Adiabatic+Genuine')
      ax1.semilogy(self.x, np.ma.masked_where(amp_genuine > big_level, amp_genuine), '--', label='Genuine')
    else:
      ax1.plot(self.x, np.ma.masked_where(amp > big_level, amp), label='Adiabatic+Genuine')
      ax1.plot(self.x, np.ma.masked_where(amp_genuine > big_level, amp_genuine), '--', label='Genuine')

    ax1.legend(frameon=False)
    ax1.margins(x=0)
    if not log:
      ax1.xaxis.set_minor_locator(AutoMinorLocator())
      ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$v$')

    fig.tight_layout()
    return fig
  
# End of class AcousProfile

###################################################

# As function of x
# Parameters
with h5py.File('./power_analysis.hdf5', 'r') as fp:
  x = np.array(fp['x'])
  rho = np.array(fp['rho'])
  pg = np.array(fp['pg'])
  pc = np.array(fp['pc'])
  b = fp.attrs['B']
  kappa = fp.attrs['kappa']
  dpcdx = np.array(fp['dpcdx'])
  dcsdx = np.array(fp['dcsdx'])

# Profile
profile = {
  'x': x, 
  'rho': rho, 
  'pg': pg, 
  'pc': pc, 
  'kappa': kappa*np.ones(np.size(x)), 
  'b': b*np.ones(np.size(x)),
  'dpcdx': dpcdx, 
  'dcsdx': dcsdx
}

prof = AcousProfile(profile, forward=True) 
fig = prof.plot()
fig.savefig('/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/power/results/analytics/profile.png', dpi=300)
fig1 = prof.amplitude(amp0=1.29e-5, log=True)
np.savetxt('/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/power/results/analytics/x.csv', prof.x)
np.savetxt('/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/power/results/analytics/amp.csv', prof.amp)
plt.show()
plt.close('all')


    
################################
# Publication plots


# Import installed packages
import sys
import numpy as np 
import numpy.linalg as LA
import numpy.ma as ma
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import matplotlib.legend as lg
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import scipy.stats as stats
import scipy.optimize as opt
import scipy.integrate as integrate 
from scipy import interpolate
import h5py

# Import athena data reader
sys.path.append('/Users/tsunhinnavintsung/Workspace/Codes/Athena++')
import athena_read3 as ar 

# Matplotlib default param
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['lines.linewidth'] = 1.
plt.rcParams['lines.markersize'] = 0.7
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
tiny_number = 1.e-10
big_number = 1.e20

class Plot1d:
  def __init__(self, inputfile, file_array, video=False, staircase=False, history=False, profile_in=None):
    # Staircase, history are specific to this problem only
    self.inputfile = inputfile
    self.file_array = file_array
    self.isothermal = True
    self.cr = False
    self.b_field = False
    self.passive = False
    self.open_file()

    # Check what physics is present
    if 'press'.encode('utf-8') in ar.athdf('./' + self.filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['VariableNames']:
      self.isothermal = False
    if 'Ec'.encode('utf-8') in ar.athdf('./' + self.filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['VariableNames']:
      self.cr = True
    if 'Bcc1'.encode('utf-8') in ar.athdf('./' + self.filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['VariableNames']:
      self.b_field = True
    if 'r0'.encode('utf-8') in ar.athdf('./' + self.filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['VariableNames']:
      self.passive = True

    if not profile_in == None:
      self.profile_object = Power(profile_in)
      self.profile_object.powerprofile(self.x1min, self.x1max, self.nx1)
    if video or staircase or history:
      pass
    else:
      self.read_data()
    self.runpan = False
    self.runshock = False # Specific to this problem only

  def is_number(self, n):
    try:
      float(n)   
    except ValueError:
      return False
    return True

  def open_file(self):
    with open(self.inputfile, 'r') as fp:
      line = fp.readline()
      self.nx1 = 0
      self.user_output = False
      while line:
        phrase = line.strip().split(' ')
        for word in phrase:
          if word == 'problem_id':
            self.filename = line.split('=')[1].strip().split(' ')[0]
          elif word == 'nx1':
            grid = int(line.split('=')[1].strip().split(' ')[0])
            self.nx1 = grid if grid > self.nx1 else self.nx1 
          elif word == 'x1min':
            for element in phrase:
              if self.is_number(element):
                self.x1min = float(element)
          elif word == 'x1max':
            for element in phrase:
              if self.is_number(element):
                self.x1max = float(element)
          elif word == 'dt':
            for element in phrase:
              if self.is_number(element):
                self.dt = float(element)
          elif word == 'gamma':
            for element in phrase:
              if self.is_number(element):
                gamma = float(element)
                if gamma != gg:
                  globals()[gg] = gamma
                  globals()[gg1] = gamma/(gamma - 1.)
                self.isothermal = False 
          elif word == 'iso_sound_speed':
            for element in phrase:
              if self.is_number(element):
                self.cs_iso = float(element)
          elif word == 'uov':
            self.user_output = True
          elif word == 'vs_flag':
            for element in phrase:
              if self.is_number(element):
                self.cr_stream = True if int(element) == 1 else False
          elif word == 'gamma_c':
            for element in phrase:
              if self.is_number(element):
                gamma_c = float(element) 
                if gamma_c != gc:
                  globals()[gc] = gamma_c 
                  globals()[gc1] = gamma_c/(gamma_c - 1.)
          elif word == 'vmax':
            for element in phrase:
              if self.is_number(element):
                self.vmax = float(element) 
          else:
            continue  
        line = fp.readline()
    return 
      
  def read_data(self):
    filename = self.filename 

    # Choosing files for display
    file_array = self.file_array 
    self.time_array = np.zeros(np.size(file_array))

    # For no adaptive mesh refinement
    self.x1v = np.array(ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v'])
    x1v = self.x1v
    self.dx = x1v[1] - x1v[0]

    # Number of parameters of interest
    self.rho_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.vx_array = np.zeros((np.size(file_array), np.size(x1v)))
    if not self.isothermal:
      self.pg_array = np.zeros((np.size(file_array), np.size(x1v)))
    if self.cr:
      self.ecr_array = np.zeros((np.size(file_array), np.size(x1v)))
      self.fcx_array = np.zeros((np.size(file_array), np.size(x1v)))
      self.vs_array = np.zeros((np.size(file_array), np.size(x1v)))
      self.sigma_adv_array = np.zeros((np.size(file_array), np.size(x1v)))
      self.sigma_diff_array = np.zeros((np.size(file_array), np.size(x1v)))
    if self.b_field:
      self.bx_array = np.zeros((np.size(file_array), np.size(x1v)))
    if self.passive:
      self.passive = True
      self.r_array = np.zeros((np.size(file_array), np.size(x1v)))

    # Preparing for uov
    if self.user_output:
      self.uovnum = ar.athdf('./' + filename + '.out2.' + format(file_array[0], '05d') + '.athdf')['NumVariables'][0]
      self.uovname = ar.athdf('./' + filename + '.out2.' + format(file_array[0], '05d') + '.athdf')['VariableNames']
      self.uovname = [self.uovname[i].decode('utf-8') for i in range(self.uovnum)] 
      self.uov_array = np.zeros((np.size(file_array), self.uovnum, np.size(x1v)))

    # Extracting data
    for i, file in enumerate(file_array):
      data = ar.athdf('./' + filename + '.out1.' + format(file, '05d') \
        + '.athdf')
      self.time_array[i] = float('{0:f}'.format(data['Time']))
      self.rho_array[i, :] = data['rho'][0, 0, :]
      self.vx_array[i, :] = data['vel1'][0, 0, :]
      if not self.isothermal:
        self.pg_array[i, :] = data['press'][0, 0, :]
      if self.cr:
        self.ecr_array[i, :] = data['Ec'][0, 0, :] 
        self.fcx_array[i, :] = data['Fc1'][0, 0, :]
        self.vs_array[i, :] = data['Vc1'][0, 0, :]
        self.sigma_adv_array[i, :] = data['Sigma_adv1'][0, 0, :]
        self.sigma_diff_array[i, :] = data['Sigma_diff1'][0, 0, :]
      if self.b_field:
        self.bx_array[i, :] = data['Bcc1'][0, 0, :]
      if self.passive:
        self.r_array[i, :] = data['r0'][0, 0, :]

      if self.user_output:
        uov_data = ar.athdf('./' + filename + '.out2.' + format(file, '05d') \
          + '.athdf')
        for j, uov_name in enumerate(self.uovname):
          self.uov_array[i, j, :] = uov_data[uov_name][0, 0, :]
      
    # For constant kappa and magnetic field
    if self.cr:
      self.kappa = (gc - 1.)*self.vmax/self.sigma_diff_array[0, 0] 
    if self.b_field:
      self.b0 = self.bx_array[0, 0]
    return 

  # Simplified data reader for making video
  # Input: equi, background to be subtracted
  def make_video(self, equi, save_path):
    filename = self.filename 

    # Choosing files for display
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))

    # For no adaptive mesh refinement
    x1v = np.array(ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v'])
    dx = x1v[1] - x1v[0]

    # Equilibrium profile
    rho_eq = equi['rho']
    if not self.isothermal:
      pg_eq = equi['pg']
    if self.cr:
      pc_eq = equi['pc']

    # Number of parameters of interest
    rho_array = np.zeros(np.size(x1v))
    if not self.isothermal:
      pg_array = np.zeros(np.size(x1v))
    if self.cr:
      ecr_array = np.zeros(np.size(x1v))
    if self.passive:
      r_array = np.zeros(np.size(x1v))

    # Extracting data
    for i, file in enumerate(file_array):
      print(file)
      data = ar.athdf('./' + filename + '.out1.' + format(file, '05d') \
        + '.athdf')
      time = float('{0:f}'.format(data['Time']))
      rho_array = data['rho'][0, 0, :]
      if not self.isothermal:
        pg_array = data['press'][0, 0, :]
      if self.cr:
        ecr_array = data['Ec'][0, 0, :]
      if self.passive:
        r_array = data['r0'][0, 0, :]

      # Plot and save image 
      if (self.isothermal and (not self.cr)):
        fig = plt.figure(figsize=(4, 4))
        grids = gs.GridSpec(1, 1, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        lab = ['$\\rho$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
      elif ((not self.isothermal) and (not self.cr)):
        fig = plt.figure(figsize=(8, 4))
        grids = gs.GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        lab = ['$\\rho$', '$P_g$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, pg_array - pg_eq, label='t={:.3f}'.format(time))
      elif (self.isothermal and self.cr):
        fig = plt.figure(figsize=(8, 4))
        grids = gs.GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        lab = ['$\\rho$', '$P_c$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, ecr_array/3. - pc_eq, label='t={:.3f}'.format(time))
      elif ((not self.isothermal) and self.cr):
        fig = plt.figure(figsize=(12, 4))
        grids = gs.GridSpec(1, 3, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        ax3 = fig.add_subplot(grids[0, 2])
        lab = ['$\\rho$', '$P_c$', '$P_g$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, ecr_array/3. - pc_eq, label='t={:.3f}'.format(time))
        ax3.plot(x1v, pg_array - pg_eq, label='t={:.3f}'.format(time))
      else:
        pass

      if self.passive:
        fig_passive = plt.figure()
        ax_passive = fig_passive.add_subplot(111) 
        ax_passive.plot(x1v, r_array, label='t={:.3f}'.format(time))

      for i, axes in enumerate(fig.axes):
        axes.legend(frameon=False)
        axes.set_xlabel('$x$')
        axes.set_ylabel(lab[i])
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

      if self.passive:
        ax_passive.legend(frameon=False)
        ax_passive.set_xlabel('$x$')
        ax_passive.set_ylabel('$r$')
        ax_passive.xaxis.set_minor_locator(AutoMinorLocator())
        ax_passive.yaxis.set_minor_locator(AutoMinorLocator())
        fig_passive.tight_layout()

      fig.tight_layout()
      video_path = save_path + 'gas_video{}.png'.format(file)
      fig.savefig(video_path, dpi=300)
      if self.passive:
        video_passive_path = save_path + 'passive_video{}.png'.format(file)
        fig_passive.savefig(video_passive_path, dpi=300)
      plt.close('all')
    return 

  def plot(self):
    file_array = self.file_array 
    time_array = self.time_array
    x1v = self.x1v
    if self.cr:
      vmax = self.vmax 

    fig = plt.figure(figsize=(12, 8))
    grids = gs.GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[0, 1])
    ax = [ax1, ax2]
    lab = ['$\\rho$', '$v_x$']
    if not self.isothermal:
      ax3 = fig.add_subplot(grids[0, 2])
      ax.append(ax3)
      lab.append('$P_g$')
    if self.cr:
      ax4 = fig.add_subplot(grids[1, 0])
      ax5 = fig.add_subplot(grids[1, 1])
      ax6 = fig.add_subplot(grids[1, 2])
      ax7 = fig.add_subplot(grids[2, 0])
      ax.append(ax4)
      ax.append(ax5)
      ax.append(ax6)
      ax.append(ax7)
      lab.append('$P_c$')
      lab.append('$F_c$')
      lab.append('$v_{s}$')
      lab.append('$\\sigma_c$')
    if self.b_field:
      ax8 = fig.add_subplot(grids[2, 1])
      ax9 = fig.add_subplot(grids[2, 2])
      ax.append(ax8)
      ax.append(ax9)
      lab.append('$\\beta$')
      lab.append('$P_g/\\rho^{\\gamma_g}$')

    if self.passive:
      fig_passive = plt.figure()
      ax_passive = fig_passive.add_subplot(111)

    for i, file in enumerate(file_array):
      if file == 0:
        ax1.plot(x1v, self.rho_array[i, :], 'k--',  label='t={:.3f}'.format(time_array[i]))
        ax2.plot(x1v, self.vx_array[i, :], 'k--', label='t={:.3f}'.format(time_array[i]))
        if not self.isothermal:
          ax3.plot(x1v, self.pg_array[i, :], 'k--', label='t={:.3f}'.format(time_array[i]))
        if self.cr:
          ax4.plot(x1v, self.ecr_array[i, :]/3., 'k--',  label='t={:.3f}'.format(time_array[i]))
          ax5.plot(x1v, self.fcx_array[i, :]*vmax, 'k--', label='t={:.3f}'.format(time_array[i]))
          ax6.plot(x1v, self.vs_array[i, :], 'k--',  label='t={:.3f}'.format(time_array[i]))
          ax7.semilogy(x1v, self.sigma_adv_array[i, :]/(self.sigma_adv_array[i, :]/self.sigma_diff_array[i, :] + 1.)/vmax, 'k--', label='t={:.3f}'.format(time_array[i]))
        if self.b_field:
          ax8.plot(x1v, 2.*self.pg_array[i, :]/self.b0**2, 'k--', label='t={:.3f}'.format(time_array[i]))
          ax9.plot(x1v, self.pg_array[i, :]/self.rho_array[i, :]**(gg), 'k--', label='t={:.3f}'.format(time_array[i]))
        if self.passive:
          ax_passive.plot(x1v, self.r_array[i, :], 'k--', label='t={:3f}'.format(time_array[i]))
      else:
        ax1.plot(x1v, self.rho_array[i, :], 'o-',  label='t={:.3f}'.format(time_array[i]))
        ax2.plot(x1v, self.vx_array[i, :], 'o-', label='t={:.3f}'.format(time_array[i]))
        if not self.isothermal:
          ax3.plot(x1v, self.pg_array[i, :], 'o-', label='t={:.3f}'.format(time_array[i]))
        if self.cr:
          ax4.plot(x1v, self.ecr_array[i, :]/3., 'o-',  label='t={:.3f}'.format(time_array[i]))
          ax5.plot(x1v, self.fcx_array[i, :]*vmax, 'o-', label='t={:.3f}'.format(time_array[i]))
          ax6.plot(x1v, self.vs_array[i, :], 'o-',  label='t={:.3f}'.format(time_array[i]))
          ax7.semilogy(x1v, self.sigma_adv_array[i, :]/(self.sigma_adv_array[i, :]/self.sigma_diff_array[i, :] + 1.)/vmax, 'o-', label='t={:.3f}'.format(time_array[i]))
        if self.b_field:
          ax8.plot(x1v, 2.*self.pg_array[i, :]/self.b0**2, 'o-', label='t={:.3f}'.format(time_array[i]))
          ax9.plot(x1v, self.pg_array[i, :]/self.rho_array[i, :]**(gg), 'o-', label='t={:.3f}'.format(time_array[i]))
        if self.passive:
          ax_passive.plot(x1v, self.r_array[i, :], label='t={:3f}'.format(time_array[i]))

    for i, axes in enumerate(ax):
      axes.legend(frameon=False)
      axes.set_xlabel('$x$')
      axes.set_ylabel(lab[i])
      if lab[i] != '$\\sigma_c$':
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()

    if self.passive:
      ax_passive.legend(frameon=False)
      ax_passive.set_xlabel('$x$')
      ax_passive.set_ylabel('Concen.')
      fig_passive.tight_layout()
      return (fig, fig_passive)

    return fig

  # Pan in 1d
  def pan(self):
    # Prompt the user to enter values
    time = float(input('Enter time: '))
    begin = float(input('Enter begining of pan: '))
    end = float(input('Enter ending of pan: '))

    file = np.argmin(np.abs(self.time_array - time))
    ibeg = np.argmin(np.abs(self.x1v - begin))
    iend = np.argmin(np.abs(self.x1v - end))
    x_pan = self.x1v[ibeg:iend]

    # Extract variables
    rho_pan = self.rho_array[file, ibeg:iend]
    vx_pan = self.vx_array[file, ibeg:iend]
    if not self.isothermal:
      pg_pan = self.pg_array[file, ibeg:iend]
    if self.cr:
      pc_pan = self.ecr_array[file, ibeg:iend]*(gc - 1.) 
    if self.b_field:
      bx_pan = self.bx_array[file, ibeg:iend] 

    # Save variables
    self.time = time 
    self.pan_begin, self.pan_end = begin, end 

    self.pan_file = file 
    self.pan_ibeg, self.pan_iend = ibeg, iend 

    self.x_pan = x_pan 
    self.rho_pan = rho_pan 
    self.vx_pan = vx_pan 
    if not self.isothermal:
      self.pg_pan = pg_pan 
    if self.cr:
      self.pc_pan = pc_pan 
    if self.b_field:
      self.bx_pan = bx_pan 

    self.runpan = True
    return 

  ##############################################################################
  # Problem specific functions
  def shock(self):
    # Prompt the user to enter values
    time = float(input('Enter time: '))
    begin = float(input('Enter begining of shock: '))
    end = float(input('Enter ending of shock: '))
    start = float(input('Enter start of check: '))
    check = float(input('Enter end of check: '))

    file = np.argmin(np.abs(self.time_array - time))
    ibeg = np.argmin(np.abs(self.x1v - begin))
    iend = np.argmin(np.abs(self.x1v - end))
    x_sh = self.x1v[ibeg:iend]
    istr = np.argmin(np.abs(x_sh - start))
    ichk = np.argmin(np.abs(x_sh - check))

    # Extract variables
    rho_sh = self.rho_array[file, ibeg:iend]
    vx_sh = self.vx_array[file, ibeg:iend]
    pg_sh = self.pg_array[file, ibeg:iend]
    pc_sh = self.ecr_array[file, ibeg:iend]/3.  
    fc_sh = self.fcx_array[file, ibeg:iend]*self.vmax 
    vs_sh = self.vs_array[file, ibeg:iend]
    sa_sh = self.sigma_adv_array[file, ibeg:iend]/self.vmax 
    sd_sh = self.sigma_diff_array[file, ibeg:iend]/self.vmax 
    sc_sh = sa_sh*sd_sh/(sa_sh + sd_sh)

    # Derived quantities
    cs_sh = np.sqrt(gg*pg_sh/rho_sh)
    cc_sh = np.sqrt(gc*pc_sh/rho_sh)
    vp_sh = np.sqrt(cs_sh**2 + cc_sh**2) 
    if self.cr_stream:
      va_sh = self.b0/np.sqrt(rho_sh)
      beta = 2.*pg_sh/self.b0 

    # Quantities involving first derivatives
    drhodx_sh = np.gradient(rho_sh, x_sh)
    dvdx_sh = np.gradient(vx_sh, x_sh)
    dpgdx_sh = np.gradient(pg_sh, x_sh)
    dpcdx_sh = np.gradient(pc_sh, x_sh)
    dfcdx_sh = np.gradient(fc_sh, x_sh)
    fcsteady = gc1*pc_sh*(vx_sh + vs_sh) - dpcdx_sh/sd_sh

    # Involving second derivatives
    d2pcdx2_sh = np.gradient(dpcdx_sh, x_sh)

    # Extract upstream and downstream variables in the shock frame
    rho1, rho2 = rho_sh[istr], rho_sh[ichk]
    pg1, pg2 = pg_sh[istr], pg_sh[ichk]
    pc1, pc2 = pc_sh[istr], pc_sh[ichk]
    cs1, cs2 = cs_sh[istr], cs_sh[ichk]
    cc1, cc2 = cc_sh[istr], cc_sh[ichk]
    vs1, vs2 = vs_sh[istr], vs_sh[ichk]

    vsh = (rho1*vx_sh[istr] - rho2*vx_sh[ichk])/(rho1 - rho2)
    vx_sh = vx_sh - vsh 
    # fc_sh = fc_sh - gc1*pc_sh*vsh
    # fcsteady = fcsteady - gc1*pc_sh*vsh
    dfcdx_sh = dfcdx_sh - gc1*dpcdx_sh*vsh
    v1, v2 = vx_sh[istr], vx_sh[ichk] 
    fc1, fc2 = fc_sh[istr], fc_sh[ichk]

    if self.cr_stream:
      va1, va2 = va_sh[istr], va_sh[ichk]
      beta1, beta2 = beta[istr], beta[ichk]
      vp_sh = np.sqrt(cs_sh**2 + cc_sh**2*(np.abs(vx_sh) - va_sh/2.)*(np.abs(vx_sh) \
        + (gg - 1.)*va_sh)/(np.abs(vx_sh)*(np.abs(vx_sh) - va_sh)))

    vp1, vp2 = vp_sh[istr], vp_sh[ichk]
    m1, m2 = np.abs(v1)/vp1, np.abs(v2)/vp2 
    n1, n2 = pc1/(pc1 + pg1), pc2/(pc2 + pg2)

    # Save variables
    self.time = time 
    self.begin, self.end = begin, end 
    self.start, self.check = start, check

    self.file = file 
    self.ibeg, self.iend = ibeg, iend 
    self.istr, self.ichk = istr, ichk

    self.x_sh = x_sh 
    self.rho_sh = rho_sh 
    self.vx_sh = vx_sh
    self.pg_sh = pg_sh 
    self.pc_sh = pc_sh 
    self.fc_sh = fc_sh 
    self.vs_sh = vs_sh 
    self.sa_sh = sa_sh 
    self.sd_sh = sd_sh 
    self.sc_sh = sc_sh

    self.cs_sh = cs_sh 
    self.cc_sh = cc_sh 
    self.vp_sh = vp_sh
    if self.cr_stream:
      self.va_sh = va_sh 
      self.beta = beta

    self.drhodx_sh = drhodx_sh 
    self.dvdx_sh = dvdx_sh 
    self.dpgdx_sh = dpgdx_sh 
    self.dpcdx_sh = dpcdx_sh 
    self.dfcdx_sh = dfcdx_sh 
    self.fcsteady = fcsteady

    self.d2pcdx2_sh = d2pcdx2_sh 

    self.rho1, self.rho2 = rho1, rho2 
    self.pg1, self.pg2 = pg1, pg2 
    self.pc1, self.pc2 = pc1, pc2 
    self.cs1, self.cs2 = cs1, cs2 
    self.cc1, self.cc2 = cc1, cc2
    self.vs1, self.vs2 = vs1, vs2

    self.vsh = vsh 
    self.v1, self.v2 = v1, v2 
    self.fc1, self.fc2 = fc1, fc2 

    if self.cr_stream:
      self.va1, self.va2 = va1, va2 
      self.beta1, self.beta2 = beta1, beta2 

    self.vp1, self.vp2 = vp1, vp2 
    self.m1, self.m2 = m1, m2 
    self.n1, self.n2 = n1, n2 

    # Mark this function as run
    self.runshock = True
    return 


  def plotshock(self):
    if self.runshock == False:
      self.shock()

    fig_width = 7
    golden_mean = (np.sqrt(5.) - 1.)/2. 
    fig_height = 5.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    grids = gs.GridSpec(5, 1, figure=fig, hspace=0)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[1, 0])
    ax3 = fig.add_subplot(grids[2, 0])
    ax4 = fig.add_subplot(grids[3, 0])
    ax5 = fig.add_subplot(grids[4, 0])

    ax1.plot(self.x_sh, self.dpcdx_sh, label='$\\nabla P_c$')

    ax2.semilogy(self.x_sh, self.sc_sh, label='$\\sigma_c$')

    ax3.semilogy(self.x_sh, np.abs((self.fc_sh - self.fcsteady)/self.fc_sh), 'b', label='$\\Delta F_c/F_c$')
    # ax3.plot(self.x_sh, self.fcsteady, 'r--', label='Steady $F_c$')

    ax4.plot(self.x_sh, self.rho_sh, label='$\\rho$')

    ax5.plot(self.x_sh, self.pc_sh, label='$P_c$')

    for axes in fig.axes:
      axes.axvline(self.x_sh[self.istr], linestyle='--', color='grey')
      axes.axvline(self.x_sh[self.ichk], linestyle='--', color='purple')
      axes.margins(x=0)
      if (axes != ax2) and (axes != ax3):
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

      if axes != ax4:
        axes.legend(frameon=False, fontsize=10)
      else:
        handles, labels = axes.get_legend_handles_labels()
        handles1, labels1 = handles[0:3], labels[0:3]
        handles2, labels2 = handles[3:-1], labels[3:-1]
        handles3, labels3 = [handles[-1]], [labels[-1]]
        axes.legend(handles1, labels1, frameon=False, loc='upper right', ncol=3, fontsize=12)
        leg2 = lg.Legend(axes, handles2, labels2, frameon=False, loc='upper left', ncol=3, fontsize=12)
        leg3 = lg.Legend(axes, handles3, labels3, frameon=False, loc='lower left', fontsize=12)
        axes.add_artist(leg2)
        axes.add_artist(leg3)

      if axes != ax5:
        axes.set_xticks([])
      else:
        axes.set_xlabel('$x$', fontsize=10)

      for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(10)

      fig.tight_layout()
      
    return fig

  # Count the number of stairs, stair height, stair width, stair plateau width
  def staircase(self, plot=False, xlim=False, time_series=False, fit=False):
    filename = self.filename 

    # Analyse only a specific region
    if xlim:
      xl = float(input('Enter starting x for staircase analysis: '))
      xu = float(input('Enter ending x for staircase analysis: '))

    if time_series:
      file_avg = int(input('Enter file for start of pc averaging: '))
      num_file_avg = np.size(self.file_array[file_avg:])

    # Choosing files for display
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))
    couple_file = dict.fromkeys(file_array)
    num_stair_file = dict.fromkeys(file_array)
    stair_start_loc_file = dict.fromkeys(file_array)
    stair_end_loc_file = dict.fromkeys(file_array)
    width_file = dict.fromkeys(file_array)
    height_file = dict.fromkeys(file_array)
    plateau_file = dict.fromkeys(file_array)

    # Extracting data
    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      time_array[i] = float('{0:f}'.format(data['Time']))
      # For no mesh refinement
      x1v = np.array(data['x1v'])
      if xlim:
        index_xl = np.argmin(np.abs(x1v - xl))
        index_xu = np.argmin(np.abs(x1v - xu)) + 1
      else:
        index_xl = 0
        index_xu = np.size(x1v)
      x1v = x1v[index_xl:index_xu]
      grids = np.size(x1v)
      dx = x1v[1] - x1v[0]
      vx = data['vel1'][0, 0, index_xl:index_xu]
      pc = data['Ec'][0, 0, index_xl:index_xu]*(gc - 1.)
      fc = data['Fc1'][0, 0, index_xl:index_xu]*self.vmax
      vs = data['Vc1'][0, 0, index_xl:index_xu]
      kappa = (gc - 1.)*self.vmax/data['Sigma_diff1'][0, 0, index_xl:index_xu] 
      if time_series:
        if fp == file_avg:
          pc_time_avg = pc/num_file_avg 
        elif fp > file_avg:
          pc_time_avg += pc/num_file_avg
      dpcdx = np.gradient(pc, x1v)
      Lc = np.zeros(grids)
      for j, gradpc in enumerate(dpcdx):
        Lc[j] = np.abs(pc[j]/gradpc) if gradpc != 0.0 else big_number
      fcsdy = gc1*pc*(vx + vs) - (kappa/(gc - 1.))*dpcdx
      dfcfc = np.abs(1. - fcsdy/fc)
      
      couple = np.zeros(grids)
      L_equi = np.abs(self.profile_object.pc_sol/self.profile_object.dpcdx_sol)
      for j in np.arange(grids):
        # Condition 1: Fc has to be close enough to steady state Fc
        if (dfcfc[j] < 100/self.vmax**2) and (Lc[j] < 5*L_equi[j]):
          couple[j] = 1
        else:
          couple[j] = 0
      # Tightly Enclosed friends are also coupled
      old_couple = np.zeros(grids)
      new_couple = np.zeros(grids)
      num_stair = 0
      num_stair_old = -10
      stair_start_location = np.zeros(grids)
      stair_end_location = np.zeros(grids)
      link_grids = 1
      while num_stair != num_stair_old:
        num_stair_old = num_stair
        print(num_stair)
        num_stair = 0
        for j in np.arange(grids):
          if (j >= link_grids) and (j < grids-link_grids):
            check_for = couple[j:j+link_grids+1]
            check_back = couple[j-link_grids:j+1]
            if np.any(check_for == 1) and np.any(check_back == 1):
              new_couple[j] = 1
            else:
              new_couple[j] = 0
          else:
            new_couple[j] = couple[j]
        # Count the stairs
        toggle_start = False
        for j in np.arange(1, grids-1):
          if new_couple[j] == 1:
            if (new_couple[j-1] == 0) and (new_couple[j+1] == 1) and (toggle_start == False): # Allow finder to begin a stair only after the end of a previous stair
              num_stair += 1
              stair_start_location[j] = 1
              toggle_start = True
            if (new_couple[j-1] == 1) and (new_couple[j+1] == 0) and (toggle_start == True):
              stair_end_location[j] = 1
              toggle_start = False 
        couple = new_couple
        new_couple = np.zeros(grids)

      # Save variable
      couple_file[fp] = couple
      num_stair_file[fp] = num_stair
      stair_start_loc_file[fp] = stair_start_location
      stair_end_loc_file[fp] = stair_end_location
      
      # Analysis
      width = np.array([])
      height = np.array([])
      plateau = np.array([])
      for j, x in enumerate(x1v):
        if stair_start_location[j] == 1:
          k = j 
          while (stair_end_location[k] == 0) and (k < grids-1):
            k += 1
          width = np.append(width, x1v[k] - x1v[j])
          height = np.append(height, np.abs(pc[k] - pc[j])) 
        if stair_end_location[j] == 1:
          k = j
          while (stair_start_location[k] == 0) and (k < grids-1):
            k += 1
          plateau = np.append(plateau, x1v[k] - x1v[j])

      width_file[fp] = width
      height_file[fp] = height
      plateau_file[fp] = plateau

    # Save variable
    self.time_array = time_array
    self.x1v = x1v
    self.index_xl = index_xl
    self.index_xu = index_xu
    self.couple = couple_file
    self.num_stair = num_stair_file
    self.stair_start_loc = stair_start_loc_file
    self.stair_end_loc = stair_end_loc_file
    self.width = width_file
    self.height = height_file
    self.plateau = plateau_file
    self.pc_time_avg = pc_time_avg

    if plot:
      if time_series:
        fig_time = plt.figure()
        ax_time = fig_time.add_subplot(111)

        for i, key in enumerate(self.num_stair.keys()):
          ax_time.scatter(stair.time_array[i], stair.num_stair[key], color='b')

        ax_time.set_xlabel('Time')
        ax_time.set_ylabel('Number of stairs')
        ax_time.xaxis.set_minor_locator(AutoMinorLocator())
        ax_time.yaxis.set_minor_locator(AutoMinorLocator())

        fig_time.tight_layout()
        plt.show()

      time = float(input('Enter time to start performing statistics: '))
      display_file = file_array[np.argmin(np.abs(self.time_array - time))]
      data = ar.athdf('./' + self.filename + '.out1.' + format(display_file, '05d') + '.athdf')
      x1v_display = np.array(data['x1v'])
      pc_display = data['Ec'][0, 0, :]*(gc - 1.)

      width_record = np.array([])
      plateau_record = np.array([])
      height_record = np.array([])
      for i, file in enumerate(file_array[display_file:]):
        width_record = np.append(width_record, self.width[file])
        plateau_record = np.append(plateau_record, self.plateau[file])
        height_record = np.append(height_record, self.height[file])

      self.width_record = width_record
      self.plateau_record = plateau_record 
      self.height_record = height_record

      fig1 = plt.figure(figsize=(13, 6))
      fig2 = plt.figure()

      grids = gs.GridSpec(1, 3, figure=fig2)
      ax1 = fig1.add_subplot(grids[0, 0])
      ax2 = fig1.add_subplot(grids[0, 1])
      ax3 = fig1.add_subplot(grids[0, 2])
      ax = [ax1, ax2, ax3]
      ax4 = fig2.add_subplot(111)

      num_bins = 100 
      width_bin = np.logspace(np.log10(np.amin(width_record)), np.log10(np.amax(width_record)), num_bins)
      plateau_bin = np.logspace(np.log10(np.amin(plateau_record)), np.log10(np.amax(plateau_record)), num_bins)
      height_bin = np.logspace(np.log10(np.amin(height_record)), np.log10(np.amax(height_record)), num_bins)

      widths = (width_bin[1:] - width_bin[:-1])
      plateaus = (plateau_bin[1:] - plateau_bin[:-1])
      heights = (height_bin[1:] - height_bin[:-1])

      width_hist = np.histogram(width_record, bins=width_bin)
      plateau_hist = np.histogram(plateau_record, bins=plateau_bin)
      height_hist = np.histogram(height_record, bins=height_bin)

      width_norm = width_hist[0]/(widths*np.size(width_hist[0]))
      plateau_norm = plateau_hist[0]/(plateaus*np.size(plateau_hist[0]))
      height_norm = height_hist[0]/(heights*np.size(height_hist[0]))

      if fit:
        rho = self.profile_object.rho_sol[stair.index_xl:stair.index_xu]
        pg = self.profile_object.pg_sol[stair.index_xl:stair.index_xu]
        pc = self.profile_object.pc_sol[stair.index_xl:stair.index_xu]
        dpcdx = self.profile_object.dpcdx_sol[stair.index_xl:stair.index_xu]
        L = np.abs(pc/dpcdx)
        cs = np.sqrt(gg*pg/rho)
        kappa = self.profile_object.kappa 
        ldiff = kappa/cs

        ldiff_avg = np.mean(ldiff)
        pc_height_avg = np.mean(pc)*ldiff_avg/np.mean(L)

        fit_func = lambda r, c, alpha, beta, chi: c - alpha*r - (np.exp(r)/beta)**(chi)

        width_delete = np.where(width_norm==0)[0]
        plateau_delete = np.where(plateau_norm==0)[0]
        height_delete = np.where(height_norm==0)[0]

        width_bin_fit = np.delete(width_bin, width_delete)
        width_norm_fit = np.delete(width_norm, width_delete)

        plateau_bin_fit = np.delete(plateau_bin, plateau_delete)
        plateau_norm_fit = np.delete(plateau_norm, plateau_delete)

        height_bin_fit = np.delete(height_bin, height_delete)
        height_norm_fit = np.delete(height_norm, height_delete)

        width_bin_fit_log = np.log(width_bin_fit/ldiff_avg)
        width_norm_fit_log = np.log(width_norm_fit)

        plateau_bin_fit_log = np.log(plateau_bin_fit/ldiff_avg)
        plateau_norm_fit_log = np.log(plateau_norm_fit)

        height_bin_fit_log = np.log(height_bin_fit/pc_height_avg)
        height_norm_fit_log = np.log(height_norm_fit)

        width_begin = ((self.x1max - self.x1min)/self.nx1)
        width_index = np.argmin(np.abs(width_bin_fit - width_begin))

        plateau_begin = ((self.x1max - self.x1min)/self.nx1)
        plateau_index = np.argmin(np.abs(plateau_bin_fit - plateau_begin))

        height_begin = np.mean(pc)*((self.x1max - self.x1min)/self.nx1)/np.mean(L)
        height_index = np.argmin(np.abs(height_bin_fit - height_begin))

        try:
          width_opt, width_cov = opt.curve_fit(fit_func, width_bin_fit_log[width_index:-1], width_norm_fit_log[width_index:], \
            p0=(np.amax(width_norm_fit_log[width_index:]), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
          width_fit_success = True
          width_amp, width_alpha, width_beta, width_chi = width_opt 
          self.width_alpha, self.width_beta, self.width_chi = width_alpha, width_beta, width_chi
        except:
          print('Cannot fit width')
          width_fit_success = False
          self.width_alpha, self.width_beta, self.width_chi = None, None, None

        try:
          plateau_opt, plateau_cov = opt.curve_fit(fit_func, plateau_bin_fit_log[plateau_index:-1], plateau_norm_fit_log[plateau_index:], \
            p0=(np.amax(plateau_norm_fit_log[plateau_index:]), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
          plateau_fit_success = True
          plateau_amp, plateau_alpha, plateau_beta, plateau_chi = plateau_opt 
          self.plateau_alpha, self.plateau_beta, self.plateau_chi = plateau_alpha, plateau_beta, plateau_chi
        except:
          print('Cannot fit plateau')
          plateau_fit_success = False
          self.plateau_alpha, self.plateau_beta, self.plateau_chi = None, None, None

        try:
          height_opt, height_cov = opt.curve_fit(fit_func, height_bin_fit_log[height_index:-1], height_norm_fit_log[height_index:], \
            p0=(np.amax(height_norm_fit_log[height_index:]), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
          height_fit_success = True
          height_amp, height_alpha, height_beta, height_chi = height_opt
          self.height_alpha, self.height_beta, self.height_chi = height_alpha, height_beta, height_chi
        except:
          print('Cannot fit height')
          height_fit_success = False
          self.height_alpha, self.height_beta, self.height_chi = None, None, None

        self.eta = self.profile_object.kappa/(gc*L[0]*cs[0])
        self.ldiff_avg, self.pc_height_avg = ldiff_avg, pc_height_avg 

      ax1.bar(width_bin[:-1], width_norm, widths)
      ax2.bar(plateau_bin[:-1], plateau_norm, plateaus)
      ax3.bar(height_bin[:-1], height_norm, heights)

      if fit:
        if width_fit_success:
          ax1.loglog(np.exp(width_bin_fit_log[:-1])*ldiff_avg, \
            np.exp(fit_func(width_bin_fit_log[:-1], width_amp, width_alpha, width_beta, width_chi)), 'k--', \
            label='$\\alpha={:.2f},\\beta={:.2f},\\chi={:.2f}$'.format(width_alpha, width_beta, width_chi))
          ax1.legend(frameon=False)
        if plateau_fit_success:
          ax2.loglog(np.exp(plateau_bin_fit_log[:-1])*ldiff_avg, \
            np.exp(fit_func(plateau_bin_fit_log[:-1], plateau_amp, plateau_alpha, plateau_beta, plateau_chi)), 'k--', \
            label='$\\alpha={:.2f},\\beta={:.2f},\\chi={:.2f}$'.format(plateau_alpha, plateau_beta, plateau_chi))
          ax2.legend(frameon=False)
        if height_fit_success:
          ax3.loglog(np.exp(height_bin_fit_log[:-1])*pc_height_avg, \
            np.exp(fit_func(height_bin_fit_log[:-1], height_amp, height_alpha, height_beta, height_chi)), 'k--', \
            label='$\\alpha={:.2f},\\beta={:.2f},\\chi={:.2f}$'.format(height_alpha, height_beta, height_chi))
          ax3.legend(frameon=False)

        ax1.axvline(x=((stair.x1max - stair.x1min)/stair.nx1), color='r', linestyle='--')
        ax2.axvline(x=((stair.x1max - stair.x1min)/stair.nx1), color='r', linestyle='--')
        ax3.axvline(x=np.mean(pc)*((stair.x1max - stair.x1min)/stair.nx1)/np.mean(L), color='r', linestyle='--')

      ax4.plot(x1v_display, pc_display, 'o-', label='t={:.3f}'.format(time))
      for i, x in enumerate(x1v_display[index_xl:index_xu]): 
        if self.stair_start_loc[display_file][i] == 1: 
          ax4.axvline(x1v_display[index_xl+i], linestyle='--', color='g')
        if self.stair_end_loc[display_file][i] == 1:
          ax4.axvline(x1v_display[index_xl+i], linestyle='--', color='r')

      lab = ['Width', 'Plateau width', '$P_c$ Height']

      for i, axes in enumerate(ax):
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel(lab[i])
        axes.set_ylabel('Distribution')
        
      ax4.legend(frameon=False)
      ax4.set_xlabel('$x$')
      ax4.set_ylabel('$P_c$')
      ax4.xaxis.set_minor_locator(AutoMinorLocator())
      ax4.yaxis.set_minor_locator(AutoMinorLocator())

      fig1.tight_layout()
      fig2.tight_layout()
     
      if time_series:
        fig3 = plt.figure()
        fig4 = plt.figure()
        ax5 = fig3.add_subplot(111)
        ax6 = fig4.add_subplot(111)

        for i, key in enumerate(self.num_stair.keys()):
          ax5.scatter(stair.time_array[i], stair.num_stair[key], color='b')

        ax6.plot(x1v, pc_time_avg, label='$\\langle P_c \\rangle$')

        ax5.set_xlabel('Time')
        ax5.set_ylabel('Number of stairs')
        ax5.xaxis.set_minor_locator(AutoMinorLocator())
        ax5.yaxis.set_minor_locator(AutoMinorLocator())

        ax6.set_xlabel('$x$')
        ax6.set_ylabel('$P_c$')
        ax6.xaxis.set_minor_locator(AutoMinorLocator())
        ax6.yaxis.set_minor_locator(AutoMinorLocator())

        fig3.tight_layout()
        fig4.tight_layout()
        return [fig1, fig2, fig3, fig4]

      return [fig1, fig2]
    return

  def convexhull(self, xlim=False, plot=False):
    filename = self.filename 

    # Analyse only a specific region
    if xlim:
      xl = float(input('Enter starting x for staircase analysis: '))
      xu = float(input('Enter ending x for staircase analysis: '))

    # Choosing files for display
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))
    x_ori = dict.fromkeys(file_array)
    rho_ori = dict.fromkeys(file_array)
    pc_ori = dict.fromkeys(file_array)
    rho_stair = dict.fromkeys(file_array)
    pc_stair = dict.fromkeys(file_array)
    x_stair = dict.fromkeys(file_array)

    # Extracting data
    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      time_array[i] = float('{0:f}'.format(data['Time']))
      # For no mesh refinement
      x1v = np.array(data['x1v'])
      if xlim:
        index_xl = np.argmin(np.abs(x1v - xl))
        index_xu = np.argmin(np.abs(x1v - xu)) + 1
      else:
        index_xl = 0
        index_xu = np.size(x1v)
      x1v = x1v[index_xl:index_xu]
      grids = np.size(x1v)
      dx = x1v[1] - x1v[0]
      rho = data['rho'][0, 0, index_xl:index_xu]
      pc = data['Ec'][0, 0, index_xl:index_xu]*(gc - 1)
      rho_max = np.amax(rho)
      rho_min = np.amin(rho)
      pc_max = np.amax(pc)
      num_rhobin = self.nx1
      rho_level = np.linspace(rho_max, rho_min, num_rhobin)
      loc_stair = np.array([], dtype=int)
      tol = 0.5*(rho_level[0] - rho_level[1]) # tolerance
      mask_loc = np.zeros(grids)
      prev_loc = np.array([np.argmax(rho)])
      for i, r in enumerate(rho_level):
        mask_rho = ma.masked_array(rho, mask_loc)
        max_loc = np.where(np.abs(mask_rho - r) < tol)[0]
        loc_stair = np.append(loc_stair, max_loc)
        if np.size(max_loc) > 0:
          mask_loc[max_loc[0]:max_loc[-1]+1] = 1
          for i, loc in enumerate(max_loc):
            if np.any(np.abs(prev_loc - loc) > 1):
              nearest_loc = prev_loc[np.argsort(np.abs(prev_loc - loc)[0])][0]
              if nearest_loc < loc:
                mask_loc[nearest_loc:loc+1] = 1
              else:
                mask_loc[loc:nearest_loc+1] = 1
          prev_loc = max_loc
      loc_stair = np.sort(loc_stair)
      x_stair[fp] = x1v[loc_stair]
      rho_stair[fp] = rho[loc_stair]
      fit_pc = lambda pc0: np.sum(pc0*(rho_stair[fp]/rho_max)**(gc/2.) - pc[loc_stair])**2
      pc_fit = opt.minimize(fit_pc, pc_max)
      pc_stair[fp] = pc_fit.x[0]*(rho_stair[fp]/rho_max)**(gc/2.)
      x_ori[fp] = x1v 
      rho_ori[fp] = rho 
      pc_ori[fp] = pc

    self.time_array = time_array
    self.x = x1v
    self.rho = rho
    self.pc = pc
    self.pc_try = pc[loc_stair]
    self.rho_level = rho_level
    self.rho_stair = rho_stair 
    self.pc_stair = pc_stair
    self.x_stair = x_stair
    self.x_ori = x_ori
    self.rho_ori = rho_ori 
    self.pc_ori = pc_ori 

    if plot:
      time = float(input('Enter time to construct convex hull: '))
      file = file_array[np.argmin(np.abs(self.time_array - time))]

      fig1 = plt.figure()
      fig2 = plt.figure()
      ax1 = fig1.add_subplot(111)
      ax2 = fig2.add_subplot(111)
      ax = [ax1, ax2]

      ax1.scatter(x_ori[file], rho_ori[file], label='t={:.3f}'.format(time))
      ax1.plot(x_stair[file], rho_stair[file], 'k--')

      ax2.scatter(x_ori[file], pc_ori[file], label='t={:.3f}'.format(time))
      ax2.plot(x_stair[file], pc_stair[file], 'k--')

      lab = ['$\\rho$', '$P_c$']

      for i, axes in enumerate(ax):
        axes.legend(frameon=False)
        axes.set_xlabel('$x$')
        axes.set_ylabel(lab[i])
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

      fig1.tight_layout()
      fig2.tight_layout()

      return [fig1, fig2]

    return

  def history(self, plot=False):
    if not self.passive:
      print('There is no passive scalars.')
      return

    filename = self.filename 
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))

    x_hist = np.zeros(np.size(file_array))
    den_hist = np.zeros(np.size(file_array))
    mom_hist = np.zeros(np.size(file_array)) 
    kin_hist = np.zeros(np.size(file_array))
    if not self.isothermal:
      therm_hist = np.zeros(np.size(file_array))
    if self.cr:
      ec_hist = np.zeros(np.size(file_array))

    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      time_array[i] = float('{0:f}'.format(data['Time']))
      x1v = data['x1v']
      r = data['r0'][0, 0, :]
      rho = data['rho'][0, 0, :]
      vx = data['vel1'][0, 0, :]
      if not self.isothermal:
        pg = data['press'][0, 0, :]
      if self.cr:
        ec = data['Ec'][0, 0, :] 
      passive_count = np.where(r > 0.5)[0]
      x_hist[i] = np.mean(x1v[passive_count])
      den_hist[i] = np.mean(rho[passive_count])
      mom_hist[i] = np.mean(rho[passive_count]*vx[passive_count]) 
      kin_hist[i] = np.mean(0.5*rho[passive_count]*vx[passive_count]**2) 
      if not self.isothermal:
        therm_hist[i] = np.mean(pg[passive_count]/(gg - 1.))
      if self.cr:
        ec_hist[i] = np.mean(ec[passive_count])

    # Save data
    self.time_array = time_array
    self.x_hist = x_hist
    self.den_hist = den_hist 
    self.mom_hist = mom_hist 
    self.kin_hist = kin_hist 
    if not self.isothermal:
      self.therm_hist = therm_hist
    if self.cr:
      self.ec_hist = ec_hist

    if plot:
      fig = plt.figure(figsize=(12, 4))
      grids = gs.GridSpec(2, 3, figure=fig)
      ax1 = fig.add_subplot(grids[0, 0])
      ax2 = fig.add_subplot(grids[0, 1])
      ax3 = fig.add_subplot(grids[0, 2])
      ax4 = fig.add_subplot(grids[1, 0])
      ax5 = fig.add_subplot(grids[1, 1])
      ax6 = fig.add_subplot(grids[1, 2])

      lab = ['$x$', '$\\langle\\rho v\\rangle$', '$\\langle E_\\mathrm{kin} \\rangle$', '$\\langle E_\\mathrm{th} \\rangle$', '$\\langle E_\\mathrm{c} \\rangle$', '$E$']

      ax1.plot(time_array, x_hist)
      ax2.plot(time_array, mom_hist)
      ax3.plot(time_array, kin_hist)
      ax4.plot(time_array, therm_hist)
      ax5.plot(time_array, ec_hist)
      ax6.semilogy(time_array, kin_hist, label='$\\langle E_\\mathrm{kin} \\rangle$')
      ax6.semilogy(time_array, therm_hist, label='$\\langle E_\\mathrm{th} \\rangle$')
      ax6.semilogy(time_array, ec_hist, label='$\\langle E_\\mathrm{c} \\rangle$')

      for i, axes in enumerate(fig.axes):
        axes.set_xlabel('t')
        axes.set_ylabel(lab[i])
        axes.margins(x=0)
        if axes == ax6:
          axes.legend(frameon=False)
        else:
          axes.xaxis.set_minor_locator(AutoMinorLocator())
          axes.yaxis.set_minor_locator(AutoMinorLocator())

      fig.tight_layout()
    return fig
# End of class Plot1d


# Background profile class
class Power:
  def __init__(self, profile_input):
    alpha = profile_input['alpha']
    beta = profile_input['beta']
    eta = profile_input['eta']
    psi = profile_input['psi']
    rho0 = profile_input['rho0']
    pg0 = profile_input['pg0']
    r0 = profile_input['r0']
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

  def powerprofile(self, xmin, xmax, grids):
    kappa = self.kappa 
    b = self.b
    va0 = self.va0
    r0 = self.r0

    func = lambda x, va: (kappa*self.d2pcdx2(x) - va*self.dpcdx(x))/(gc*self.pc(x))

    dx = (xmax - xmin)/grids 
    x_eval = np.zeros(grids)
    x_eval[0] = xmin + 0.5*dx 
    for i in np.arange(np.size(x_eval) - 1):
      x_eval[i+1] = x_eval[i] + dx 
    index = np.argmin(np.abs(x_eval - r0))
    index = index if (x_eval[index] > r0) else index + 1

    sol_front = integrate.solve_ivp(func, (r0, x_eval[-1]), [va0], t_eval=x_eval[index:])
    if not index == 0:
      sol_back = integrate.solve_ivp(func, (r0, x_eval[0]), [va0], t_eval=x_eval[0:index][::-1])
      va_sol = np.append(sol_back.y[0][::-1], sol_front.y[0])
    else:
      va_sol = sol_front.y[0]

    rho_sol = (b/va_sol)**2
    pg_sol = self.pg(x_eval)
    pc_sol = self.pc(x_eval)
    fc_sol = gc1*pc_sol*va_sol - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    g_sol = (self.dpgdx(x_eval) + self.dpcdx(x_eval))/rho_sol
    H_sol = va_sol*self.dpcdx(x_eval)

    # Save data
    self.dx = dx
    self.xmin = xmin
    self.xmax = xmax
    self.x_sol = x_eval
    self.va_sol = va_sol 
    self.rho_sol = rho_sol
    self.pg_sol = pg_sol
    self.pc_sol = pc_sol
    self.dpcdx_sol = self.dpcdx(x_eval)
    self.fc_sol = fc_sol 
    self.g_sol = g_sol
    self.H_sol = H_sol

    return

# End of Power class
####################################
inputfile = 'athinput.cr_power'
file_array = np.array([0, 250]) 

# Background
profile_in = dict()
profile_in['alpha'] = 1.
profile_in['beta'] = 1.
profile_in['eta'] = 0.001
profile_in['psi'] = 0.

profile_in['rho0'] = 1. 
profile_in['pg0'] = 1. 
profile_in['r0'] = 1.

one = Plot1d(inputfile, file_array, video=False, staircase=False, profile_in=profile_in)
if one.passive:
  fig, fig_pass = one.plot()
  fig.savefig('./1dplot.png', dpi=300)
  fig_pass.savefig('./1dplot_passive.png', dpi=300)
else:
  fig = one.plot()
  fig.savefig('./1dplot.png', dpi=300)
plt.show()
plt.close('all')

# one.shock()
# shkfig = one.plotshock()
# shkfig.savefig('./1dshock.png', dpi=300)
# plt.show()

# with h5py.File('../analytic/shock.hdf5', 'w') as fp:
#   dset = fp.create_dataset('x', data=one.x_sh)
#   dset = fp.create_dataset('rho', data=one.rho_sh)
#   dset = fp.create_dataset('v', data=one.vx_sh)
#   dset = fp.create_dataset('pg', data=one.pg_sh)
#   dset = fp.create_dataset('pc', data=one.pc_sh)
#   dset = fp.create_dataset('fc', data=one.fc_sh)

# # Plot amplitude 
# x = np.loadtxt('../analytics/x.csv')
# amp = np.loadtxt('../analytics/amp.csv')

# latexify(columns=1)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(one.x1v, one.vx_array[1, :])
# ax1.plot(x, amp, '--')
# ax1.margins(x=0)
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$v$')
# fig.tight_layout()
# fig.savefig('./amp_compare.png', dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()

#################################
# Make video
# Equilibrium
equi = {}
equi['rho'] = 0.0
equi['pc'] = 0.0
equi['pg'] = 0.0

video_array = np.arange(300)
video_path = '/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/power/results/sims/output/video/'
video = Plot1d(inputfile, video_array, video=True, staircase=False, profile_in=profile_in)
video.make_video(equi, video_path)

################################
# # Staricase identification
# # stair_array = np.array([311])
# stair_array = np.arange(300)
# stair = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=profile_in)
# plotdefault()
# stairfig_stat, stairfig_pc, stairfig_time, stairfig_avgpc = stair.staircase(plot=True, xlim=False, time_series=True, fit=True)
# stairfig_stat.savefig('./staircase_stat.png', dpi=300)
# stairfig_pc.savefig('./staircase_pc.png', dpi=300)
# stairfig_time.savefig('./staircase_time.png', dpi=300) # Need to comment out if time_series=False
# stairfig_avgpc.savefig('./staircase_avgpc.png', dpi=300) # Need to comment out if time_series=False

# plt.show()
# plt.close('all')

###############################
# # Construct convex hull for density and reconstruct Pc
# stair_array = np.array([150])
# stair2 = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=profile_in)
# plotdefault()
# stair2fig_rho, stair2fig_pc = stair2.convexhull(xlim=False, plot=True)

# stair2fig_rho.savefig('./rho_hull.png', dpi=300)
# stair2fig_pc.savefig('./pc_hull.png', dpi=300)

# plt.show()
# plt.close('all')

###############################
# Staircase fitting
# try:
#   with h5py.File('fit_record.hdf5', 'r+') as fp:
#     eta_record = np.array(fp['eta'])
#     ldiff_record = np.array(fp['ldiff_avg'])
#     pc_height_record = np.array(fp['pc_height_avg'])
#     alpha_width_record = np.array(fp['alpha_width'])
#     alpha_plateau_record = np.array(fp['alpha_plateau'])
#     alpha_height_record = np.array(fp['alpha_height'])
#     beta_width_record = np.array(fp['beta_width'])
#     beta_plateau_record = np.array(fp['beta_plateau'])
#     beta_height_record = np.array(fp['beta_height'])
#     chi_width_record = np.array(fp['chi_width'])
#     chi_plateau_record = np.array(fp['chi_plateau'])
#     chi_height_record = np.array(fp['chi_height'])
#     del fp['eta']
#     del fp['ldiff_avg']
#     del fp['pc_height_avg']
#     del fp['alpha_width']
#     del fp['alpha_plateau']
#     del fp['alpha_height']
#     del fp['beta_width']
#     del fp['beta_plateau']
#     del fp['beta_height']
#     del fp['chi_width']
#     del fp['chi_plateau']
#     del fp['chi_height']
#     eta_record = np.append(eta_record, stair.eta)
#     ldiff_record = np.append(ldiff_record, stair.ldiff_avg)
#     pc_height_record = np.append(pc_height_record, stair.pc_height_avg)
#     alpha_width_record = np.append(alpha_width_record, stair.width_alpha)
#     alpha_plateau_record = np.append(alpha_plateau_record, stair.plateau_alpha)
#     alpha_height_record = np.append(alpha_height_record, stair.height_alpha)
#     beta_width_record = np.append(beta_width_record, stair.width_beta)
#     beta_plateau_record = np.append(beta_plateau_record, stair.plateau_beta)
#     beta_height_record = np.append(beta_height_record, stair.height_beta)
#     chi_width_record = np.append(chi_width_record, stair.width_chi)
#     chi_plateau_record = np.append(chi_plateau_record, stair.plateau_chi)
#     chi_height_record = np.append(chi_height_record, stair.height_chi)
#     fp.create_dataset('eta', data=eta_record)
#     fp.create_dataset('ldiff_avg', data=ldiff_record)
#     fp.create_dataset('pc_height_avg', data=pc_height_record)
#     fp.create_dataset('alpha_width', data=alpha_width_record)
#     fp.create_dataset('alpha_plateau', data=alpha_plateau_record)
#     fp.create_dataset('alpha_height', data=alpha_height_record)
#     fp.create_dataset('beta_width', data=beta_width_record)
#     fp.create_dataset('beta_plateau', data=beta_plateau_record)
#     fp.create_dataset('beta_height', data=beta_height_record)
#     fp.create_dataset('chi_width', data=chi_width_record)
#     fp.create_dataset('chi_plateau', data=chi_plateau_record)
#     fp.create_dataset('chi_height', data=chi_height_record)

# except:
#   with h5py.File('fit_record.hdf5', 'w-') as fp:
#     eta_record = np.array([stair.eta])
#     ldiff_record = np.array([stair.ldiff_avg])
#     pc_height_record = np.array([stair.pc_height_avg])
#     alpha_width_record = np.array([stair.width_alpha])
#     alpha_plateau_record = np.array([stair.plateau_alpha])
#     alpha_height_record = np.array([stair.height_alpha])
#     beta_width_record = np.array([stair.width_beta])
#     beta_plateau_record = np.array([stair.plateau_beta])
#     beta_height_record = np.array([stair.height_beta])
#     chi_width_record = np.array([stair.width_chi])
#     chi_plateau_record = np.array([stair.plateau_chi])
#     chi_height_record = np.array([stair.height_chi])
#     fp.create_dataset('eta', data=eta_record)
#     fp.create_dataset('ldiff_avg', data=ldiff_record)
#     fp.create_dataset('pc_height_avg', data=pc_height_record)
#     fp.create_dataset('alpha_width', data=alpha_width_record)
#     fp.create_dataset('alpha_plateau', data=alpha_plateau_record)
#     fp.create_dataset('alpha_height', data=alpha_height_record)
#     fp.create_dataset('beta_width', data=beta_width_record)
#     fp.create_dataset('beta_plateau', data=beta_plateau_record)
#     fp.create_dataset('beta_height', data=beta_height_record)
#     fp.create_dataset('chi_width', data=chi_width_record)
#     fp.create_dataset('chi_plateau', data=chi_plateau_record)
#     fp.create_dataset('chi_height', data=chi_height_record)

#####################################
# History plot
hist_array = np.arange(250)
hist = Plot1d(inputfile, hist_array, video=False, staircase=False, history=True, profile_in=profile_in)
histfig = hist.history(plot=True)
histfig.savefig('./hist.png', dpi=300)

plt.show()
plt.close('all')

#########################
# Plots for publication


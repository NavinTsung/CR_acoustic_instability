# Import installed packages
import sys
import numpy as np 
import numpy.linalg as LA
import numpy.ma as ma
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import matplotlib.legend as lg
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import scipy.stats as stats
import scipy.optimize as opt
import scipy.integrate as integrate 
import scipy.signal as signal
from scipy import interpolate
from statistics import median
import h5py
from mpi4py import MPI

# Import athena data reader
athena_path = '/Users/tsunhinnavintsung/Workspace/Codes/Athena++'
if athena_path not in sys.path:
  sys.path.append(athena_path) 
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
  square_size = 0.8
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
  def __init__(self, inputfile, file_array, video=False, staircase=False, history=False, avg=False, profile_in=None, with_v=False):
    # Staircase, history are specific to this problem only
    self.inputfile = inputfile
    self.file_array = file_array
    self.with_v = with_v
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
      if self.isothermal:
        self.profile_object = Power_v_iso(profile_in) if with_v else Power_iso(profile_in)
        self.profile_object.powerprofile(self.x1min, self.x1max, self.nx1)
      else:
        self.profile_object = Power_v(profile_in) if with_v else Power(profile_in)
        self.profile_object.powerprofile(self.x1min, self.x1max, self.nx1)
    if video or staircase or history or avg:
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
  def make_video(self, equi, save_path, logx=False, logy=False, initial_data=None, xlim=None):
    filename = self.filename 
    initial = False
    if initial_data != None:
      initial = True

    # Choosing files for display
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))

    # For no adaptive mesh refinement
    x1v = np.array(ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v'])
    
    if (xlim != None):
      xl = xlim[0]
      xu = xlim[1]
      index_xl = np.argmin(np.abs(x1v - xl))
      index_xu = np.argmin(np.abs(x1v - xu))
    else:
      index_xl = 0
      index_xu = np.size(x1v)

    x1v = x1v[index_xl:index_xu]

    if initial:
      rho0 = initial_data['rho'][index_xl:index_xu]
      v0 = initial_data['v'][index_xl:index_xu]
      if not self.isothermal:
        pg0 = initial_data['pg'][index_xl:index_xu]
      if self.cr:
        ecr0 = initial_data['ec'][index_xl:index_xu]
        fc0 = initial_data['fc'][index_xl:index_xu]*self.vmax

    # Equilibrium profile
    rho_eq = equi['rho'][index_xl:index_xu] if (np.size(equi['rho']) != 1) else equi['rho']
    v_eq = equi['v'][index_xl:index_xu] if (np.size(equi['rho']) != 1) else equi['v']
    if not self.isothermal:
      pg_eq = equi['pg'][index_xl:index_xu] if (np.size(equi['pg']) != 1) else equi['pg']
    if self.cr:
      pc_eq = equi['pc'][index_xl:index_xu] if (np.size(equi['pc']) != 1) else equi['pc']
      fc_eq = equi['fc'][index_xl:index_xu] if (np.size(equi['fc']) != 1) else equi['fc']

    # Number of parameters of interest
    rho_array = np.zeros(np.size(x1v))
    v_array = np.zeros(np.size(x1v))
    if not self.isothermal:
      pg_array = np.zeros(np.size(x1v))
    if self.cr:
      ecr_array = np.zeros(np.size(x1v))
      fc_array = np.zeros(np.size(x1v))
    if self.passive:
      r_array = np.zeros(np.size(x1v))

    # Extracting data
    for i, file in enumerate(file_array):
      print(file)
      data = ar.athdf('./' + filename + '.out1.' + format(file, '05d') \
        + '.athdf')
      time = float('{0:f}'.format(data['Time']))
      rho_array = data['rho'][0, 0, index_xl:index_xu]
      v_array = data['vel1'][0, 0, index_xl:index_xu]
      if not self.isothermal:
        pg_array = data['press'][0, 0, index_xl:index_xu]
      if self.cr:
        ecr_array = data['Ec'][0, 0, index_xl:index_xu]
        fc_array = data['Fc1'][0, 0, index_xl:index_xu]*self.vmax
      if self.passive:
        r_array = data['r0'][0, 0, index_xl:index_xu]

      # Plot and save image 
      if (self.isothermal and (not self.cr)):
        fig = plt.figure(figsize=(6, 3))
        grids = gs.GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        lab = ['$\\rho$', '$v$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, v_array - v_eq, label='t={:.3f}'.format(time))
        if initial:
          ax1.plot(x1v, rho0 - rho_eq, 'k--')
          ax2.plot(x1v, v0 - v_eq, 'k--')
      elif ((not self.isothermal) and (not self.cr)):
        fig = plt.figure(figsize=(9, 3))
        grids = gs.GridSpec(1, 3, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        ax3 = fig.add_subplot(grids[0, 2])
        lab = ['$\\rho$', '$v$', '$P_g$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, v_array - v_eq, label='t={:.3f}'.format(time))
        ax3.plot(x1v, pg_array - pg_eq, label='t={:.3f}'.format(time))
        if initial:
          ax1.plot(x1v, rho0 - rho_eq, 'k--')
          ax2.plot(x1v, v0 - v_eq, 'k--')
          ax3.plot(x1v, pg0 - pg_eq, 'k--')
      elif (self.isothermal and self.cr):
        fig = plt.figure(figsize=(12, 3))
        grids = gs.GridSpec(1, 4, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        ax3 = fig.add_subplot(grids[0, 2])
        ax4 = fig.add_subplot(grids[0, 3])
        lab = ['$\\rho$', '$v$', '$P_c$', '$F_c$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, v_array - v_eq, label='t={:.3f}'.format(time))
        ax3.plot(x1v, ecr_array/3. - pc_eq, label='t={:.3f}'.format(time))
        ax4.plot(x1v, fc_array - fc_eq, label='t={:.3f}'.format(time))
        if initial:
          ax1.plot(x1v, rho0 - rho_eq, 'k--')
          ax2.plot(x1v, v0 - v_eq, 'k--')
          ax3.plot(x1v, ecr0/3. - pc_eq, 'k--')
          ax4.plot(x1v, fc0 - fc_eq, 'k--')
      elif ((not self.isothermal) and self.cr):
        # fig = plt.figure(figsize=(12, 3))
        fig = plt.figure(figsize=(6, 3))
        # grids = gs.GridSpec(1, 5, figure=fig)
        grids = gs.GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        # ax3 = fig.add_subplot(grids[0, 2])
        # ax4 = fig.add_subplot(grids[0, 3])
        # ax5 = fig.add_subplot(grids[0, 4])
        # lab = ['$\\rho$', '$v$', '$P_c$', '$P_g$', '$F_c$']
        lab = ['$\\rho$', '$P_c$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        # ax2.plot(x1v, v_array - v_eq, label='t={:.3f}'.format(time))
        # ax3.plot(x1v, ecr_array/3. - pc_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, ecr_array/3. - pc_eq, label='t={:.3f}'.format(time))
        # ax4.plot(x1v, pg_array - pg_eq, label='t={:.3f}'.format(time))
        # ax5.plot(x1v, fc_array - fc_eq, label='t={:.3f}'.format(time))
        if initial:
          ax1.plot(x1v, rho0 - rho_eq, 'k--')
          # ax2.plot(x1v, v0 - v_eq, 'k--')
          # ax3.plot(x1v, ecr0/3. - pc_eq, 'k--')
          ax2.plot(x1v, ecr0/3. - pc_eq, 'k--')
          # ax4.plot(x1v, pg0 - pg_eq, 'k--')
          # ax5.plot(x1v, fc0 - fc_eq, 'k--')
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
        if logx:
          axes.set_xscale('log')
        if logy and (axes != ax2):
          if not self.isothermal:
            if (axes != ax5):
              axes.set_yscale('log')
          else:
            axes.set_yscale('log')
        if not(logx or logy):
          axes.xaxis.set_minor_locator(AutoMinorLocator())
          axes.yaxis.set_minor_locator(AutoMinorLocator())

      if self.passive:
        ax_passive.legend(frameon=False)
        ax_passive.set_xlabel('$x$')
        ax_passive.set_ylabel('$r$')
        if logx:
          axes.set_xscale('log')
        if logy:
          axes.set_yscale('log')
        if not(logx or logy):
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

  def plot(self, logx=False, logy=False):
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
      lab.append('$T$')

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
        if self.b_field and (not self.isothermal):
          ax8.plot(x1v, 2.*self.pg_array[i, :]/self.b0**2, 'k--', label='t={:.3f}'.format(time_array[i]))
          ax9.plot(x1v, self.pg_array[i, :]/self.rho_array[i, :], 'k--', label='t={:.3f}'.format(time_array[i]))
        elif self.b_field and self.isothermal:
          ax8.plot(x1v, 2.*self.cs_iso**2*self.rho_array[i, :]/self.b0**2, 'k--', label='t={:.3f}'.format(time_array[i]))
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
        if self.b_field and (not self.isothermal):
          ax8.plot(x1v, 2.*self.pg_array[i, :]/self.b0**2, 'o-', label='t={:.3f}'.format(time_array[i]))
          ax9.plot(x1v, self.pg_array[i, :]/self.rho_array[i, :], 'o-', label='t={:.3f}'.format(time_array[i]))
        elif self.b_field and self.isothermal:
          ax8.plot(x1v, 2.*self.cs_iso**2*self.rho_array[i, :]/self.b0**2, 'o-', label='t={:.3f}'.format(time_array[i]))
        if self.passive:
          ax_passive.plot(x1v, self.r_array[i, :], label='t={:3f}'.format(time_array[i]))

    for i, axes in enumerate(ax):
      if not self.isothermal and not ax9:
        axes.legend(frameon=False)
      axes.set_xlabel('$x$')
      axes.set_ylabel(lab[i])
      if logx and (axes != ax2) and (axes != ax5) and (axes != ax6) and (axes != ax7):
        axes.set_xscale('log')
      if logy and (axes != ax2) and (axes != ax5) and (axes != ax6) and (axes != ax7): 
        axes.set_yscale('log')
      if (not logx) and (not logy) and (lab[i] != '$\\sigma_c$'):
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()

    if self.passive:
      ax_passive.legend(frameon=False)
      ax_passive.set_xlabel('$x$')
      ax_passive.set_ylabel('Concen.')
      if logx and (axes != ax2) and (axes != ax5) and (axes != ax6) and (axes != ax7):
        axes.set_xscale('log')
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
    if not self.isothermal:
      pg_sh = self.pg_array[file, ibeg:iend]
    pc_sh = self.ecr_array[file, ibeg:iend]/3.  
    fc_sh = self.fcx_array[file, ibeg:iend]*self.vmax 
    vs_sh = self.vs_array[file, ibeg:iend]
    sa_sh = self.sigma_adv_array[file, ibeg:iend]/self.vmax 
    sd_sh = self.sigma_diff_array[file, ibeg:iend]/self.vmax 
    sc_sh = sa_sh*sd_sh/(sa_sh + sd_sh)

    # Derived quantities
    if not self.isothermal:
      cs_sh = np.sqrt(gg*pg_sh/rho_sh)
    else:
      cs_sh = self.cs_iso
    cc_sh = np.sqrt(gc*pc_sh/rho_sh)
    vp_sh = np.sqrt(cs_sh**2 + cc_sh**2) 
    if self.cr_stream:
      va_sh = self.b0/np.sqrt(rho_sh)
      if not self.isothermal:
        beta = 2.*pg_sh/self.b0**2 
      else:
        beta = 2.*rho_sh*self.cs_iso**2/self.b0**2

    # Quantities involving first derivatives
    drhodx_sh = np.gradient(rho_sh, x_sh)
    dvdx_sh = np.gradient(vx_sh, x_sh)
    if not self.isothermal:
      dpgdx_sh = np.gradient(pg_sh, x_sh)
    dpcdx_sh = np.gradient(pc_sh, x_sh)
    dfcdx_sh = np.gradient(fc_sh, x_sh)
    fcsteady = gc1*pc_sh*(vx_sh + vs_sh) - dpcdx_sh/sd_sh

    # Involving second derivatives
    d2pcdx2_sh = np.gradient(dpcdx_sh, x_sh)

    # Extract upstream and downstream variables in the shock frame
    rho1, rho2 = rho_sh[istr], rho_sh[ichk]
    if not self.isothermal:
      pg1, pg2 = pg_sh[istr], pg_sh[ichk]
    pc1, pc2 = pc_sh[istr], pc_sh[ichk]
    if not self.isothermal:
      cs1, cs2 = cs_sh[istr], cs_sh[ichk]
    else:
      cs1, cs2 = cs_sh, cs_sh
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
    if not self.isothermal:
      n1, n2 = pc1/(pc1 + pg1), pc2/(pc2 + pg2)
    else:
      n1, n2 = pc1/(pc1 + rho1*cs_sh**2), pc2/(pc2 + rho2*cs_sh**2)

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
    if not self.isothermal:
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
    if not self.isothermal:
      self.dpgdx_sh = dpgdx_sh 
    self.dpcdx_sh = dpcdx_sh 
    self.dfcdx_sh = dfcdx_sh 
    self.fcsteady = fcsteady

    self.d2pcdx2_sh = d2pcdx2_sh 

    self.rho1, self.rho2 = rho1, rho2 
    if not self.isothermal:
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
  def staircase(self, plot=False, xlim=None, time_series=False, fit=False, inset=False):
    filename = self.filename 

    # Analyse only a specific region
    if (xlim != None):
      xl = xlim[0]
      xu = xlim[1]

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
    jump_loc_file = dict.fromkeys(file_array)
    height_file = dict.fromkeys(file_array)
    plateau_file = dict.fromkeys(file_array)
    ldiff_file = dict.fromkeys(file_array)
    crpress_file = dict.fromkeys(file_array)
    Lc_file = dict.fromkeys(file_array)
    coupled_mass_fraction_file = dict.fromkeys(file_array)
    pc_ratio_file = dict.fromkeys(file_array)
    shock_start_loc_file = dict.fromkeys(file_array)
    shock_end_loc_file = dict.fromkeys(file_array)
    mach_file = dict.fromkeys(file_array)
    vsh_file = dict.fromkeys(file_array)
    vsh_unitcs_file = dict.fromkeys(file_array)
    v1_unitcs_file = dict.fromkeys(file_array)

    x1v0 = ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v']
    if (xlim != None):
      index_xl0 = np.argmin(np.abs(x1v0 - xl))
      index_xu0 = np.argmin(np.abs(x1v0 - xu)) + 1
    else:
      index_xl0 = 0
      index_xu0 = np.size(x1v0)
    x1v0 = x1v0[index_xl0:index_xu0]

    # Merging plot
    couple_grid = np.zeros((np.size(file_array), np.size(x1v0)))

    # Extracting data
    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      time_array[i] = float('{0:f}'.format(data['Time']))
      # For no mesh refinement
      x1v = np.array(data['x1v'])
      dx = x1v[1] - x1v[0]
      if (xlim != None):
        index_xl = np.argmin(np.abs(x1v - xl))
        index_xu = np.argmin(np.abs(x1v - xu)) + 1
      else:
        index_xl = 0
        index_xu = np.size(x1v)
      x1v = x1v[index_xl:index_xu]
      grids = np.size(x1v)
      dx = x1v[1] - x1v[0]
      rho = data['rho'][0, 0, index_xl:index_xu]
      vx = data['vel1'][0, 0, index_xl:index_xu]
      if not self.isothermal:
        pg = data['press'][0, 0, index_xl:index_xu]
        c_sound = np.sqrt(gg*pg/rho)
      else:
        c_sound = self.cs_iso
      pc = data['Ec'][0, 0, index_xl:index_xu]*(gc - 1.)
      fc = data['Fc1'][0, 0, index_xl:index_xu]*self.vmax
      vs = data['Vc1'][0, 0, index_xl:index_xu]
      kappa = (gc - 1.)*self.vmax/data['Sigma_diff1'][0, 0, index_xl:index_xu] 
      if time_series:
        if fp == file_avg:
          rho_time_avg = rho/num_file_avg
          v_time_avg = vx/num_file_avg
          pc_time_avg = pc/num_file_avg 
        elif fp > file_avg:
          rho_time_avg += rho/num_file_avg
          v_time_avg += vx/num_file_avg
          pc_time_avg += pc/num_file_avg
      dpcdx = np.gradient(pc, x1v)
      Lc = np.zeros(grids)
      for j, gradpc in enumerate(dpcdx):
        Lc[j] = np.abs(pc[j]/gradpc) if gradpc != 0.0 else big_number
      fcsdy = gc1*pc*(vx + vs) - (kappa/(gc - 1.))*dpcdx
      dfcfc = np.abs(1. - fcsdy/fc)
      
      couple = np.zeros(grids)
      # L_equi = np.abs(self.profile_object.pc_sol/self.profile_object.dpcdx_sol)
      for j in np.arange(grids):
        # Condition 1: Fc has to be close enough to steady state Fc
        # if (dfcfc[j] < 100/self.vmax**2) and (Lc[j] < 5*L_equi[j]):
        if (dx/Lc[j] > 0.01*np.abs(vs[j])/self.vmax):
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
      couple_grid[i, :] = couple
      couple_file[fp] = couple
      num_stair_file[fp] = num_stair
      stair_start_loc_file[fp] = stair_start_location
      stair_end_loc_file[fp] = stair_end_location
      
      # Analysis
      width = np.array([])
      jump_loc = np.array([])
      height = np.array([])
      plateau = np.array([])
      ldiffusive = np.array([])
      cr_press = np.array([])
      Lc = np.array([])
      coupled_mass_fraction = np.array([])
      pc_ratio = np.array([])
      shock_start_location = np.zeros(grids)
      shock_end_location = np.zeros(grids)
      mach = np.array([])
      vsh = np.array([])
      vsh_unitcs = np.array([])
      v1_unitcs = np.array([])
      for j, x in enumerate(x1v):
        if stair_start_location[j] == 1:
          k = j 
          while (stair_end_location[k] == 0) and (k < grids-1):
            k += 1
          width = np.append(width, x1v[k] - x1v[j])
          jump_loc = np.append(jump_loc, 0.5*(x1v[k] - x1v[j]))
          height = np.append(height, np.abs(pc[k] - pc[j])) 
          if not self.isothermal:
            ldiffusive = np.append(ldiffusive, kappa[0]/np.mean(c_sound[j:(k+1)]))
            ldiff_plat = kappa[0]/np.mean(c_sound[j:(k+1)])
          else:
            ldiffusive = np.append(ldiffusive, kappa[0]/np.mean(c_sound))
            ldiff_plat = kappa[0]/np.mean(c_sound)
          cr_press = np.append(cr_press, np.mean(pc[j:(k+1)]))
          grad_pc = np.gradient(pc[j:(k+1)], x1v[j:(k+1)])
          Lc = np.append(Lc, np.mean(np.abs(pc[j:(k+1)]/grad_pc)))
          coupled_mass_fraction = np.append(coupled_mass_fraction, np.sum(rho[j:(k+1)])/np.sum(rho)) # for evenly spaced grids
          pc_ratio = np.append(pc_ratio, np.amin(pc[j:(k+1)])/np.amax(pc[j:(k+1)]))
          # Search for shock start and end
          search_grid = 20 
          if (j > search_grid) and (j < (grids - search_grid)):
            start_loc = np.argmin(rho[(j-search_grid):j])
            end_loc = np.argmax(rho[j:(j+search_grid)])
            absolute_start_loc = j - search_grid+start_loc
            absolute_end_loc = j + end_loc
            rho_start = rho[absolute_start_loc]
            rho_end = rho[absolute_end_loc]
            v_start = vx[absolute_start_loc]
            v_end = vx[absolute_end_loc]
            if not self.isothermal:
              cs_start = c_sound[absolute_start_loc]
              cs_end = c_sound[absolute_end_loc]
            else:
              cs_start = self.cs_iso
              cs_end = self.cs_iso
            v_sh = (rho_end*v_end - rho_start*v_start)/(rho_end - rho_start)
            v_shock_frame = vx - v_sh
            v_shock_frame_start = v_shock_frame[absolute_start_loc]
            v_shock_frame_end = v_shock_frame[absolute_end_loc]
            mach_start = v_shock_frame_start/cs_start
            if ((v_shock_frame_start > v_shock_frame_end) and (mach_start > 1.) and (absolute_end_loc > absolute_start_loc)):
              shock_start_location[absolute_start_loc] = 1 
              shock_end_location[absolute_end_loc] = 1
              mach = np.append(mach, mach_start)
              vsh = np.append(vsh, v_sh)
              vsh_unitcs = np.append(vsh_unitcs, v_sh/cs_start)
        if stair_end_location[j] == 1:
          k = j
          while (stair_start_location[k] == 0) and (k < grids-1):
            k += 1
          try:
            plateau = np.append(plateau, (x1v[k] - x1v[j])/ldiff_plat)
          except:
            pass

      width_file[fp] = width
      jump_loc_file[fp] = jump_loc
      height_file[fp] = height
      plateau_file[fp] = plateau
      ldiff_file[fp] = ldiffusive
      crpress_file[fp] = cr_press 
      Lc_file[fp] = Lc 
      coupled_mass_fraction_file[fp] = coupled_mass_fraction
      pc_ratio_file[fp] = pc_ratio 
      shock_start_loc_file[fp] = shock_start_location 
      shock_end_loc_file[fp] = shock_end_location
      mach_file[fp] = mach
      vsh_file[fp] = vsh 
      vsh_unitcs_file[fp] = vsh_unitcs

    # Save variable
    self.time_array = time_array
    self.x1v = x1v
    self.size_x1v = np.size(x1v)
    self.index_xl = index_xl
    self.index_xu = index_xu
    self.couple_grid = couple_grid
    self.couple = couple_file
    self.num_stair = num_stair_file
    self.stair_start_loc = stair_start_loc_file
    self.stair_end_loc = stair_end_loc_file
    self.width = width_file
    self.jump_loc = jump_loc_file
    self.height = height_file
    self.plateau = plateau_file
    self.ldiffusive = ldiff_file
    self.cr_press = crpress_file
    self.Lc = Lc_file
    self.coupled_mass_fraction = coupled_mass_fraction_file
    self.pc_ratio = pc_ratio_file 
    self.shock_start_loc = shock_start_loc_file 
    self.shock_end_loc = shock_end_loc_file
    self.mach = mach_file
    self.vsh = vsh_file 
    self.vsh_unitcs = vsh_unitcs_file
    if time_series:
      self.rho_time_avg = rho_time_avg
      self.v_time_avg = v_time_avg
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
      time_pc = float(input('Enter time to display pc staircase: '))
      display_index = np.argmin(np.abs(self.time_array - time))
      display_file = file_array[np.argmin(np.abs(self.time_array - time))]
      display_file_pc = file_array[np.argmin(np.abs(self.time_array - time_pc))]
      data = ar.athdf('./' + self.filename + '.out1.' + format(display_file_pc, '05d') + '.athdf')
      x1v_display = np.array(data['x1v'])
      pc_display = data['Ec'][0, 0, :]*(gc - 1.)
      rho_display = data['rho'][0, 0, :]

      width_record = np.array([])
      plateau_record = np.array([])
      height_record = np.array([])
      ldiff_record = np.array([])
      pc_record = np.array([])
      Lc_record = np.array([])
      for i, file in enumerate(file_array[display_index:]):
        width_record = np.append(width_record, self.width[file])
        plateau_record = np.append(plateau_record, self.plateau[file])
        height_record = np.append(height_record, self.height[file])
        ldiff_record = np.append(ldiff_record, self.ldiffusive[file])
        pc_record = np.append(pc_record, self.cr_press[file])
        Lc_record = np.append(Lc_record, self.Lc[file])

      self.width_record = width_record
      self.plateau_record = plateau_record 
      self.height_record = height_record
      self.ldiff_record = ldiff_record
      self.pc_record = pc_record 
      self.Lc_record = Lc_record

      # fig1 = plt.figure(figsize=(13, 6))
      fig1 = plt.figure()
      fig2 = plt.figure()
      fig3 = plt.figure()

      grids = gs.GridSpec(1, 1, figure=fig2)
      ax1 = fig1.add_subplot(grids[0, 0])
      # ax2 = fig1.add_subplot(grids[0, 1])
      # ax3 = fig1.add_subplot(grids[0, 0])
      # ax = [ax1, ax2, ax3]
      ax = [ax1]
      ax4 = fig2.add_subplot(111)
      ax5 = fig3.add_subplot(111)

      num_bins = 100 
      width_bin = np.logspace(np.log10(np.amin(width_record/ldiff_record)), np.log10(np.amax(width_record/ldiff_record)), num_bins)
      plateau_bin = np.logspace(np.log10(np.amin(plateau_record)), np.log10(np.amax(plateau_record)), num_bins)
      height_bin = np.logspace(np.log10(np.amin(height_record/(pc_record*ldiff_record/Lc_record))), np.log10(np.amax(height_record/(pc_record*ldiff_record/Lc_record))), num_bins)

      widths = (width_bin[1:] - width_bin[:-1])
      plateaus = (plateau_bin[1:] - plateau_bin[:-1])
      heights = (height_bin[1:] - height_bin[:-1])

      width_hist = np.histogram(width_record/ldiff_record, bins=width_bin)
      plateau_hist = np.histogram(plateau_record, bins=plateau_bin)
      height_hist = np.histogram(height_record, bins=height_bin)

      width_norm = width_hist[0]/(widths*np.sum(width_hist[0]))
      plateau_norm = plateau_hist[0]/(plateaus*np.sum(plateau_hist[0]))
      height_norm = height_hist[0]/(heights*np.sum(height_hist[0]))

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

        # width_bin_fit_log = np.log(width_bin_fit/ldiff_avg)
        width_bin_fit_log = np.log(width_bin_fit)
        width_norm_fit_log = np.log(width_norm_fit)

        plateau_bin_fit_log = np.log(plateau_bin_fit/ldiff_avg)
        plateau_norm_fit_log = np.log(plateau_norm_fit)

        height_bin_fit_log = np.log(height_bin_fit)
        height_norm_fit_log = np.log(height_norm_fit)

        width_begin = ((self.x1max - self.x1min)/self.nx1)
        width_index = np.argmin(np.abs(width_bin_fit - width_begin))

        plateau_begin = ((self.x1max - self.x1min)/self.nx1)
        plateau_index = np.argmin(np.abs(plateau_bin_fit - plateau_begin))

        height_begin = np.mean(pc)*((self.x1max - self.x1min)/self.nx1)/np.mean(L)
        height_index = np.argmin(np.abs(height_bin_fit - height_begin))

        self.width_bin_fit_log = width_bin_fit_log 
        self.width_norm_fit_log = width_norm_fit_log
        self.widths_fit = np.exp(width_bin_fit_log[1:]) - np.exp(width_bin_fit_log[:-1])

        self.height_bin_fit_log = height_bin_fit_log 
        self.height_norm_fit_log = height_norm_fit_log 
        self.heights_fit = np.exp(height_bin_fit_log[1:]) - np.exp(height_bin_fit_log[:-1])

        try:
          width_opt, width_cov = opt.curve_fit(fit_func, width_bin_fit_log[:-1], width_norm_fit_log, \
            p0=(np.amax(width_norm_fit_log), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
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
          height_opt, height_cov = opt.curve_fit(fit_func, height_bin_fit_log[:-1], height_norm_fit_log, \
            p0=(np.amax(height_norm_fit_log), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
          height_fit_success = True
          height_amp, height_alpha, height_beta, height_chi = height_opt
          self.height_alpha, self.height_beta, self.height_chi = height_alpha, height_beta, height_chi
        except:
          print('Cannot fit height')
          height_fit_success = False
          self.height_alpha, self.height_beta, self.height_chi = None, None, None

        self.eta = self.profile_object.kappa/(gc*L[0]*cs[0])
        self.ldiff_avg, self.pc_height_avg = ldiff_avg, pc_height_avg 


      # ax1.bar(width_bin[:-1], width_norm, widths)
      # ax1.bar(np.exp(width_bin_fit_log[:-2]), np.exp(width_norm_fit_log[:-1]), self.widths_fit[:-1], align='edge')
      # ax2.bar(plateau_bin[:-1], plateau_norm, plateaus)
      # ax3.bar(height_bin[:-1], height_norm, heights)
      # ax3.bar(np.exp(height_bin_fit_log[:-2]), np.exp(height_norm_fit_log[:-1]), self.heights_fit[:-1], align='edge')

      # if fit:
        # if width_fit_success:
        #   ax1.loglog(np.exp(width_bin_fit_log[:-1]), \
        #     np.exp(fit_func(width_bin_fit_log[:-1], width_amp, width_alpha, width_beta, width_chi)), 'k--', \
        #     label='$\\alpha={:.2f},\\beta={:.2f},\\chi={:.2f}$'.format(width_alpha, width_beta, width_chi))
          # ax1.legend(frameon=False)
        # if plateau_fit_success:
        #   ax2.loglog(np.exp(plateau_bin_fit_log[:-1])*ldiff_avg, \
        #     np.exp(fit_func(plateau_bin_fit_log[:-1], plateau_amp, plateau_alpha, plateau_beta, plateau_chi)), 'k--', \
        #     label='$\\alpha={:.2f},\\beta={:.2f},\\chi={:.2f}$'.format(plateau_alpha, plateau_beta, plateau_chi))
        #   ax2.legend(frameon=False)
        # if height_fit_success:
        #   ax3.loglog(np.exp(height_bin_fit_log[:-1])*pc_height_avg, \
        #     np.exp(fit_func(height_bin_fit_log[:-1], height_amp, height_alpha, height_beta, height_chi)), 'k--', \
        #     label='$\\alpha={:.2f},\\beta={:.2f},\\chi={:.2f}$'.format(height_alpha, height_beta, height_chi))
        #   ax3.legend(frameon=False)

        # ax1.axvline(x=((stair.x1max - stair.x1min)/stair.nx1), color='r', linestyle='--')
        # ax2.axvline(x=((stair.x1max - stair.x1min)/stair.nx1), color='r', linestyle='--')
        # ax3.axvline(x=np.mean(pc)*((stair.x1max - stair.x1min)/stair.nx1)/np.mean(L), color='r', linestyle='--')

      ax4.plot(x1v_display, pc_display, 'o-')
      ax5.plot(x1v_display, rho_display, 'o-')
      for i, x in enumerate(x1v_display[index_xl:index_xu]): 
        if self.stair_start_loc[display_file_pc][i] == 1: 
          ax4.axvline(x1v_display[index_xl+i], linestyle='--', color='g')
        if self.stair_end_loc[display_file_pc][i] == 1:
          ax4.axvline(x1v_display[index_xl+i], linestyle='--', color='r')
        if self.shock_start_loc[display_file_pc][i] == 1:
          ax5.axvline(x1v_display[index_xl+i], linestyle='--', color='b')
        if self.shock_end_loc[display_file_pc][i] == 1:
          ax5.axvline(x1v_display[index_xl+i], linestyle='--', color='orange')

      if inset:
        add_box = 'y'
        ax_in_number = dict()
        number = 1
        while add_box == 'y':
          name = 'ax_in{:d}'.format(number)
          x_box = float(input('x location of the lower left corner: '))
          y_box = float(input('y location of the lower left corner: '))
          w_box = float(input('Width of the box: '))
          h_box = float(input('Height of the box: '))
          xlim_start = float(input('xlim start: '))
          xlim_end = float(input('xlim end: '))
          ylim_start = float(input('ylim start: '))
          ylim_end = float(input('ylim end: '))
          ax_in_number[name] = ax4.inset_axes([x_box, y_box, w_box, h_box], transform=ax4.transData)
          ax_in_number[name].plot(x1v_display, pc_display, 'o-')
          for i, x in enumerate(x1v_display[index_xl:index_xu]): 
            if self.stair_start_loc[display_file_pc][i] == 1: 
              ax_in_number[name].axvline(x1v_display[index_xl+i], linestyle='--', color='g')
            if self.stair_end_loc[display_file_pc][i] == 1:
              ax_in_number[name].axvline(x1v_display[index_xl+i], linestyle='--', color='r')
          ax_in_number[name].set_xlim(xlim_start, xlim_end)
          ax_in_number[name].set_ylim(ylim_start, ylim_end) 
          ax_in_number[name].set_xticklabels('')
          ax_in_number[name].set_yticklabels('')
          ax4.indicate_inset_zoom(ax_in_number[name])
          add_box = input('Want to add another box (y/n)?: ')
          number += 1


      # lab = ['Width ($l_\\mathrm{diff}$)', 'Plateau width', '$P_c$ Height']
      lab = ['Width ($l_\\mathrm{diff}$)']
      # lab = ['Height ($P_c l_\\mathrm{diff}/L_c$)']

      for i, axes in enumerate(ax):
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel(lab[i])
        axes.set_ylabel('Distribution')
        axes.set_xlim(right=1.)
        
      ax4.margins(x=0)
      ax4.set_xlabel('$x$')
      ax4.set_ylabel('$P_c$')
      ax4.set_title('$t={:.2f}$'.format(time_pc))
      ax4.xaxis.set_minor_locator(AutoMinorLocator())
      ax4.yaxis.set_minor_locator(AutoMinorLocator())

      fig1.tight_layout()
      fig2.tight_layout()
     
      if time_series:
        fig3 = plt.figure()
        fig4 = plt.figure()
        fig5 = plt.figure()
        ax5 = fig3.add_subplot(111)
        ax6 = fig4.add_subplot(111)
        ax7 = fig5.add_subplot(111)

        for i, key in enumerate(self.num_stair.keys()):
          ax5.scatter(stair.time_array[i], stair.num_stair[key], color='b')

        ax6.plot(self.profile_object.x_sol, self.profile_object.pc_sol, 'k--', label='$t=0$')
        ax6.plot(x1v, pc_time_avg, label='$\\langle P_c \\rangle$')

        if self.with_v:
          ax7.plot(self.profile_object.x_sol, self.profile_object.v_sol, 'k--', label='$t=0$')
        else:
          ax7.plot(x1v, np.zeros(np.size(x1v)), 'k--', label='$t=0$')
        ax7.plot(x1v, v_time_avg, label='$\\langle v \\rangle$')

        ax5.set_xlabel('Time')
        ax5.set_ylabel('Number of stairs')
        ax5.xaxis.set_minor_locator(AutoMinorLocator())
        ax5.yaxis.set_minor_locator(AutoMinorLocator())

        ax6.set_xlabel('$x$')
        ax6.set_ylabel('$P_c$')
        ax6.xaxis.set_minor_locator(AutoMinorLocator())
        ax6.yaxis.set_minor_locator(AutoMinorLocator())

        ax7.set_xlabel('$x$')
        ax7.set_ylabel('$v$')
        ax7.xaxis.set_minor_locator(AutoMinorLocator())
        ax7.yaxis.set_minor_locator(AutoMinorLocator())

        fig3.tight_layout()
        fig4.tight_layout()
        fig5.tight_layout()

        plotdefault()

        return [fig1, fig2, fig3, fig4, fig5]


      return [fig1, fig2]
    return

  def convexhull(self, xlim=None, plot=False, video=False, save_path=None, average=False):
    filename = self.filename 

    # Analyse only a specific region
    if xlim != None:
      xl = xlim[0]
      xu = xlim[1]

    # Choosing files for display
    file_array = self.file_array 
    num_of_file = np.size(file_array)
    time_array = np.zeros(np.size(file_array))
    delta_pc_predict = np.zeros(np.size(file_array))
    delta_pc_real = np.zeros(np.size(file_array))
    x_ori = dict.fromkeys(file_array)
    rho_ori = dict.fromkeys(file_array)
    pc_ori = dict.fromkeys(file_array)
    rho_stair = dict.fromkeys(file_array)
    pc_stair = dict.fromkeys(file_array)
    x_stair = dict.fromkeys(file_array)

    if average:
      x1v0 = ar.athdf('./' + filename + '.out1.' + format(0, '05d') + '.athdf')['x1v']
      if xlim:
        index_xl = np.argmin(np.abs(x1v0 - xl))
        index_xu = np.argmin(np.abs(x1v0 - xu)) + 1
      else:
        index_xl = 0
        index_xu = np.size(x1v0)
      x1v0 = x1v0[index_xl:index_xu]
      grids = np.size(x1v0)
      pc_avg = np.zeros(grids)
      pc_predict_avg = np.zeros(grids)

    # Extracting data
    for ip, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      time_array[ip] = float('{0:f}'.format(data['Time']))
      # For no mesh refinement
      x1v = np.array(data['x1v'])
      if xlim != None:
        index_xl = np.argmin(np.abs(x1v - xl))
        index_xu = np.argmin(np.abs(x1v - xu)) + 1
      else:
        index_xl = 0
        index_xu = np.size(x1v)
      x1v = x1v[index_xl:index_xu]
      grids = np.size(x1v)
      dx = x1v[1] - x1v[0]
      if self.with_v:
        b0 = data['Bcc1'][0, 0, 0]
        rho = 1./(data['vel1'][0, 0, index_xl:index_xu] + b0/np.sqrt(data['rho'][0, 0, index_xl:index_xu]))
      else:
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
      peak_higher_than_inject = False
      # cr_const = np.array([pc_max/rho_max**(gc/2.)]) if self.with_v == False else np.array([pc_max/rho_max**(gc)])
      for i, r in enumerate(rho_level):
        mask_rho = ma.masked_array(rho, mask_loc)
        max_loc = np.where(np.abs(mask_rho - r) < tol)[0]
        loc_stair = np.append(loc_stair, max_loc)
        if np.size(max_loc) > 0:
          mask_loc[max_loc[0]:max_loc[-1]+1] = 1
          if (np.sort(max_loc)[0] != 0) and (peak_higher_than_inject == False):
            mask_loc[0:max_loc[0]] = 1
            peak_higher_than_inject = True
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
      if peak_higher_than_inject:
        x_stair[fp] = np.insert(x_stair[fp], 0, x1v[0])
        rho_stair[fp] = np.insert(rho_stair[fp], 0, rho_max)
      # if self.with_v:
      #   fit_pc = lambda pc0: np.sum(pc0*(rho_stair[fp]/rho_max)**(gc) - pc[loc_stair])**2
      # else:
      #   fit_pc = lambda pc0: np.sum(pc0*(rho_stair[fp]/rho_max)**(gc/2.) - pc[loc_stair])**2
      # pc_fit = opt.minimize(fit_pc, pc_max)
      # pc_stair[fp] = pc_fit.x[0]*(rho_stair[fp]/rho_max)**(gc/2.)
      pc_stair[fp] = pc_max*(rho_stair[fp]/rho_max)**(gc/2.) if self.with_v == False else pc_max*(rho_stair[fp]/rho_max)**(gc)
      x_ori[fp] = x1v 
      rho_ori[fp] = rho 
      pc_ori[fp] = pc
      delta_pc_real[ip] = np.amax(pc) - np.amin(pc)
      delta_pc_predict[ip] = np.amax(pc_stair[fp]) - np.amin(pc_stair[fp])

      if average:
        pc_avg += pc/num_of_file 
        iloc = 0
        pc_predict_draft = np.zeros(grids)
        for i, xp in enumerate(x_stair[fp]):
          jloc = np.argmin(np.abs(x1v - xp))
          pc_predict_draft[iloc:(jloc+1)] = pc_stair[fp][i]
          iloc = jloc
        pc_predict_avg += pc_predict_draft/num_of_file

      if video:
        try:
          fig = plt.figure(figsize=(6, 3))
          grids = gs.GridSpec(1, 2, figure=fig)
          ax1 = fig.add_subplot(grids[0, 0])
          ax2 = fig.add_subplot(grids[0, 1])
          ax = [ax1, ax2]

          ax1.scatter(x_ori[fp], rho_ori[fp], label='t={:.3f}'.format(time_array[ip]))
          ax1.plot(x_stair[fp], rho_stair[fp], 'k--')
          ax2.scatter(x_ori[fp], pc_ori[fp], label='t={:.3f}'.format(time_array[ip]))
          ax2.plot(x_stair[fp], pc_stair[fp], 'k--')

          lab = ['$\\rho$', '$P_c$'] if self.with_v == False else ['$(v + v_A)^{-1}$', '$P_c$']

          for i, axes in enumerate(ax):
            axes.legend(frameon=False)
            axes.set_xlabel('$x$')
            axes.set_ylabel(lab[i])
            axes.xaxis.set_minor_locator(AutoMinorLocator())
            axes.yaxis.set_minor_locator(AutoMinorLocator())

          fig.tight_layout()
          video_path = save_path + 'convex_video{}.png'.format(fp)
          fig.savefig(video_path, dpi=300)
          plt.close('all')
        except:
          plt.close('all')
          pass

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
    self.delta_pc_predict = delta_pc_predict
    self.delta_pc_real = delta_pc_real 
    if average:
      self.pc_avg = pc_avg 
      self.pc_predict_avg = pc_predict_avg

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

      lab = ['$\\rho$', '$P_c$'] if self.with_v == False else ['$(v + v_A)^{-1}$', '$P_c$']

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

  def convexhull2(self, plot=False, logx=False, logy=False):
    filename = self.filename
    file_array = self.file_array 

    time_array = np.zeros(np.size(file_array))
    vcrsh_file = dict.fromkeys(file_array)
    pc_stair_file = dict.fromkeys(file_array)
    pc_real_file = dict.fromkeys(file_array)

    # Extracting data
    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') + '.athdf')
      time_array[i] = float('{0:f}'.format(data['Time']))
      x1v = np.array(data['x1v'])
      dx = x1v[1] - x1v[0]
      grids = np.size(x1v)
      # Extract data
      rho = data['rho'][0, 0, :]
      vx = data['vel1'][0, 0, :]
      pg = data['press'][0, 0, :]
      pc = data['Ec'][0, 0, :]*(gc - 1.)
      vs = data['Vc1'][0, 0, :]
      b0 = data['Bcc1'][0, 0, 0]
      dpcdx = np.gradient(pc, x1v)
      Lc = np.zeros(grids)
      for j, gradpc in enumerate(dpcdx):
        Lc[j] = np.abs(pc[j]/gradpc) if gradpc != 0.0 else big_number 

      couple = np.zeros(grids)
      for k in np.arange(grids):
        if (dx/Lc[k] > 0.01*np.abs(vs[k])/self.vmax):
          couple[k] = 1
        else:
          couple[k] = 0 
      # Tightly enclosed friends are also coupled
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
        for l in np.arange(grids):
          if (l >= link_grids) and (l < grids-link_grids):
            check_for = couple[l:l+link_grids+1]
            check_back = couple[l-link_grids:l+1]
            if np.any(check_for == 1) and np.any(check_back == 1):
              new_couple[l] = 1 
            else:
              new_couple[l] = 0
          else:
            new_couple[l] = couple[l]
        # Count the stairs
        toggle_start = False 
        search_grid = 10
        for m in np.arange(1, grids-1):
          if new_couple[m] == 1:
            if (new_couple[m-1] == 0) and (new_couple[m+1] == 1) and (toggle_start == False):
              num_stair += 1 
              stair_start_location[m] = 1 
              toggle_start = True 
            if (new_couple[m-1] == 1) and (new_couple[m+1] == 0) and (toggle_start == True):
              stair_end_location[m] = 1 
              toggle_start = False 
        for r in np.arange(1, grids-1):
          if stair_start_location[r] == 1:
            s = r 
            while (stair_end_location[s] == 0) and (s < grids):
              s += 1 
            min_grid = np.argmin(vx[r:s] + b0/np.sqrt(rho[r:s]))
            if min_grid != 0:
              stair_start_location[r] = 0 
              stair_start_location[r+min_grid] = 1
        couple = new_couple 
        new_couple = np.zeros(grids)

      # Construct convexhull
      vcr_sh = np.array([])
      pc_stair = np.zeros(grids)
      first_jump = True
      last_jump = False
      for n, x in enumerate(x1v):
        if stair_start_location[n] == 1:
          q = n 
          if np.all(stair_start_location[n+1:] == 0):
            last_jump = True
          while (stair_end_location[q] == 0) and (q < grids):
            q += 1 
          pc1, pc2 = pc[n], pc[q] 
          v1lab, v2lab = vx[n], vx[q]
          va1, va2 = b0/np.sqrt(rho[n]), b0/np.sqrt(rho[q])
          vcrsh = ((pc2/pc1)**(1./gc)*(v2lab + va2) - (v1lab + va1))/((pc2/pc1)**(1./gc) - 1.)
          # Store vcrsh data
          rho1, rho2 = rho[n], rho[q] 
          pg1, pg2 = pg[n], pg[q]
          cs1, cs2 = np.sqrt(gg*pg1/rho1), np.sqrt(gg*pg2/rho2)
          vcr_sh = np.append(vcr_sh, vcrsh/cs2)
          vpvamvsh1 = v1lab + va1 - vcrsh
          vpvamvsh = vx[n:q+1] + b0/np.sqrt(rho[n:q+1]) - vcrsh
          if first_jump:
            pc_stair[0:n] = pc[0:n]
            pc_stair[n:q+1] = pc1*(vpvamvsh1/vpvamvsh)**gc 
            pc_previous = pc_stair[q]
            q_previous = q
            first_jump = False
          else:
            pc_stair[q_previous:n] = pc_previous
            pc_stair[n:q+1] = pc_previous*(vpvamvsh1/vpvamvsh)**gc 
            pc_previous = pc_stair[q]
            q_previous = q 
          if last_jump:
            pc_stair[q:] = pc_previous 

      vcrsh_file[fp] = vcr_sh
      pc_stair_file[fp] = pc_stair 
      pc_real_file[fp] = pc

    # Save data
    self.x1v = x1v 
    self.time_array = time_array
    self.vcrsh = vcrsh_file
    self.pc_stair = pc_stair_file
    self.pc_real = pc_real_file 

    if plot:
      time = float(input('Enter time to construct convex hull: '))
      file = file_array[np.argmin(np.abs(self.time_array - time))]

      fig1 = plt.figure()
      ax1 = fig1.add_subplot(111)

      ax1.scatter(x1v, pc_real_file[file], label='t={:.3f}'.format(time))
      ax1.plot(x1v, pc_stair_file[file], 'k--')

      ax1.legend(frameon=False)
      ax1.set_xlabel('$x$')
      ax1.set_ylabel('$P_c$')
      if logx:
        ax1.set_xscale('log')
      else:
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
      if logy:
        ax1.set_yscale('log')
      else:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

      fig1.tight_layout() 

      return fig1 

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
    v_hist = np.zeros(np.size(file_array))
    mom_hist = np.zeros(np.size(file_array)) 
    kin_hist = np.zeros(np.size(file_array))
    if not self.isothermal:
      therm_hist = np.zeros(np.size(file_array))
    if self.cr:
      ec_hist = np.zeros(np.size(file_array))
      fc_hist = np.zeros(np.size(file_array))

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
        fc = data['Fc1'][0, 0, :]*self.vmax
      passive_count = np.where(r > 0.5)[0]
      x_hist[i] = np.mean(x1v[passive_count])
      den_hist[i] = np.mean(rho[passive_count])
      v_hist[i] = np.mean(vx[passive_count])
      mom_hist[i] = np.mean(rho[passive_count]*vx[passive_count]) 
      kin_hist[i] = np.mean(0.5*rho[passive_count]*vx[passive_count]**2) 
      if not self.isothermal:
        therm_hist[i] = np.mean(pg[passive_count]/(gg - 1.))
      if self.cr:
        ec_hist[i] = np.mean(ec[passive_count])
        fc_hist[i] = np.mean(fc[passive_count])

    # Save data
    self.time_array = time_array
    self.x_hist = x_hist
    self.den_hist = den_hist 
    self.v_hist = v_hist
    self.mom_hist = mom_hist 
    self.kin_hist = kin_hist 
    if not self.isothermal:
      self.therm_hist = therm_hist
    if self.cr:
      self.ec_hist = ec_hist
      self.fc_hist = fc_hist

    if plot:
      fig = plt.figure()
      # fig = plt.figure(figsize=(12, 4))
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
      if not self.isothermal:
        ax4.plot(time_array, therm_hist)
      ax5.plot(time_array, ec_hist)
      ax6.semilogy(time_array, kin_hist, label='$\\langle E_\\mathrm{kin} \\rangle$')
      if not self.isothermal:
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

      # Second plot
      fig2 = plt.figure()
      # fig2 = plt.figure(figsize=(8, 4))
      grids2 = gs.GridSpec(2, 2, figure=fig2)
      ax21 = fig2.add_subplot(grids2[0, 0])
      ax22 = fig2.add_subplot(grids2[0, 1])
      ax23 = fig2.add_subplot(grids2[1, 0])
      ax24 = fig2.add_subplot(grids2[1, 1])

      if not self.isothermal:
        lab = ['$\\rho$', '$v$', '$P_g$', '$P_c$']
      else:
        lab = ['$\\rho$', '$v$', '$F_c$', '$P_c$']
      ax21.plot(x_hist, den_hist)
      ax22.plot(x_hist, v_hist)
      if not self.isothermal:
        ax23.plot(x_hist, (gg - 1.)*therm_hist)
      elif self.cr:
        ax23.plot(x_hist, fc_hist)
      if self.cr:
        ax24.plot(x_hist, (gc - 1.)*ec_hist)
      if self.profile_object != None:
        ax21.plot(self.profile_object.x_sol, self.profile_object.rho_sol, 'k--')
        if self.with_v:
          ax22.plot(self.profile_object.x_sol, self.profile_object.v_sol, 'k--')
        else:
          ax22.plot(self.profile_object.x_sol, np.zeros(np.size(self.profile_object.x_sol)), 'k--')
        if not self.isothermal:
          ax23.plot(self.profile_object.x_sol, self.profile_object.pg_sol, 'k--')
        elif self.cr:
          ax23.plot(self.profile_object.x_sol, self.profile_object.fc_sol, 'k--')
        ax24.plot(self.profile_object.x_sol, self.profile_object.pc_sol, 'k--')

      for i, axes in enumerate(fig2.axes):
        axes.set_xlabel('x')
        axes.set_ylabel(lab[i])
        axes.margins(x=0)
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

      fig.tight_layout()
      fig2.tight_layout()
    return [fig, fig2]

  def history2(self, start, plot=False):
    filename = self.filename 
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))

    x_hist = np.zeros(np.size(file_array))
    den_hist = np.zeros(np.size(file_array))
    v_hist = np.zeros(np.size(file_array))
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
      x_index = np.argmin(np.abs(x1v - start)) if (i == 0) else np.argmin(np.abs(x1v - x))
      x_hist[i] = x1v[x_index]
      rho = data['rho'][0, 0, x_index]
      vx = data['vel1'][0, 0, x_index]
      if not self.isothermal:
        pg = data['press'][0, 0, x_index]
      if self.cr:
        ec = data['Ec'][0, 0, x_index] 
      x = start + vx*self.dt if (i == 0) else x + vx*self.dt
      den_hist[i] = rho 
      v_hist[i] = vx 
      mom_hist[i] = rho*vx 
      kin_hist[i] = 0.5*rho*vx**2 
      if not self.isothermal:
        therm_hist[i] = pg/(gg - 1.)
      if self.cr:
        ec_hist[i] = ec

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
      if not self.isothermal:
        ax4.plot(time_array, therm_hist)
      ax5.plot(time_array, ec_hist)
      ax6.semilogy(time_array, kin_hist, label='$\\langle E_\\mathrm{kin} \\rangle$')
      if not self.isothermal:
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

      # Second plot
      fig2 = plt.figure(figsize=(8, 4))
      grids2 = gs.GridSpec(2, 2, figure=fig2)
      ax21 = fig2.add_subplot(grids2[0, 0])
      ax22 = fig2.add_subplot(grids2[0, 1])
      ax23 = fig2.add_subplot(grids2[1, 0])
      ax24 = fig2.add_subplot(grids2[1, 1])

      lab = ['$\\rho$', '$v$', '$P_g$', '$P_c$']
      ax21.plot(x_hist, den_hist)
      ax22.plot(x_hist, v_hist)
      ax23.plot(x_hist, (gg - 1.)*therm_hist)
      ax24.plot(x_hist, (gc - 1.)*ec_hist)
      if self.profile_object != None:
        ax21.plot(self.profile_object.x_sol, self.profile_object.rho_sol, 'k--')
        ax22.plot(self.profile_object.x_sol, self.profile_object.v_sol, 'k--')
        ax23.plot(self.profile_object.x_sol, self.profile_object.pg_sol, 'k--')
        ax24.plot(self.profile_object.x_sol, self.profile_object.pc_sol, 'k--')

      for i, axes in enumerate(fig2.axes):
        axes.set_xlabel('x')
        axes.set_ylabel(lab[i])
        axes.margins(x=0)
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

      fig.tight_layout()
      fig2.tight_layout()
      return [fig, fig2]
    return

  def time_avg(self, plot=False, logx=False, logy=False, compare=False, xlim=None):
    filename = self.filename
    file_array = self.file_array 
    num_file = np.size(file_array)
    time_array = np.zeros(num_file)

    if (xlim != None):
      xl = xlim[0]
      xu = xlim[1]

    x_avg = ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v']
    if (xlim != None):
      index_xl = np.argmin(np.abs(x_avg - xl))
      index_xu = np.argmin(np.abs(x_avg - xu)) + 1
    else:
      index_xl = 0
      index_xu = np.size(x_avg)
    x_avg = x_avg[index_xl:index_xu]
    dx = x_avg[1] - x_avg[0]
    rho_avg = np.zeros(np.size(x_avg))
    v_avg = np.zeros(np.size(x_avg))
    rhov_avg = np.zeros(np.size(x_avg))
    if not self.isothermal:
      pg_avg = np.zeros(np.size(x_avg))
    if self.cr:
      pc_avg = np.zeros(np.size(x_avg))
      fc_avg = np.zeros(np.size(x_avg))
      va_avg = np.zeros(np.size(x_avg))
      sc_avg = np.zeros(np.size(x_avg))
      ev_avg = np.zeros(np.size(x_avg))
      steady_fc_avg = np.zeros(np.size(x_avg))
      cr_pushing_avg = np.zeros(np.size(x_avg))
      cr_heating_avg = np.zeros(np.size(x_avg))
      cr_workdone_avg = np.zeros(np.size(x_avg))
      cr_couple_avg = np.zeros(np.size(x_avg))
      invariant = np.zeros(np.size(x_avg))
      total_couple_heating_time = np.zeros(num_file)
      total_general_heating_time = np.zeros(num_file)
      dfc_time = np.zeros(num_file)

    if compare:
      rho0 = np.zeros(np.size(x_avg))
      v0 = np.zeros(np.size(x_avg))
      if not self.isothermal:
        pg0 = np.zeros(np.size(x_avg))
      if self.cr:
        pc0 = np.zeros(np.size(x_avg))
        fc0 = np.zeros(np.size(x_avg))
        va0 = np.zeros(np.size(x_avg))

    kappa = (gc - 1.)*self.vmax/ar.athdf ('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['Sigma_diff1'][0, 0, 0]

    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      time_array[i] = float('{0:f}'.format(data['Time']))
      rho = data['rho'][0, 0, index_xl:index_xu]
      vx = data['vel1'][0, 0, index_xl:index_xu]
      if not self.isothermal:
        pg = data['press'][0, 0, index_xl:index_xu]
      if self.cr:
        ec = data['Ec'][0, 0, index_xl:index_xu] 
        fc = data['Fc1'][0, 0, index_xl:index_xu]*self.vmax
        vs = data['Vc1'][0, 0, index_xl:index_xu]
        va = np.abs(data['Vc1'][0, 0, index_xl:index_xu])
        ss = data['Sigma_adv1'][0, 0, index_xl:index_xu]/self.vmax
        sd = data['Sigma_diff1'][0, 0, index_xl:index_xu]/self.vmax
        sc = 1./(1./ss + 1./sd)
      rho_avg += rho/num_file
      v_avg += vx/num_file
      rhov_avg += rho*vx/num_file 
      if not self.isothermal:
        pg_avg += pg/num_file
      if self.cr:
        pc_avg += ec*(gc - 1.)/num_file
        fc_avg += fc/num_file
        va_avg += va/num_file
        sc_avg += sc/num_file
        ev_avg += ec*vx/num_file
        steady_fc_avg += gc*ec*(vx + va)/num_file 
        cr_pushing_avg += -sc*(fc - gc*vx*ec)/num_file
        cr_heating_avg += -vs*sc*(fc - gc*vx*ec)/num_file
        cr_workdone_avg += -vx*sc*(fc - gc*vx*ec)/num_file
        cr_couple_avg += (vx + vs)*np.gradient(ec*(gc - 1.), x_avg)/num_file
        invariant += (ec*(gc - 1.))*(vx + va)**(gc)/num_file
        total_general_heating_time[i] = np.sum((vx + vs)*sc*(fc - gc*vx*ec)*dx)
        total_couple_heating_time[i] = np.sum(-(vx + va)*np.gradient(ec*(gc - 1.), x_avg)*dx)
        dfc_time[i] = np.sum(np.gradient(fc, x_avg)*dx)
      if i == 0:
        rho0 = data['rho'][0, 0, index_xl:index_xu]
        v0 = data['vel1'][0, 0, index_xl:index_xu]
        if not self.isothermal:
          pg0 = data['press'][0, 0, index_xl:index_xu]
        if self.cr:
          pc0 = data['Ec'][0, 0, index_xl:index_xu]*(gc - 1.)
          fc0 = data['Fc1'][0, 0, index_xl:index_xu]*self.vmax
          va0 = np.abs(data['Vc1'][0, 0, index_xl:index_xu])

    # Save data
    self.time_array = time_array
    self.x_avg = x_avg 
    self.dx = dx
    self.rho_avg = rho_avg 
    self.v_avg = v_avg 
    self.rhov_avg = rhov_avg
    if not self.isothermal:
      self.pg_avg = pg_avg 
    if self.cr:
      self.pc_avg = pc_avg
      self.fc_avg = fc_avg 
      self.va_avg = va_avg
      self.sc_avg = sc_avg
      self.ev_avg = ev_avg
      self.steady_fc_avg = steady_fc_avg
      self.cr_pushing_avg = cr_pushing_avg
      self.cr_heating_avg = cr_heating_avg
      self.cr_workdone_avg = cr_workdone_avg
      self.cr_couple_avg = cr_couple_avg
      self.invariant = invariant
      self.total_couple_heating_time = total_couple_heating_time
      self.total_general_heating_time = total_general_heating_time
      self.dfc_time = dfc_time

    if compare:
      self.rho0 = rho0 
      self.v0 = v0 
      if not self.isothermal:
        self.pg0 = pg0 
      if self.cr: 
        self.pc0 = pc0 
        self.fc0 = fc0 
        self.va0 = va0

    self.kappa = kappa

    if plot:
      fig = plt.figure()
      grids = gs.GridSpec(2, 3, figure=fig) 
      ax1 = fig.add_subplot(grids[0, 0])
      ax2 = fig.add_subplot(grids[0, 1])
      ax3 = fig.add_subplot(grids[0, 2])
      if self.cr:
        ax4 = fig.add_subplot(grids[1, 0])
        ax5 = fig.add_subplot(grids[1, 1])
      if not self.isothermal:
        ax6 = fig.add_subplot(grids[1, 2])

      lab = ['$\\rho$', '$v$', '$\\rho v$', '$P_c$', '$F_c$', '$P_g$'] if not self.isothermal else ['$\\rho$', '$v$', '$\\rho v$', '$P_c$', '$F_c$']

      ax1.plot(x_avg, rho_avg)
      ax2.plot(x_avg, v_avg)
      ax3.plot(x_avg, rhov_avg)
      if self.cr:
        ax4.plot(x_avg, pc_avg)
        ax5.plot(x_avg, fc_avg)
      if not self.isothermal:
        ax6.plot(x_avg, pg_avg)

      if compare:
        ax1.plot(x_avg, rho0, 'k--')
        ax2.plot(x_avg, v0, 'k--')
        ax3.plot(x_avg, rho0*v0, 'k--')
        if self.cr:
          ax4.plot(x_avg, pc0, 'k--')
          ax5.plot(x_avg, fc0, 'k--')
        if not self.isothermal:
          ax6.plot(x_avg, pg0, 'k--')

      for i, axes in enumerate(fig.axes):
        if logx and (axes != ax2) and (axes != ax3) and (axes != ax5):
          axes.set_xscale('log')
        else:
          axes.xaxis.set_minor_locator(AutoMinorLocator())
        if logy and (axes != ax2) and (axes != ax3) and (axes != ax5):
          axes.set_yscale('log')
        else:
          axes.yaxis.set_minor_locator(AutoMinorLocator())
        axes.margins(x=0)
        axes.set_xlabel('$x$')
        axes.set_ylabel(lab[i])

      fig.tight_layout()
      return fig
    return
      
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



class Power_iso:
  def __init__(self, profile_input):
    alpha = profile_input['alpha']
    beta = profile_input['beta']
    eta = profile_input['eta']
    psi = profile_input['psi']
    rho0 = profile_input['rho0']
    cs0 = profile_input['cs0']
    r0 = profile_input['r0']
    self.alpha = alpha # pc_h/pg_h
    self.beta = beta # 2*pg_h/b^2
    self.eta = eta # kappa/gc L c_s
    self.psi = psi 
    self.phi = (gc/(2. - gc))*(1. - psi)
    self.rho0 = rho0 
    self.cs0 = cs0 
    self.r0 = r0 
    self.pg0 = cs0**2*rho0
    self.pc0 = alpha*self.pg0 
    self.b = np.sqrt(2.*self.pg0/beta)
    self.cc0 = np.sqrt(gc*self.pc0/rho0)
    self.va0 = self.b/np.sqrt(rho0)
    self.Lc0 = r0/self.phi
    self.kappa = eta*gc*self.Lc0*self.cs0 
    self.ldiff = self.kappa/self.cs0
  # End of init

  def pc(self, x):
    pc0 = self.pc0
    r0 = self.r0
    phi = self.phi
    cr_press = pc0*(x/r0)**(-phi)
    return cr_press

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
    cs0 = self.cs0
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
    pg_sol = cs0**2*rho_sol
    pc_sol = self.pc(x_eval)
    fc_sol = gc1*pc_sol*va_sol - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    dpgdx_sol = cs0**2*rho_sol*(va_sol*self.dpcdx(x_eval) - kappa*self.d2pcdx2(x_eval))/(gc*pc_sol*0.5*va_sol)
    g_sol = (dpgdx_sol + self.dpcdx(x_eval))/rho_sol

    # Save data
    self.dx = dx
    self.xmin = xmin
    self.xmax = xmax
    self.x_sol = x_eval
    self.va_sol = va_sol 
    self.rho_sol = rho_sol
    self.pg_sol = pg_sol
    self.dpgdx_sol = dpgdx_sol
    self.pc_sol = pc_sol
    self.dpcdx_sol = self.dpcdx(x_eval)
    self.fc_sol = fc_sol 
    self.g_sol = g_sol

    return

# End of Power_iso class



class Power_v:
  def __init__(self, profile_input):
    alpha = profile_input['alpha']
    beta = profile_input['beta']
    eta = profile_input['eta']
    ms = profile_input['ms']
    psi = profile_input['psi']
    rho0 = profile_input['rho0']
    pg0 = profile_input['pg0']
    r0 = profile_input['r0']
    self.alpha = alpha # pc_h/pg_h
    self.beta = beta # 2*pg_h/b^2
    self.eta = eta # kappa/gc L c_s
    self.ms = ms # v/cs sonic mach number, must be greater than 0
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

  def powerprofile(self, xmin, xmax, grids):
    kappa = self.kappa 
    rho0 = self.rho0
    b = self.b
    va0 = self.va0
    v0 = self.v0
    r0 = self.r0

    func = lambda x, vpva: (kappa*self.d2pcdx2(x) - vpva*self.dpcdx(x))/(gc*self.pc(x))

    dx = (xmax - xmin)/grids 
    x_eval = np.zeros(grids)
    x_eval[0] = xmin + 0.5*dx
    for i in np.arange(np.size(x_eval) - 1):
      x_eval[i+1] = x_eval[i] + dx 
    index = np.argmin(np.abs(x_eval - r0))
    index = index if (x_eval[index] > r0) else index + 1

    sol_front = integrate.solve_ivp(func, (r0, x_eval[-1]), [v0 + va0], t_eval=x_eval[index:])
    if index != 0:
      sol_back = integrate.solve_ivp(func, (r0, x_eval[0]), [v0 + va0], t_eval=x_eval[0:index][::-1])
      vpva_sol = np.append(sol_back.y[0][::-1], sol_front.y[0])
    else:
      vpva_sol = sol_front.y[0]

    # Find va from vpva_sol
    va_sol = np.zeros(np.size(vpva_sol)) 
    for i, vp_va in enumerate(vpva_sol):
      va_sol[i] = 0.5*(-va0/v0 + np.sqrt((va0/v0)**2 + 4.*vp_va/v0))*va0

    rho_sol = (b/va_sol)**2
    v_sol = vpva_sol - va_sol 
    dvdx_sol = np.gradient(v_sol, x_eval)
    pg_sol = self.pg(x_eval)
    pc_sol = self.pc(x_eval)
    fc_sol = gc1*pc_sol*(v_sol + va_sol) - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    g_sol = (rho0*v0*dvdx_sol + self.dpgdx(x_eval) + self.dpcdx(x_eval))/rho_sol
    H_sol = (v_sol*self.dpgdx(x_eval) + gg*pg_sol*dvdx_sol + (gg - 1.)*va_sol*self.dpcdx(x_eval))/(gg - 1.)

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
    self.dpcdx_sol = self.dpcdx(x_eval)
    self.fc_sol = fc_sol 
    self.g_sol = g_sol
    self.H_sol = H_sol

    return

# End of Power_v class



class Power_v_iso:
  def __init__(self, profile_input):
    alpha = profile_input['alpha']
    beta = profile_input['beta']
    eta = profile_input['eta']
    ms = profile_input['ms']
    psi = profile_input['psi']
    rho0 = profile_input['rho0']
    cs0 = profile_input['cs0']
    r0 = profile_input['r0']
    self.alpha = alpha # pc_h/pg_h
    self.beta = beta # 2*pg_h/b^2
    self.eta = eta # kappa/gc L c_s
    self.ms = ms # v/cs sonic mach number, must be greater than 0
    self.psi = psi 
    self.phi = (gc/(2. - gc))*(1. - psi)
    self.rho0 = rho0 
    self.cs0 = cs0 
    self.r0 = r0 
    self.pg0 = cs0**2*rho0
    self.pc0 = alpha*self.pg0 
    self.b = np.sqrt(2.*self.pg0/beta)
    self.cc0 = np.sqrt(gc*self.pc0/rho0)
    self.va0 = self.b/np.sqrt(rho0)
    self.v0 = ms*self.cs0
    self.Lc0 = r0/self.phi
    self.kappa = eta*gc*self.Lc0*self.cs0 
    self.ldiff = self.kappa/self.cs0

    if ms < 0.0:
      raise ValueError('Use profile.py instead.')
  # End of init

  def pc(self, x):
    pc0 = self.pc0
    r0 = self.r0
    phi = self.phi
    cr_press = pc0*(x/r0)**(-phi)
    return cr_press

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

  def powerprofile(self, xmin, xmax, grids):
    kappa = self.kappa 
    rho0 = self.rho0
    b = self.b
    cs0 = self.cs0
    va0 = self.va0
    v0 = self.v0
    r0 = self.r0

    func = lambda x, vpva: (kappa*self.d2pcdx2(x) - vpva*self.dpcdx(x))/(gc*self.pc(x))

    dx = (xmax - xmin)/grids 
    x_eval = np.zeros(grids)
    x_eval[0] = xmin + 0.5*dx
    for i in np.arange(np.size(x_eval) - 1):
      x_eval[i+1] = x_eval[i] + dx 
    index = np.argmin(np.abs(x_eval - r0))
    index = index if (x_eval[index] > r0) else index + 1

    sol_front = integrate.solve_ivp(func, (r0, x_eval[-1]), [v0 + va0], t_eval=x_eval[index:])
    if index != 0:
      sol_back = integrate.solve_ivp(func, (r0, x_eval[0]), [v0 + va0], t_eval=x_eval[0:index][::-1])
      vpva_sol = np.append(sol_back.y[0][::-1], sol_front.y[0])
    else:
      vpva_sol = sol_front.y[0]

    # Find va from vpva_sol
    va_sol = np.zeros(np.size(vpva_sol)) 
    for i, vp_va in enumerate(vpva_sol):
      va_sol[i] = 0.5*(-va0/v0 + np.sqrt((va0/v0)**2 + 4.*vp_va/v0))*va0

    rho_sol = (b/va_sol)**2
    v_sol = vpva_sol - va_sol 
    dvdx_sol = np.gradient(v_sol, x_eval)
    pg_sol = cs0**2*rho_sol
    pc_sol = self.pc(x_eval)
    fc_sol = gc1*pc_sol*(v_sol + va_sol) - (kappa/(gc - 1.))*self.dpcdx(x_eval)
    dpgdx_sol = cs0**2*rho_sol*(vpva_sol*self.dpcdx(x_eval) - kappa*self.d2pcdx2(x_eval))/(gc*pc_sol*(v_sol + 0.5*va_sol))
    g_sol = (rho0*v0*dvdx_sol + dpgdx_sol + self.dpcdx(x_eval))/rho_sol

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
    self.dpgdx_sol = dpgdx_sol
    self.pc_sol = pc_sol
    self.dpcdx_sol = self.dpcdx(x_eval)
    self.fc_sol = fc_sol 
    self.g_sol = g_sol

    return

# End of Power_v_iso class



class Power_v_cool:
  def __init__(self, profile_input):
    alpha = profile_input['alpha']
    beta = profile_input['beta']
    eta = profile_input['eta']
    ms = profile_input['ms']
    delta = profile_input['delta']
    epsilon = profile_input['epsilon']
    psi = profile_input['psi']
    rho0 = profile_input['rho0']
    cs0 = profile_input['cs0']
    r0 = profile_input['r0']
    self.alpha = alpha # pc_h/pg_h
    self.beta = beta # 2*pg_h/b^2
    self.eta = eta # kappa/gc L c_s
    self.ms = ms # v/cs sonic mach number, must be greater than 0
    self.delta = delta # t_cool/t_box
    self.epsilon = epsilon
    self.psi = psi 
    self.phi = (gc/(2. - gc))*(1. - psi)
    self.rho0 = rho0 
    self.pg0 = pg0 
    self.T0 = pg0/rho0
    self.r0 = r0 
    self.pc0 = alpha*pg0 
    self.b = np.sqrt(2.*pg0/beta)
    self.cs0 = np.sqrt(gg*pg0/rho0)
    self.cc0 = np.sqrt(gc*self.pc0/rho0)
    self.va0 = self.b/np.sqrt(rho0)
    self.v0 = ms*self.cs0
    self.Lc0 = r0/self.phi
    self.kappa = eta*gc*self.Lc0*self.cs0 
    self.ldiff = self.kappa/self.cs0
    self.lambda0 = self.phi*pg0*self.cs0/((gg - 1.)*rho0**2*r0*delta)

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

  def powerprofile(self, xmin, xmax, grids):
    kappa = self.kappa 
    rho0 = self.rho0
    pg0 = self.pg0
    b = self.b
    va0 = self.va0
    v0 = self.v0
    r0 = self.r0
    T0 = self.T0
    epsilon = self.epsilon
    lambda0 = self.lambda0 

    func = lambda x, vpva: (kappa*self.d2pcdx2(x) - vpva*self.dpcdx(x))/(gc*self.pc(x))

    dx = (xmax - xmin)/grids 
    x_eval = np.zeros(grids)
    x_eval[0] = xmin + 0.5*dx
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
    T_sol = pg_sol/rho_sol 
    cool_sol = lambda0*(T_sol/T0)**epsilon 
    H_sol = (v_sol*self.dpgdx(x_eval) + gg*pg_sol*dvdx_sol + (gg - 1.)*va_sol*self.dpcdx(x_eval) + (gg - 1.)*rho_sol**2*cool_sol)/((gg - 1.)*rho_sol)

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
    self.T_sol = T_sol
    self.H_sol = H_sol

    return

# End Power_v_cool class
####################################
inputfile = 'athinput.cr_power_v'
file_array = np.array([0, 1130]) 

# # Background for static 
# profile_in = dict()
# profile_in['alpha'] = 0.1
# profile_in['beta'] = 1.
# profile_in['eta'] = 0.01
# profile_in['psi'] = 0.

# profile_in['rho0'] = 1. 
# profile_in['pg0'] = 1. 
# profile_in['r0'] = 1.

# # Background for static iso
# profile_in = dict()
# profile_in['alpha'] = 1.
# profile_in['beta'] = 1.
# profile_in['eta'] = 0.01
# profile_in['psi'] = 0.5

# profile_in['rho0'] = 1. 
# profile_in['cs0'] = 1. 
# profile_in['r0'] = 1.

# # Background for static cool
# profile_in = dict()
# profile_in['alpha'] = 1.
# profile_in['beta'] = 1.
# profile_in['eta'] = 0.01
# profile_in['delta'] = 0.1
# profile_in['epsilon'] = -2.
# profile_in['psi'] = 0.

# profile_in['rho0'] = 1. 
# profile_in['pg0'] = 1. 
# profile_in['r0'] = 1.

# # Background for v
# profile_in = dict()
# profile_in['alpha'] = 1.
# profile_in['beta'] = 1.
# profile_in['eta'] = 0.01
# profile_in['ms'] = 0.015
# profile_in['psi'] = 0.

# profile_in['rho0'] = 1. 
# profile_in['pg0'] = 1. 
# profile_in['r0'] = 1.

# Background for v_iso
profile_in = dict()
profile_in['alpha'] = 1.
profile_in['beta'] = 1.
profile_in['eta'] = 0.01
profile_in['ms'] = 0.015
profile_in['psi'] = 0.

profile_in['rho0'] = 1. 
profile_in['cs0'] = 1. 
profile_in['r0'] = 1.

# # Background for v_cool
# profile_in = dict()
# profile_in['alpha'] = 1.
# profile_in['beta'] = 1.
# profile_in['eta'] = 0.01
# profile_in['ms'] = 0.05
# profile_in['delta'] = 1.
# profile_in['epsilon'] = -0.66667
# profile_in['psi'] = 0.

# profile_in['rho0'] = 1. 
# profile_in['pg0'] = 1. 
# profile_in['r0'] = 1.

one = Plot1d(inputfile, file_array, video=False, staircase=False, profile_in=None, with_v=False)
if one.passive:
  fig, fig_pass = one.plot(logx=False, logy=True)
  fig.savefig('./1dplot.png', dpi=300)
  fig_pass.savefig('./1dplot_passive.png', dpi=300)
else:
  fig = one.plot(logx=False, logy=True)
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
# # Make video
# # Equilibrium
# equi = {}
# equi['rho'] = 0.0
# equi['v'] = 0.0
# equi['pc'] = 0.0
# equi['pg'] = 0.0
# equi['fc'] = 0.0 

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# file_start = 0
# file_end = 250

# total_num_file = file_end - file_start 
# each = total_num_file//size

# comm.Barrier()

# video_array = np.arange(file_start + rank*each, file_start + (rank+1)*each)
# video_path = './output/video/'
# video = Plot1d(inputfile, video_array, video=True, staircase=False, profile_in=None, with_v=True)

# data0 = ar.athdf('./' + video.filename + '.out1.' + format(file_start, '05d') + '.athdf')
# initial = {}
# initial['rho'] = data0['rho'][0, 0, :]
# initial['v'] = data0['vel1'][0, 0, :]
# initial['pg'] = data0['press'][0, 0, :]
# initial['ec'] = data0['Ec'][0, 0, :]
# initial['fc'] = data0['Fc1'][0, 0, :]

# # video.make_video(equi, video_path, logx=False, logy=True, initial_data=initial, xlim=[1, 6.])
# video.make_video(equi, video_path, logx=False, logy=False, initial_data=None)

################################
# # Staricase identification
# stair_array = np.array([147])
# # stair_array = np.arange(0)
# stair = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=profile_in, with_v=True)
# # latexify(columns=1, square=True)
# stairfig_stat, stairfig_pc, stairfig_time, stairfig_avgpc, stairfig_avgv = stair.staircase(plot=True, xlim=[1., 6.], time_series=True, fit=False)
# # plotdefault()
# stairfig_stat.savefig('./staircase_stat.png', dpi=300)
# stairfig_pc.savefig('./staircase_pc.png', dpi=300)
# stairfig_time.savefig('./staircase_time.png', dpi=300) # Need to comment out if time_series=False
# stairfig_avgpc.savefig('./staircase_avgpc.png', dpi=300) # Need to comment out if time_series=False
# stairfig_avgv.savefig('./staircase_avgv.png', dpi=300) # Need to comment out if time_series=False

# plt.show()
# plt.close('all')

# # Parallel
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# stair_total_array = np.arange(100, 120)
# each = np.size(stair_total_array)//size

# stair_array = stair_total_array[rank*each:(rank+1)*each]
# stair = Plot1d(inputfile, stair_array, video=False, staircase=True, history=False, avg=False, profile_in=None, with_v=True)
# stair.staircase(plot=False, xlim=[1., 6.])

# size_of_arr = 0 
# size_of_shock_arr = 0
# size_of_arr_array = np.zeros(each, dtype=np.int)
# size_of_shock_arr_array = np.zeros(each, dtype=np.int)

# size_x1v = stair.size_x1v 
# total_merge_arr = np.zeros(size_x1v*each*size) 
# merge_count = size_x1v*each*np.ones(size, dtype=np.int)

# for i, key in enumerate(stair.width.keys()):
#   size_of_arr += np.size(stair.width[key])
#   size_of_arr_array[i] = np.size(stair.width[key]) 
#   size_of_shock_arr += np.size(stair.mach[key])
#   size_of_shock_arr_array[i] = np.size(stair.mach[key])

# size_of_each_proc = comm.allgather(size_of_arr)
# total_size_of_arr_array = np.zeros(np.size(stair_total_array), dtype=np.int)
# total_size_of_arr = comm.allreduce(size_of_arr, op=MPI.SUM)
# count = each*np.ones(size, dtype=np.int)
# comm.Gatherv(size_of_arr_array, [total_size_of_arr_array, count, MPI.LONG], root=0)

# size_of_each_shock_proc = comm.allgather(size_of_shock_arr)
# total_size_of_shock_arr_array = np.zeros(np.size(stair_total_array), dtype=np.int)
# total_size_of_shock_arr = comm.allreduce(size_of_shock_arr, op=MPI.SUM)
# comm.Gatherv(size_of_shock_arr_array, [total_size_of_shock_arr_array, count, MPI.LONG], root=0)

# now_time_array = stair.time_array
# now_couple_grid = stair.couple_grid.reshape(-1)
# now_width = stair.width
# now_jump_loc = stair.jump_loc
# now_height = stair.height 
# now_plateau = stair.plateau
# now_ldiffusive = stair.ldiffusive 
# now_cr_press = stair.cr_press 
# now_Lc = stair.Lc 
# now_coupled_mass_fraction = stair.coupled_mass_fraction 
# now_pc_ratio = stair.pc_ratio 
# now_mach = stair.mach
# now_vsh = stair.vsh 
# now_vsh_unitcs = stair.vsh_unitcs

# comm.Gatherv(now_couple_grid, [total_merge_arr, merge_count, MPI.DOUBLE], root=0)

# store_now_width = np.array([], dtype=np.double)
# store_now_jump_loc = np.array([], dtype=np.double)
# store_now_height = np.array([], dtype=np.double)
# store_now_plateau = np.array([], dtype=np.double)
# store_now_ldiffusive = np.array([], dtype=np.double)
# store_now_cr_press = np.array([], dtype=np.double)
# store_now_Lc = np.array([], dtype=np.double)
# store_now_coupled_mass_fraction = np.array([], dtype=np.double)
# store_now_pc_ratio = np.array([], dtype=np.double)
# store_now_mach = np.array([], dtype=np.double)
# store_now_vsh = np.array([], dtype=np.double)
# store_now_vsh_unitcs = np.array([], dtype=np.double)

# print(size_of_each_proc)

# for i, file in enumerate(now_width.keys()):
#   store_now_width = np.append(store_now_width, now_width[file])
#   store_now_jump_loc = np.append(store_now_jump_loc, now_jump_loc[file])
#   store_now_height = np.append(store_now_height, now_height[file])
#   store_now_plateau = np.append(store_now_plateau, now_plateau[file])
#   store_now_ldiffusive = np.append(store_now_ldiffusive, now_ldiffusive[file])
#   store_now_cr_press = np.append(store_now_cr_press, now_cr_press[file])
#   store_now_Lc = np.append(store_now_Lc, now_Lc[file])
#   store_now_coupled_mass_fraction = np.append(store_now_coupled_mass_fraction, now_coupled_mass_fraction[file])
#   store_now_pc_ratio = np.append(store_now_pc_ratio, now_pc_ratio[file])
#   store_now_mach = np.append(store_now_mach, now_mach[file])
#   store_now_vsh = np.append(store_now_vsh, now_vsh[file])
#   store_now_vsh_unitcs = np.append(store_now_vsh_unitcs, now_vsh_unitcs[file])

# time_array = np.zeros(np.size(stair_total_array), dtype=np.double)
# width = np.zeros(total_size_of_arr, dtype=np.double)
# jump_loc = np.zeros(total_size_of_arr, dtype=np.double)
# height = np.zeros(total_size_of_arr, dtype=np.double)
# plateau = np.zeros(total_size_of_arr, dtype=np.double)
# ldiffusive = np.zeros(total_size_of_arr, dtype=np.double)
# cr_press = np.zeros(total_size_of_arr, dtype=np.double)
# Lc = np.zeros(total_size_of_arr, dtype=np.double)
# coupled_mass_fraction = np.zeros(total_size_of_arr, dtype=np.double)
# pc_ratio = np.zeros(total_size_of_arr, dtype=np.double)
# mach = np.zeros(total_size_of_shock_arr, dtype=np.double)
# vsh = np.zeros(total_size_of_shock_arr, dtype=np.double)
# vsh_unitcs = np.zeros(total_size_of_shock_arr, dtype=np.double)

# comm.Gatherv(now_time_array, [time_array, count, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_width, [width, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_jump_loc, [jump_loc, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_height, [height, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_plateau, [plateau, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_ldiffusive, [ldiffusive, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_cr_press, [cr_press, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_Lc, [Lc, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_coupled_mass_fraction, [coupled_mass_fraction, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_pc_ratio, [pc_ratio, size_of_each_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_mach, [mach, size_of_each_shock_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_vsh, [vsh, size_of_each_shock_proc, MPI.DOUBLE], root=0)
# comm.Gatherv(store_now_vsh_unitcs, [vsh_unitcs, size_of_each_shock_proc, MPI.DOUBLE], root=0)

# if rank == 0:
#   num_bins = 100 
#   abso_width_bin = np.logspace(np.log10(np.amin(width)), np.log10(np.amax(width)), num_bins)
#   width_bin = np.logspace(np.log10(np.amin(width/ldiffusive)), np.log10(np.amax(width/ldiffusive)), num_bins)
#   height_bin = np.logspace(np.log10(np.amin(height/cr_press)), np.log10(np.amax(height/cr_press)), num_bins)
#   plateau = np.delete(plateau, np.where(plateau==0)[0])
#   plateau_bin = np.logspace(np.log10(np.amin(plateau)), np.log10(np.amax(plateau)), num_bins)
#   ldiff_bin = np.logspace(np.log10(np.amin(ldiffusive)), np.log10(np.amax(ldiffusive)), num_bins)
#   pc_ratio_bin = np.logspace(np.log10(np.amin(pc_ratio)), np.log10(np.amax(pc_ratio)), num_bins)
#   mach_bin = np.logspace(np.log10(np.amin(mach)), np.log10(np.amax(mach)), num_bins)
#   vsh_bin = np.logspace(np.log10(np.amin(np.abs(vsh))), np.log10(np.amax(np.abs(vsh))), num_bins)
  
#   abso_widths = (abso_width_bin[1:] - abso_width_bin[:-1])
#   abso_width_hist = np.histogram(width, bins=abso_width_bin)
#   abso_width_norm = abso_width_hist[0]/(abso_widths*np.sum(abso_width_hist[0]))

#   widths = (width_bin[1:] - width_bin[:-1])
#   width_hist = np.histogram(width/ldiffusive, bins=width_bin)
#   width_norm = width_hist[0]/(widths*np.sum(width_hist[0]))

#   heights = (height_bin[1:] - height_bin[:-1])
#   height_hist = np.histogram(height/cr_press, bins=height_bin)
#   height_norm = height_hist[0]/(heights*np.sum(height_hist[0]))

#   plateaus = (plateau_bin[1:] - plateau_bin[:-1])
#   plateau_hist = np.histogram(plateau, bins=plateau_bin)
#   plateau_norm = plateau_hist[0]/(plateaus*np.sum(plateau_hist[0]))

#   ldiffs = (ldiff_bin[1:] - ldiff_bin[:-1])
#   ldiff_hist = np.histogram(ldiffusive, bins=ldiff_bin)
#   ldiff_norm = ldiff_hist[0]/(ldiffs*np.sum(ldiff_hist[0]))

#   pc_ratios = (pc_ratio_bin[1:] - pc_ratio_bin[:-1])
#   pc_ratio_hist = np.histogram(pc_ratio, bins=pc_ratio_bin)
#   pc_ratio_norm = pc_ratio_hist[0]/(pc_ratios*np.sum(pc_ratio_hist[0]))

#   machs = (mach_bin[1:] - mach_bin[:-1])
#   mach_hist = np.histogram(mach, bins=mach_bin)
#   mach_norm = mach_hist[0]/(machs*np.sum(mach_hist[0]))

#   vshs = (vsh_bin[1:] - vsh_bin[:-1])
#   vsh_hist = np.histogram(np.abs(vsh), bins=vsh_bin)
#   vsh_norm = vsh_hist[0]/(vshs*np.sum(vsh_hist[0]))

#   fit_func = lambda r, c, alpha, beta, chi: c - alpha*r - (np.exp(r)/beta)**(chi)

#   abso_width_delete = np.where(abso_width_norm==0)[0]
#   abso_width_bin_fit = np.delete(abso_width_bin, abso_width_delete)
#   abso_width_norm_fit = np.delete(abso_width_norm, abso_width_delete)
#   abso_width_bin_fit_log = np.log(abso_width_bin_fit)
#   abso_width_norm_fit_log = np.log(abso_width_norm_fit)
#   abso_widths_fit = np.exp(abso_width_bin_fit_log[1:]) - np.exp(abso_width_bin_fit_log[:-1])

#   width_delete = np.where(width_norm==0)[0]
#   width_bin_fit = np.delete(width_bin, width_delete)
#   width_norm_fit = np.delete(width_norm, width_delete)
#   width_bin_fit_log = np.log(width_bin_fit)
#   width_norm_fit_log = np.log(width_norm_fit)
#   widths_fit = np.exp(width_bin_fit_log[1:]) - np.exp(width_bin_fit_log[:-1])

#   height_delete = np.where(height_norm==0)[0]
#   height_bin_fit = np.delete(height_bin, height_delete)
#   height_norm_fit = np.delete(height_norm, height_delete)
#   height_bin_fit_log = np.log(height_bin_fit)
#   height_norm_fit_log = np.log(height_norm_fit)
#   heights_fit = np.exp(height_bin_fit_log[1:]) - np.exp(height_bin_fit_log[:-1])

#   plateau_delete = np.where(plateau_norm==0)[0]
#   plateau_bin_fit = np.delete(plateau_bin, plateau_delete)
#   plateau_norm_fit = np.delete(plateau_norm, plateau_delete)
#   plateau_bin_fit_log = np.log(plateau_bin_fit)
#   plateau_norm_fit_log = np.log(plateau_norm_fit)
#   plateaus_fit = np.exp(plateau_bin_fit_log[1:]) - np.exp(plateau_bin_fit_log[:-1])

#   ldiff_delete = np.where(ldiff_norm==0)[0]
#   ldiff_bin_fit = np.delete(ldiff_bin, ldiff_delete)
#   ldiff_norm_fit = np.delete(ldiff_norm, ldiff_delete)
#   ldiff_bin_fit_log = np.log(ldiff_bin_fit)
#   ldiff_norm_fit_log = np.log(ldiff_norm_fit)
#   ldiffs_fit = np.exp(ldiff_bin_fit_log[1:]) - np.exp(ldiff_bin_fit_log[:-1])

#   pc_ratio_delete = np.where(pc_ratio_norm==0)[0]
#   pc_ratio_bin_fit = np.delete(pc_ratio_bin, pc_ratio_delete)
#   pc_ratio_norm_fit = np.delete(pc_ratio_norm, pc_ratio_delete)
#   pc_ratio_bin_fit_log = np.log(pc_ratio_bin_fit)
#   pc_ratio_norm_fit_log = np.log(pc_ratio_norm_fit)
#   pc_ratios_fit = np.exp(pc_ratio_bin_fit_log[1:]) - np.exp(pc_ratio_bin_fit_log[:-1])

#   mach_delete = np.where(mach_norm==0)[0]
#   mach_bin_fit = np.delete(mach_bin, mach_delete)
#   mach_norm_fit = np.delete(mach_norm, mach_delete)
#   mach_bin_fit_log = np.log(mach_bin_fit)
#   mach_norm_fit_log = np.log(mach_norm_fit)
#   machs_fit = np.exp(mach_bin_fit_log[1:]) - np.exp(mach_bin_fit_log[:-1])

#   vsh_delete = np.where(vsh_norm==0)[0]
#   vsh_bin_fit = np.delete(vsh_bin, vsh_delete)
#   vsh_norm_fit = np.delete(vsh_norm, vsh_delete)
#   vsh_bin_fit_log = np.log(vsh_bin_fit)
#   vsh_norm_fit_log = np.log(vsh_norm_fit)
#   vshs_fit = np.exp(vsh_bin_fit_log[1:]) - np.exp(vsh_bin_fit_log[:-1])

#   # fit_func = lambda r, c, alpha, beta, chi: c - alpha*r - (np.exp(r)/beta)**(chi)
#   fit_func = lambda r, c, alpha, beta: c - alpha*r - (np.exp(r)/beta)

#   # try:
#   #   abso_width_opt, abso_width_cov = opt.curve_fit(fit_func, abso_width_bin_fit_log[:-1], abso_width_norm_fit_log, \
#   #     p0=(np.amax(abso_width_norm_fit_log), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
#   #   abso_width_fit_success = True 
#   #   abso_width_amp, abso_width_alpha, abso_width_beta, abso_width_chi = abso_width_opt 
#   # except:
#   #   print('Cannot fit abso width')
#   #   abso_width_fit_success = False
#   #   abso_width_alpha, abso_width_beta, abso_width_chi = None, None, None 

#   # try:
#   #   width_opt, width_cov = opt.curve_fit(fit_func, width_bin_fit_log[:-1], width_norm_fit_log, \
#   #     p0=(np.amax(width_norm_fit_log), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
#   #   width_fit_success = True 
#   #   width_amp, width_alpha, width_beta, width_chi = width_opt 
#   # except:
#   #   print('Cannot fit width')
#   #   width_fit_success = False
#   #   width_alpha, width_beta, width_chi = None, None, None 

#   try:
#     height_index = np.argmax(height_norm_fit_log)
#     # height_opt, height_cov = opt.curve_fit(fit_func, height_bin_fit_log[height_index:-1], height_norm_fit_log[height_index:], \
#     #   p0=(np.amax(height_norm_fit_log), 1., 1., 1.), bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
#     height_opt, height_cov = opt.curve_fit(fit_func, height_bin_fit_log[height_index:-1], height_norm_fit_log[height_index:], \
#       p0=(np.amax(height_norm_fit_log), 1., 1.), bounds=(0, [np.inf, np.inf, np.inf]))
#     height_fit_success = True
#     # height_amp, height_alpha, height_beta, height_chi = height_opt
#     height_amp, height_alpha, height_beta = height_opt
#   except:
#     print('Cannot fit height')
#     height_fit_success = False
#     # height_alpha, height_beta, height_chi = None, None, None
#     height_alpha, height_beta = None, None, None
#   latexify(columns=1, square=True)

#   fig1 = plt.figure()
#   fig2 = plt.figure() 
#   fig3 = plt.figure()
#   fig4 = plt.figure()
#   fig5 = plt.figure()
#   fig6 = plt.figure()
#   fig7 = plt.figure()
#   fig8 = plt.figure()
#   fig9 = plt.figure()
#   fig10 = plt.figure()
#   fig11 = plt.figure()
#   fig12 = plt.figure()
#   fig13 = plt.figure()
#   ax1 = fig1.add_subplot(111)
#   ax2 = fig2.add_subplot(111)
#   ax3 = fig3.add_subplot(111)
#   ax4 = fig4.add_subplot(111)
#   ax5 = fig5.add_subplot(111)
#   ax6 = fig6.add_subplot(111)
#   ax7 = fig7.add_subplot(111)
#   ax8 = fig8.add_subplot(111)
#   ax9 = fig9.add_subplot(111)
#   ax10 = fig10.add_subplot(111)
#   ax11 = fig11.add_subplot(111)
#   ax12 = fig12.add_subplot(111)
#   ax13 = fig13.add_subplot(111)

#   ax1.bar(np.exp(width_bin_fit_log[:-2]), np.exp(width_norm_fit_log[:-1]), widths_fit[:-1], align='edge')
#   ax3.bar(np.exp(ldiff_bin_fit_log[:-2]), np.exp(ldiff_norm_fit_log[:-1]), ldiffs_fit[:-1], align='edge')
#   ax4.bar(np.exp(abso_width_bin_fit_log[:-2]), np.exp(abso_width_norm_fit_log[:-1]), abso_widths_fit[:-1], align='edge')
#   ax6.bar(np.exp(pc_ratio_bin_fit_log[:-2]), np.exp(pc_ratio_norm_fit_log[:-1]), pc_ratios_fit[:-1], align='edge')
#   ax7.bar(np.exp(mach_bin_fit_log[:-2]), np.exp(mach_norm_fit_log[:-1]), machs_fit[:-1], align='edge')
#   ax8.bar(np.exp(vsh_bin_fit_log[:-2]), np.exp(vsh_norm_fit_log[:-1]), vshs_fit[:-1], align='edge')
#   ax10.bar(np.exp(height_bin_fit_log[:-2]), np.exp(height_norm_fit_log[:-1]), heights_fit[:-1], align='edge')
#   ax12.bar(np.exp(plateau_bin_fit_log[:-2]), np.exp(plateau_norm_fit_log[:-1]), plateaus_fit[:-1], align='edge')

#   ax2.scatter(width, height)
#   ax11.scatter(width/ldiffusive, height/cr_press)
#   wh_fit = np.polyfit(np.log(width/ldiffusive), np.log(height/cr_press), deg=1)
#   plot_fit_widthx = np.logspace(-2, 2, 100)
#   ax11.plot(plot_fit_widthx, np.exp(wh_fit[1])*(plot_fit_widthx)**(wh_fit[0]), 'k--', label='coeff={:.4f}, index={:.2f}'.format(np.exp(wh_fit[1]), wh_fit[0]))

#   print('The h-w coefficient = {:f}'.format(np.exp(wh_fit[1])))

#   ax5.scatter(time_array, total_size_of_arr_array)
#   num_stair_fit = np.polyfit(time_array, total_size_of_arr_array, 0)
#   ax5.plot(time_array, num_stair_fit[0]*np.ones(np.size(time_array)), 'k--', label='N = {:d}'.format(int(num_stair_fit[0])))

#   ax9.scatter(mach, np.abs(vsh_unitcs))

#   if height_fit_success:
#     # ax10.loglog(np.exp(height_bin_fit_log[height_index:-1]), \
#     #   np.exp(fit_func(height_bin_fit_log[height_index:-1], height_amp, height_alpha, height_beta, height_chi)), 'k--', \
#     #   label='$\\nu={:.4f},h_*={:.4f},\\chi={:.2f}$'.format(height_alpha, height_beta, height_chi))
#     ax10.loglog(np.exp(height_bin_fit_log[height_index:-1]), \
#       np.exp(fit_func(height_bin_fit_log[height_index:-1], height_amp, height_alpha, height_beta)), 'k--', \
#       label='$\\nu={:.4f},h_*={:.4f}$'.format(height_alpha, height_beta))
#     ax10.legend(frameon=False)

#   # Staircase merge plot
#   merge = total_merge_arr.reshape(each*size, size_x1v)
#   ax13.imshow(merge.T, cmap='gray', aspect='auto', interpolation='nearest', origin='lower', extent=[time_array[0], time_array[-1], stair.x1v[0], stair.x1v[-1]])

#   ax1.set_xscale('log')
#   ax1.set_yscale('log')
#   ax1.set_xlabel('Width ($l_\\mathrm{diff}$)')
#   ax1.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax1.xaxis.set_minor_locator(x_minor)

#   ax2.set_xscale('log')
#   ax2.set_yscale('log')
#   ax2.set_xlabel('Width')
#   ax2.set_ylabel('$P_c$ jump height')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax2.xaxis.set_minor_locator(x_minor)

#   ax3.set_xscale('log')
#   ax3.set_yscale('log')
#   ax3.set_xlabel('$l_\\mathrm{diff}$')
#   ax3.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax3.xaxis.set_minor_locator(x_minor)

#   ax4.set_xscale('log')
#   ax4.set_yscale('log')
#   ax4.set_xlabel('Absolute Width (Code units)')
#   ax4.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax4.xaxis.set_minor_locator(x_minor)

#   ax5.legend(frameon=0)
#   ax5.set_xlabel('$t$')
#   ax5.set_ylabel('Number of stairs')
#   ax5.xaxis.set_minor_locator(AutoMinorLocator())
#   ax5.yaxis.set_minor_locator(AutoMinorLocator())

#   ax6.set_xscale('log')
#   ax6.set_yscale('log')
#   ax6.set_xlabel('$P_{c,\\mathrm{min}}/P_{c,\\mathrm{max}}$')
#   ax6.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax6.xaxis.set_minor_locator(x_minor)

#   ax7.set_xscale('log')
#   ax7.set_yscale('log')
#   ax7.set_xlabel('$\\mathcal{M}$')
#   ax7.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax7.xaxis.set_minor_locator(x_minor)

#   ax8.set_xscale('log')
#   ax8.set_yscale('log')
#   ax8.set_xlabel('$v_\\mathrm{sh}$')
#   ax8.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax8.xaxis.set_minor_locator(x_minor)

#   ax9.set_xscale('log')
#   ax9.set_yscale('log')
#   ax9.set_xlabel('$\\mathcal{M}$')
#   ax9.set_ylabel('$v_\\mathrm{sh}/c_s$')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax9.xaxis.set_minor_locator(x_minor)

#   ax10.set_xscale('log')
#   ax10.set_yscale('log')
#   ax10.set_xlabel('Height ($P_c$)')
#   ax10.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax10.xaxis.set_minor_locator(x_minor)

#   ax11.legend(frameon=False)
#   ax11.set_xscale('log')
#   ax11.set_yscale('log')
#   ax11.set_xlabel('Width $w$')
#   ax11.set_ylabel('Jump height $h$')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax11.xaxis.set_minor_locator(x_minor)

#   ax12.set_xscale('log')
#   ax12.set_yscale('log')
#   ax12.set_xlabel('Plateau width H ($l_\\mathrm{diff}$)')
#   ax12.set_ylabel('Distribution')
#   x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
#   ax12.xaxis.set_minor_locator(x_minor)

#   ax13.margins(x=0)
#   ax13.set_xlabel('$t$')
#   ax13.set_ylabel('$x$')
#   ax13.xaxis.set_minor_locator(AutoMinorLocator())

#   fig1.tight_layout()
#   fig2.tight_layout()
#   fig3.tight_layout()
#   fig4.tight_layout()
#   fig5.tight_layout()
#   fig6.tight_layout()
#   fig7.tight_layout()
#   fig8.tight_layout()
#   fig9.tight_layout()
#   fig10.tight_layout()
#   fig11.tight_layout()
#   fig12.tight_layout()
#   fig13.tight_layout()
#   fig1.savefig('./stair_width.png', dpi=300)
#   fig2.savefig('./width_height.png', dpi=300)
#   fig3.savefig('./stair_ldiff.png', dpi=300)
#   fig4.savefig('./stair_abso_width.png', dpi=300)
#   fig5.savefig('./stair_number.png', dpi=300)
#   fig6.savefig('./stair_pc_ratio.png', dpi=300)
#   fig7.savefig('./stair_mach.png', dpi=300)
#   fig8.savefig('./stair_vsh.png', dpi=300)
#   fig9.savefig('./mach_vsh.png', dpi=300)
#   fig10.savefig('./stair_height.png', dpi=300)
#   fig11.savefig('./scale_width_height.png', dpi=300)
#   fig12.savefig('./stair_plateau.png', dpi=300)
#   fig13.savefig('./merge.png', dpi=300)

#   print('The mean Mach number is: {:f}'.format(median(np.abs(mach))))

#   plt.show(fig13)

#   plt.close('all')
#   plotdefault()

###############################
# # Construct convex hull for density and reconstruct Pc
# stair_array = np.array([1000])
# stair2 = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=profile_in, with_v=False)
# latexify(columns=1, square=True)
# stair2fig_rho, stair2fig_pc = stair2.convexhull(xlim=False, plot=True)
# stair2fig_pc = stair2.convexhull2(plot=True, logx=False, logy=True)

# stair2fig_rho.savefig('./rho_hull.png', dpi=300)
# stair2fig_pc.savefig('./pc_hull.png', dpi=300)
# plotdefault()

# plt.show()
# plt.close('all')

# # Parallel 
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# file_start = 0
# file_end = 8550

# total_num_file = file_end - file_start 
# each = total_num_file//size

# comm.Barrier()

# stair_array = np.arange(file_start + rank*each, file_start + (rank+1)*each)
# video_path = './convex/'
# stair2 = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=None, with_v=True)
# stair2.convexhull(xlim=False, plot=False, video=True, save_path=video_path, average=True)

# # # Deltas
# # comm.Barrier()

# # total_time_array = np.zeros(total_num_file)
# # total_delta_pc_real = np.zeros(total_num_file)
# # total_delta_pc_predict = np.zeros(total_num_file)

# # comm.Gatherv(stair2.time_array, [total_time_array, each*np.ones(size, dtype=np.int), MPI.DOUBLE], root=0)
# # comm.Gatherv(stair2.delta_pc_real, [total_delta_pc_real, each*np.ones(size, dtype=np.int), MPI.DOUBLE], root=0)
# # comm.Gatherv(stair2.delta_pc_predict, [total_delta_pc_predict, each*np.ones(size, dtype=np.int), MPI.DOUBLE], root=0)

# # if rank == 0:
# #   plotdefault()
# #   fig = plt.figure()
# #   ax = fig.add_subplot(111)

# #   ax.plot(total_time_array, total_delta_pc_real)
# #   ax.plot(total_time_array, total_delta_pc_predict, 'k--')

# #   ax.margins(x=0)
# #   ax.set_xlabel('$t$')
# #   ax.set_ylabel('$\\Delta P_c$')

# #   ax.xaxis.set_minor_locator(AutoMinorLocator())
# #   ax.yaxis.set_minor_locator(AutoMinorLocator())

# #   fig.tight_layout()
# #   fig.savefig('./delta_pc.png', dpi=300)

# #   plt.show()
# #   plt.close('all')

# # Convex average
# comm.Barrier()

# stair_total_array = np.arange(0, total_num_file)

# avg_pc = np.zeros(np.size(stair2.x))
# avg_predict_pc = np.zeros(np.size(stair2.x))

# comm.Reduce(stair2.pc_avg*np.size(stair_array)/np.size(stair_total_array), avg_pc, op=MPI.SUM, root=0)
# comm.Reduce(stair2.pc_predict_avg*np.size(stair_array)/np.size(stair_total_array), avg_predict_pc, op=MPI.SUM, root=0)

# if rank == 0:
#   plotdefault()

#   logx = False
#   logy = True 

#   fig = plt.figure()
#   ax1 = fig.add_subplot(111)

#   ax1.plot(stair2.x, avg_pc)
#   ax1.plot(stair2.x, avg_predict_pc, 'k--')

#   ax1.margins(x=0)
#   ax1.set_xlabel('$x$')
#   ax1.set_ylabel('$P_c$')
#   if logx:
#     ax1.set_xscale('log')
#   else:
#     ax1.xaxis.set_minor_locator(AutoMinorLocator())
#   if logy:
#     ax1.set_yscale('log')
#   else:
#     ax2.yaxis.set_minor_locator(AutoMinorLocator())

#   fig.tight_layout()
#   fig.savefig('./average_with_predicted.png', dpi=300)

###############################
# # Staircase fitting
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
# # History plot
# hist_array = np.arange(1001)
# hist = Plot1d(inputfile, hist_array, video=False, staircase=False, history=True, profile_in=profile_in, with_v=True)
# latexify(columns=2)
# histfigt, histfigx = hist.history(plot=True)
# # histfigt, histfigx = hist.history2(start=1.1, plot=True)
# histfigt.savefig('./hist_time.png', dpi=300)
# histfigx.savefig('./hist_x.png', dpi=300)
# plotdefault()

# plt.show()
# plt.close('all')

######################################
# # Time avg plot
# avg_array = np.arange(0, 20000)
# # avg_array = np.array([700])
# latexify(columns=2)

# avg = Plot1d(inputfile, avg_array, video=False, staircase=False, history=False, avg=True, profile_in=None, with_v=True)
# # avg.time_avg(plot=False)
# avg_fig = avg.time_avg(plot=True, logx=False, logy=True, compare=True)
# avg_fig.savefig('./time_avg.png', dpi=300)

# plt.show()
# plt.close()

# plotdefault()

# # Parallel
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# avg_total_array = np.arange(0, 20)
# each = np.size(avg_total_array)//size

# latexify(columns=2)

# avg_array = avg_total_array[rank*each:(rank+1)*each]
# avg = Plot1d(inputfile, avg_array, video=False, staircase=False, history=False, avg=True, profile_in=None, with_v=True)
# avg.time_avg(plot=False, compare=True, xlim=[1., 6.])

# # Combine values
# avg_time = np.zeros(np.size(avg_total_array))
# avg_rho = np.zeros(np.size(avg.x_avg))
# avg_v = np.zeros(np.size(avg.x_avg))
# avg_rhov = np.zeros(np.size(avg.x_avg))
# if not avg.isothermal:
#   avg_pg = np.zeros(np.size(avg.x_avg))
# if avg.cr:
#   avg_pc = np.zeros(np.size(avg.x_avg))
#   avg_fc = np.zeros(np.size(avg.x_avg))
#   avg_va = np.zeros(np.size(avg.x_avg))
#   avg_sc = np.zeros(np.size(avg.x_avg))
#   avg_ev = np.zeros(np.size(avg.x_avg))
#   avg_steady_fc = np.zeros(np.size(avg.x_avg))
#   avg_cr_pushing = np.zeros(np.size(avg.x_avg))
#   avg_cr_heating = np.zeros(np.size(avg.x_avg))
#   avg_cr_workdone = np.zeros(np.size(avg.x_avg))
#   avg_cr_couple = np.zeros(np.size(avg.x_avg))
#   invari = np.zeros(np.size(avg.x_avg))
#   total_general_heating = np.zeros(np.size(avg_total_array))
#   dfc = np.zeros(np.size(avg_total_array))

# comm.Gatherv(avg.time_array, [avg_time, each*np.ones(size, dtype=np.int), MPI.DOUBLE], root=0)
# comm.Reduce(avg.rho_avg*np.size(avg_array)/np.size(avg_total_array), avg_rho, op=MPI.SUM, root=0)
# comm.Reduce(avg.v_avg*np.size(avg_array)/np.size(avg_total_array), avg_v, op=MPI.SUM, root=0)
# comm.Reduce(avg.rhov_avg*np.size(avg_array)/np.size(avg_total_array), avg_rhov, op=MPI.SUM, root=0)
# if not avg.isothermal:
#   comm.Reduce(avg.pg_avg*np.size(avg_array)/np.size(avg_total_array), avg_pg, op=MPI.SUM, root=0)
# if avg.cr:
#   comm.Reduce(avg.pc_avg*np.size(avg_array)/np.size(avg_total_array), avg_pc, op=MPI.SUM, root=0)
#   comm.Reduce(avg.fc_avg*np.size(avg_array)/np.size(avg_total_array), avg_fc, op=MPI.SUM, root=0)
#   comm.Reduce(avg.va_avg*np.size(avg_array)/np.size(avg_total_array), avg_va, op=MPI.SUM, root=0)
#   comm.Reduce(avg.sc_avg*np.size(avg_array)/np.size(avg_total_array), avg_sc, op=MPI.SUM, root=0)
#   comm.Reduce(avg.ev_avg*np.size(avg_array)/np.size(avg_total_array), avg_ev, op=MPI.SUM, root=0)
#   comm.Reduce(avg.steady_fc_avg*np.size(avg_array)/np.size(avg_total_array), avg_steady_fc, op=MPI.SUM, root=0)
#   comm.Reduce(avg.cr_pushing_avg*np.size(avg_array)/np.size(avg_total_array), avg_cr_pushing, op=MPI.SUM, root=0)
#   comm.Reduce(avg.cr_heating_avg*np.size(avg_array)/np.size(avg_total_array), avg_cr_heating, op=MPI.SUM, root=0)
#   comm.Reduce(avg.cr_workdone_avg*np.size(avg_array)/np.size(avg_total_array), avg_cr_workdone, op=MPI.SUM, root=0)
#   comm.Reduce(avg.cr_couple_avg*np.size(avg_array)/np.size(avg_total_array), avg_cr_couple, op=MPI.SUM, root=0)
#   comm.Reduce(avg.invariant*np.size(avg_array)/np.size(avg_total_array), invari, op=MPI.SUM, root=0)
#   comm.Gatherv(avg.total_general_heating_time, [total_general_heating, each*np.ones(size, dtype=np.int), MPI.DOUBLE], root=0)
#   comm.Gatherv(avg.dfc_time, [dfc, each*np.ones(size, dtype=np.int), MPI.DOUBLE], root=0)

# # # video_path = '/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/wind/results/sims/output/video/'
# # # avg.fc_compare(video_path, logx=False, logy=True)

# if rank == 0:
#   avg.time_array = avg_time
#   avg.rho_avg = avg_rho 
#   avg.v_avg = avg_v
#   avg.rhov_avg = avg_rhov
#   if not avg.isothermal:
#     avg.pg_avg = avg_pg 
#   if avg.cr:
#     avg.pc_avg = avg_pc 
#     avg.fc_avg = avg_fc 
#     avg.va_avg = avg_va 
#     avg.sc_avg = avg_sc 
#     avg.ev_avg = avg_ev 
#     avg.steady_fc_avg = avg_steady_fc 
#     avg.cr_pushing_avg = avg_cr_pushing
#     avg.cr_heating_avg = avg_cr_heating
#     avg.cr_workdoen_avg = avg_cr_workdone
#     avg.cr_couple_avg = avg_cr_couple
#     avg.invariant = invari
#     avg.total_general_heating_time = total_general_heating
#     avg.dfc_time = dfc

#   # Plot average plots
#   logx = False 
#   logy = True

#   fig = plt.figure()
#   grids = gs.GridSpec(2, 3, figure=fig) 
#   ax1 = fig.add_subplot(grids[0, 0])
#   ax2 = fig.add_subplot(grids[0, 1])
#   ax3 = fig.add_subplot(grids[0, 2])
#   if avg.cr:
#     ax4 = fig.add_subplot(grids[1, 0])
#     ax5 = fig.add_subplot(grids[1, 1])
#   if not avg.isothermal:
#     ax6 = fig.add_subplot(grids[1, 2])

#   lab = ['$\\rho$', '$v$', '$\\rho v$', '$P_c$', '$F_c$', '$P_g$'] if not avg.isothermal else ['$\\rho$', '$v$', '$\\rho v$', '$P_c$', '$F_c$']

#   # Calculate ratio of Mdot
#   mdot = np.mean(avg.rhov_avg)
#   mdot0 = np.mean(avg.rho0*avg.v0)
#   mdot_ratio = mdot/mdot0

#   # Calculate ratio of Delta Pc and Fc 
#   Dpc = avg.pc_avg[0] - avg.pc_avg[-1]
#   Dpc0 = avg.pc0[0] - avg.pc0[-1]
#   Dpc_ratio = Dpc/Dpc0
#   Dfc = avg.fc_avg[0] - avg.fc_avg[-1] 
#   Dfc0 = avg.fc0[0] - avg.fc0[-1] 
#   Dfc_ratio = Dfc/Dfc0

#   ax1.plot(avg.x_avg, avg.rho_avg)
#   ax2.plot(avg.x_avg, avg.v_avg)
#   ax3.plot(avg.x_avg, avg.rhov_avg)
#   ax3.axhline(mdot_ratio, linestyle=':', color='r', label='$\\dot{{M}}/\\dot{{M}}_0={:.3f}$'.format(mdot_ratio))
#   if avg.cr:
#     ax4.plot(avg.x_avg, avg.pc_avg, label='$\\Delta P_c/\\Delta P_{{c0}}={:.3f}$'.format(Dpc_ratio))
#     ax5.plot(avg.x_avg, avg.fc_avg, label='$\\Delta F_c/\\Delta F_{{c0}}={:.3f}$'.format(Dfc_ratio))
#   if not avg.isothermal:
#     ax6.plot(avg.x_avg, avg.pg_avg)

#   ax1.plot(avg.x_avg, avg.rho0, 'k--')
#   ax2.plot(avg.x_avg, avg.v0, 'k--')
#   ax3.plot(avg.x_avg, avg.rho0*avg.v0, 'k--')
#   if avg.cr:
#     ax4.plot(avg.x_avg, avg.pc0, 'k--')
#     ax5.plot(avg.x_avg, avg.fc0, 'k--')
#   if not avg.isothermal:
#     ax6.plot(avg.x_avg, avg.pg0, 'k--')

#   for i, axes in enumerate(fig.axes):
#     if logx and (axes != ax2) and (axes != ax3) and (axes != ax5):
#       axes.set_xscale('log')
#     else:
#       axes.xaxis.set_minor_locator(AutoMinorLocator())
#     if logy and (axes != ax2) and (axes != ax3) and (axes != ax5):
#       axes.set_yscale('log')
#     else:
#       axes.yaxis.set_minor_locator(AutoMinorLocator())
#     axes.legend(frameon=False)
#     axes.margins(x=0)
#     axes.set_xlabel('$x$')
#     axes.set_ylabel(lab[i])

#   fig.tight_layout()
#   fig.savefig('./time_averaged.png', dpi=300)

#   # Return back to source terms
#   latexify(columns=1, square=False)

#   # x_vpva = (avg.v_avg + avg.va_avg)**(-gc)
#   # smooth_vpva = signal.savgol_filter(x_vpva, 31, 3)
#   # smooth_pc = signal.savgol_filter(avg.pc_avg, 31, 3)
#   # # dlogpcdcoup = np.gradient(np.log(avg.pc_avg), np.log(x_vpva))
#   # # dlogpcdcoup_delete = np.where(dlogpcdcoup < 0)[0]
#   # dlogpcdcoup = np.gradient(np.log(smooth_pc), np.log(smooth_vpva))
#   # dlogpcdcoup_delete = np.where(dlogpcdcoup < 0)[0]
#   # smooth_vpva = np.delete(smooth_vpva, dlogpcdcoup_delete)
#   # dlogpcdcoup = np.delete(dlogpcdcoup, dlogpcdcoup_delete)
#   # # dlogpcdcoup_smooth = signal.savgol_filter(dlogpcdcoup, 31, 3)
#   # fit = np.polyfit(np.log(smooth_vpva), dlogpcdcoup, deg=0)

#   # fig = plt.figure()
#   # ax = fig.add_subplot(111)

#   # ax.loglog(smooth_vpva, dlogpcdcoup)
#   # # ax.loglog(x_rho, dlogpcdcoup)
#   # # ax.loglog(x_vpva, dlogpcdcoup_smooth, 'r')
#   # ax.axhline(1.0, color='k', linestyle='--')
#   # ax.axhline(fit[0], color='brown', linestyle=':')

#   # ax.margins(x=0)
#   # ax.set_xlabel('$(v + v_A)^{-\\gamma_c}$')
#   # # ax.set_xlabel('$\\rho^{\\gamma_c/2}$')
#   # ax.set_ylabel('$\\mathrm{dlog} P_c/\\mathrm{dlog} (v + v_A)^{-\\gamma_c}$')
#   # # ax.set_ylabel('$\\mathrm{dlog} P_c/\\mathrm{dlog} \\rho^{\\gamma_c/2}$')

#   # fig.tight_layout()
#   # plt.show()

#   # Pc invariant
#   # fit_invar = np.polyfit(avg.x_avg, avg.invariant, deg=1)
#   # fit_invar2 = np.polyfit(avg.x_avg, avg.pc_avg*(avg.v_avg + avg.va_avg)**(gc), deg=1)
#   # fit_push = np.polyfit(np.log(-np.gradient(avg.pc_avg, avg.x_avg)), np.log(-avg.cr_pushing_avg), deg=1)
#   # fit_heat = np.polyfit(np.log(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg)), np.log(-avg.cr_heating_avg), deg=1)

#   fig1 = plt.figure()
#   fig2 = plt.figure()
#   fig3 = plt.figure()
#   fig4 = plt.figure()
#   fig5 = plt.figure()
#   fig6 = plt.figure()
#   fig7 = plt.figure()
#   fig8 = plt.figure()
#   fig9 = plt.figure()
#   fig10 = plt.figure()
#   fig11 = plt.figure()
#   fig12 = plt.figure()
#   ax1 = fig1.add_subplot(111)
#   ax2 = fig2.add_subplot(111)
#   ax3 = fig3.add_subplot(111)
#   ax4 = fig4.add_subplot(111)
#   ax5 = fig5.add_subplot(111)
#   ax6 = fig6.add_subplot(111)
#   ax7 = fig7.add_subplot(111)
#   ax8 = fig8.add_subplot(111)
#   ax9 = fig9.add_subplot(111)
#   ax10 = fig10.add_subplot(111)
#   ax11 = fig11.add_subplot(111)
#   ax12 = fig12.add_subplot(111)

#   ax1.plot(avg.x_avg, avg.invariant, 'b', label='$\\langle P_c (v + v_A)^{\\gamma_c}\\rangle$')
#   ax1.plot(avg.x_avg, avg.pc_avg*(avg.v_avg + avg.va_avg)**(gc), 'r-o', label='$\\langle P_c\\rangle (\\langle v\\rangle + \\langle v_A\\rangle)^{\\gamma_c}$')
#   # ax1.plot(avg.x_avg, fit_invar[0]*avg.x_avg + fit_invar[1], 'k--', label='slope=${:.3f}$'.format(fit_invar[0]))
#   # ax1.plot(avg.x_avg, fit_invar2[0]*avg.x_avg + fit_invar2[1], color='brown', linestyle='--', label='slope=${:.3f}$'.format(fit_invar2[0]))

#   # ax2.loglog(-np.gradient(avg.pc_avg, avg.x_avg), -avg.cr_pushing_avg)
#   # ax2.loglog(-np.gradient(avg.pc_avg, avg.x_avg), np.exp(fit_push[1])*(-np.gradient(avg.pc_avg, avg.x_avg))**(fit_push[0]), 'k--', label='slope=${:.3f}$'.format(fit_push[0]))
#   ax2.loglog(avg.x_avg, -np.gradient(avg.pc0, avg.x_avg), 'k--', label='Initial')
#   # ax2.loglog(avg.x_avg, -np.gradient(avg.pc_avg, avg.x_avg), 'b', label='$-\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax2.loglog(avg.x_avg, -avg.cr_pushing_avg, 'r', label='$\\langle\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   # ax3.loglog(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg), -avg.cr_heating_avg) 
#   # ax3.loglog(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg), np.exp(fit_heat[1])*(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg))**(fit_heat[0]), \
#     # 'k--', label='slope=${:.3f}$'.format(fit_heat[0]))
#   ax3.semilogy(avg.x_avg, -(avg.v0 + avg.va0)*np.gradient(avg.pc0, avg.x_avg), 'k--', label='Initial')
#   ax3.semilogy(avg.x_avg, -(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg), 'b', label='$-(\\langle v\\rangle + \\langle v_A\\rangle)\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax3.semilogy(avg.x_avg, -avg.cr_heating_avg - avg.cr_workdone_avg, 'r', label='$\\langle (v + v_A)\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   # ax3.loglog(avg.x_avg, -avg.cr_couple_avg, label='$\\langle -(v + v_A)\\nabla P_c\\rangle$')
#   # ax3.loglog(avg.x_avg, -np.gradient(avg.fc_avg, avg.x_avg), 'g', label='$\\mathrm{d} \\langle F_c\\rangle/\\mathrm{d} x$')

#   ax4.plot(avg.x_avg, np.abs(avg.cr_heating_avg/((avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg))))

#   ax5.plot(avg.x_avg, avg.fc_avg, label='$\\langle F_c \\rangle$')
#   ax5.plot(avg.x_avg, gc1*avg.pc_avg*(avg.v_avg + avg.va_avg) - avg.kappa*np.gradient(avg.pc_avg, avg.x_avg)/(gc - 1.), 'k--', label='$\\langle (v + v_A) \\rangle \\langle E_c + P_c\\rangle - \\kappa\\mathrm{d} \\langle E_c\\rangle/\\mathrm{d} x$')
#   # ax5.plot(avg.x_avg, avg.steady_fc_avg, 'k--', label='$\\langle (v + v_A) (E_c + P_c) - \\kappa\\mathrm{d} E_c/\\mathrm{d} x$')

#   ax6.plot(avg.x_avg, ((avg.fc_avg/(gc1*avg.pc_avg)) - avg.v_avg)/avg.va_avg)

#   ax7.plot(avg.x_avg, np.gradient(avg.fc_avg, avg.x_avg)/avg.cr_heating_avg)

#   ax8.plot(avg.x_avg, np.gradient(avg.pc_avg, avg.x_avg)/avg.cr_pushing_avg)

#   ax9.plot(avg.x_avg, ((avg.fc_avg/(gc1*avg.pc_avg)) - avg.v_avg))

#   ax10.plot(avg.time_array, -avg.total_general_heating_time - avg.dfc_time)

#   ax11.semilogy(avg.x_avg, -avg.v0*np.gradient(avg.pc0, avg.x_avg), 'k--', label='Initial')
#   ax11.semilogy(avg.x_avg, -avg.v_avg*np.gradient(avg.pc_avg, avg.x_avg), 'b', label='$-\\langle v\\rangle\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax11.semilogy(avg.x_avg, -avg.cr_workdone_avg, 'r', label='$\\langle v \\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')

#   ax12.semilogy(avg.x_avg, -avg.va0*np.gradient(avg.pc0, avg.x_avg), 'k--', label='Initial')
#   ax12.semilogy(avg.x_avg, -avg.va_avg*np.gradient(avg.pc_avg, avg.x_avg), 'b', label='$-\\langle v_A\\rangle\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax12.semilogy(avg.x_avg, -avg.cr_heating_avg, 'r', label='$\\langle v_A \\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')

#   # fit_time = np.polyfit(avg.time_array, -avg.total_general_heating_time - avg.dfc_time, 0)
#   # time_integrated_pc = fit_time[0]
#   # time_int_pc = np.sum(avg.pc_avg*avg.dx) - np.sum(avg.pc0*avg.dx)
#   # print('Time integrated P_c: {:f}'.format(time_integrated_pc))
#   # print('Space integrated P_c: {:f}'.format(time_int_pc))

#   ax1.margins(x=0)
#   ax1.set_xlabel('$x$')
#   ax1.legend(frameon=False)
#   # ax1.set_ylabel('$\\langle P_c\\rangle (\\langle v\\rangle + \\langle v_A\\rangle)^{\\gamma_c}$')

#   ax2.legend(frameon=False)
#   ax3.legend(frameon=False)
#   ax11.legend(frameon=False)
#   ax12.legend(frameon=False)
#   ax2.margins(x=0)
#   ax3.margins(x=0)
#   ax4.margins(x=0)
#   ax11.margins(x=0)
#   ax12.margins(x=0)
#   ax4.set_ylim(top=1.0)
#   # ax2.set_xlabel('$-\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   # ax3.set_xlabel('$-(\\langle v\\rangle + \\langle v_A\\rangle)\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax2.set_xlabel('$x$')
#   ax3.set_xlabel('$x$')
#   ax11.set_xlabel('$x$')
#   ax12.set_xlabel('$x$')
#   ax4.set_xlabel('$x$')
#   # ax2.set_ylabel('$\\langle\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   # ax3.set_ylabel('$\\langle(v + v_A)\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   ax4.set_ylabel('Suppression ratio')
#   ax4.xaxis.set_minor_locator(AutoMinorLocator())
#   ax4.yaxis.set_minor_locator(AutoMinorLocator())

#   ax5.legend(frameon=False)
#   ax5.margins(x=0)
#   ax5.set_xlabel('$x$')
#   ax5.set_ylabel('$F_c$')
#   ax5.xaxis.set_minor_locator(AutoMinorLocator())
#   ax5.yaxis.set_minor_locator(AutoMinorLocator())

#   ax6.margins(x=0)
#   ax6.set_xlabel('$x$')
#   ax6.set_ylabel('$v_\\mathrm{s, eff}/v_A$')
#   ax6.xaxis.set_minor_locator(AutoMinorLocator())
#   ax6.yaxis.set_minor_locator(AutoMinorLocator())

#   ax7.margins(x=0)
#   ax7.axhline(1., color='k', linestyle='--')
#   ax7.set_ylim(bottom=0)
#   ax7.set_xlabel('$x$')
#   ax7.set_ylabel('$\\nabla\\langle F_c\\rangle/\\langle (v + v_A)\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   ax7.xaxis.set_minor_locator(AutoMinorLocator())
#   ax7.yaxis.set_minor_locator(AutoMinorLocator())

#   ax8.margins(x=0)
#   ax8.axhline(1., color='k', linestyle='--')
#   ax8.set_ylim(bottom=0, top=1.2)
#   ax8.set_xlabel('$x$')
#   ax8.set_ylabel('$\\nabla\\langle P_c\\rangle/\\langle \\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   ax8.xaxis.set_minor_locator(AutoMinorLocator())
#   ax8.yaxis.set_minor_locator(AutoMinorLocator())

#   ax9.margins(x=0)
#   ax9.set_xlabel('$x$')
#   ax9.set_ylabel('$v_\\mathrm{s, eff}$')
#   ax9.xaxis.set_minor_locator(AutoMinorLocator())
#   ax9.yaxis.set_minor_locator(AutoMinorLocator())

#   ax10.margins(x=0)
#   ax10.set_xlabel('$t$')
#   ax10.set_ylabel('$\\Sigma \\partial P_c/\\partial t$')
#   ax10.xaxis.set_minor_locator(AutoMinorLocator())
#   ax10.yaxis.set_minor_locator(AutoMinorLocator())

#   fig1.tight_layout()
#   fig2.tight_layout()
#   fig3.tight_layout()
#   fig4.tight_layout()
#   fig5.tight_layout()
#   fig6.tight_layout()
#   fig7.tight_layout()
#   fig8.tight_layout()
#   fig9.tight_layout()
#   fig10.tight_layout()
#   fig11.tight_layout()
#   fig12.tight_layout()
#   fig1.savefig('./invar.png', dpi=300)
#   fig2.savefig('./source_pcgrad.png', dpi=300)
#   fig3.savefig('./source_heating.png', dpi=300)
#   fig4.savefig('./suppressed', dpi=300)
#   fig5.savefig('./fc_avg_compare.png', dpi=300)
#   fig6.savefig('./effective_vs.png', dpi=300)
#   fig7.savefig('./ratio_heating.png', dpi=300)
#   fig8.savefig('./ratio_forcing.png', dpi=300)
#   fig9.savefig('./v_eff.png', dpi=300)
#   fig10.savefig('./rate_of_ec.png', dpi=300)
#   fig11.savefig('./source_workdone.png', dpi=300)
#   fig12.savefig('./source_vaheat.png', dpi=300)

#   plt.close('all')
#   plotdefault()

#########################
# # Plots for publication

# # Bottleneck stationary
# latexify(columns=1, square=True)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)

# ax1.plot(one.x1v, one.rho_array[0], label='$\\rho$')
# ax1.plot(one.x1v, one.ecr_array[0]/3., label='$P_c$')
# ax1.plot(one.x1v, one.fcx_array[0]*one.vmax/3., label='$F_c/3$')
# # ax1.plot(stair2.x_stair[400], stair2.pc_stair[400], 'k--')

# ax1.axhline(4.216369986534119, linestyle='--', color='k')
# ax1.axhline(2.8726056365288173, linestyle='--', color='k')
# ax1.axhline(3.1621907885216043, linestyle=':', color='r')
# ax1.axhline(0.6812982125487347, linestyle=':', color='r')

# ax1.legend(frameon=False, loc='center right')
# ax1.margins(x=0)
# ax1.set_xlabel('$x$')
# ax1.set_title('$t=2000$')
# ax1.xaxis.set_minor_locator(AutoMinorLocator())
# ax1.yaxis.set_minor_locator(AutoMinorLocator())

# fig1.tight_layout()
# fig1.savefig('./bottleneck_publish.png', dpi=300)

# plt.show()
# plt.close('all')
# plotdefault()



# # Bottleneck move
# # Shock check time:3000, 320-340-411-440
# v_jump = -0.9128709291752768
# vpvavj = one.vx_sh + one.va_sh - v_jump
# index_vpvavj_min = np.argmin(vpvavj)
# vpvavj_min = vpvavj[index_vpvavj_min]
# pc0 = one.pc_sh[0]
# pc = pc0*vpvavj_min**gc/vpvavj[index_vpvavj_min:one.ichk]**gc

# latexify(columns=1, square=True)
# fig1 = plt.figure()
# fig2 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)

# ax1.plot(one.x1v, one.rho_array[0], 'k--')
# ax1.plot(one.x1v, one.rho_array[1])

# ax2.plot(one.x1v, one.ecr_array[0]/3., 'k--')
# ax2.plot(one.x1v, one.ecr_array[1]/3.)
# ax2.plot(one.x_sh[:index_vpvavj_min], pc0*np.ones(np.size(one.x_sh[:index_vpvavj_min])), 'C1', linewidth=2)
# ax2.plot(one.x_sh[index_vpvavj_min:one.ichk], pc, 'C1', linewidth=2, label='w/ $v_\\mathrm{jump}$')
# ax2.plot(one.x_sh[one.ichk:], pc[-1]*np.ones(np.size(one.x_sh[one.ichk:])), 'C1', linewidth=2)

# ax1.arrow(600, 0.7, -200, 0, head_length=50, width=0.02, color='r')
# ax1.margins(x=0)
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$\\rho$')
# ax1.xaxis.set_minor_locator(AutoMinorLocator())
# ax1.yaxis.set_minor_locator(AutoMinorLocator())

# ax2.arrow(600, 2.3, -200, 0, head_length=50, width=0.05, color='r')
# ax2.margins(x=0)
# ax2.set_xlabel('$x$')
# ax2.set_ylabel('$P_c$')
# ax2.xaxis.set_minor_locator(AutoMinorLocator())
# ax2.yaxis.set_minor_locator(AutoMinorLocator())
# ax2.legend(frameon=False)

# fig1.tight_layout()
# fig2.tight_layout()
# fig1.savefig('./bottleneck_moving_publish_rho.png', dpi=300)
# fig2.savefig('./bottleneck_moving_publish_pc.png', dpi=300)

# plt.show()
# plt.close('all')
# plotdefault()



# # Bottleneck with and without bumpts
# # alpha=1, beta=1, eta=0.01, ms=0.015, psi=0, 1<x<6
# latexify(columns=1)

# fig1 = plt.figure()
# fig2 = plt.figure()
# fig3 = plt.figure()
# fig = [fig1, fig2, fig3]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax3 = fig3.add_subplot(111)
# ax = [ax1, ax2, ax3]

# with h5py.File('../analytics/power_analysis.hdf5', 'r') as fp:
#   x = np.array(fp['x'])
#   rho = np.array(fp['rho'])
#   v = np.array(fp['v'])
#   B = fp.attrs['B']

# ax1.semilogy(x, 1./(v + B/np.sqrt(rho)), 'k--', label='No bumps')
# ax1.semilogy(one.x1v, 1./(one.vx_array[1] + one.b0/np.sqrt(one.rho_array[1])), label='With bumps')
# ax2.semilogy(one.x1v, one.ecr_array[0]*(gc - 1.), 'k--')
# ax2.semilogy(one.x1v, one.ecr_array[1]*(gc - 1.))
# ax3.plot(one.x1v, one.fcx_array[0]*one.vmax, 'k--')
# ax3.plot(one.x1v, one.fcx_array[1]*one.vmax)

# ax1.margins(x=0)
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$(v + v_A)^{-1}$')
# ax1.xaxis.set_minor_locator(AutoMinorLocator())
# ax1.legend(frameon=False)

# ax2.margins(x=0)
# ax2.set_xlabel('$x$')
# ax2.set_ylabel('$P_c$')
# ax2.xaxis.set_minor_locator(AutoMinorLocator())

# ax3.margins(x=0)
# ax3.set_xlabel('$x$')
# ax3.set_ylabel('$F_c$')
# ax3.xaxis.set_minor_locator(AutoMinorLocator())
# ax3.yaxis.set_minor_locator(AutoMinorLocator())

# for i, figu in enumerate(fig):
#   figu.tight_layout()

# fig1.savefig('./rho_bottle_bumps.png', dpi=300)
# fig2.savefig('./pc_bottle_bumps.png', dpi=300)
# fig3.savefig('./fc_bottle_bumps.png', dpi=300)

# plt.show()
# plt.close('all')
# plotdefault()



# # Time Snapshot
# xl = 1. 
# xu = 6. 
# index_xl = np.argmin(np.abs(one.x1v - xl))
# index_xu = np.argmin(np.abs(one.x1v - xu))

# x = one.x1v[index_xl:index_xu]

# rho0 = one.rho_array[0][index_xl:index_xu]
# v0 = one.vx_array[0][index_xl:index_xu]
# pg0 = one.pg_array[0][index_xl:index_xu]
# pc0 = one.ecr_array[0][index_xl:index_xu]*(gc - 1.)
# fc0 = one.fcx_array[0][index_xl:index_xu]*one.vmax 
# T0 = pg0/rho0 

# rho = one.rho_array[1][index_xl:index_xu]
# v = one.vx_array[1][index_xl:index_xu]
# pg = one.pg_array[1][index_xl:index_xu]
# pc = one.ecr_array[1][index_xl:index_xu]*(gc - 1.)
# fc = one.fcx_array[1][index_xl:index_xu]*one.vmax 
# T = pg/rho 

# Tmin = 0.01

# latexify(columns=2)
# fig1 = plt.figure()
# grids = gs.GridSpec(2, 3, figure=fig1)
# ax1 = fig1.add_subplot(grids[0, 0])
# ax2 = fig1.add_subplot(grids[0, 1])
# ax3 = fig1.add_subplot(grids[0, 2])
# ax4 = fig1.add_subplot(grids[1, 0])
# ax5 = fig1.add_subplot(grids[1, 1])
# ax6 = fig1.add_subplot(grids[1, 2])
# ax = [ax1, ax2, ax3, ax4, ax5, ax6]

# ax1.semilogy(x, rho)
# ax1.semilogy(x, rho0, 'k--')
# ax2.plot(x, v)
# ax2.plot(x, v0, 'k--')
# ax3.semilogy(x[T>Tmin], pg[T>Tmin])
# ax3.semilogy(x, pg0, 'k--')
# ax4.semilogy(x, pc)
# ax4.semilogy(x, pc0, 'k--')
# ax5.plot(x, fc)
# ax5.plot(x, fc0, 'k--')
# ax6.semilogy(x[T>Tmin], T[T>Tmin])
# ax6.semilogy(x, T0, 'k--')

# lab = ['$\\rho$', '$v$', '$P_g$', '$P_c$', '$F_c$', '$T$']

# for i, axes in enumerate(ax):
#   axes.margins(x=0)
#   axes.set_xlabel('$x$')
#   axes.set_ylabel(lab[i])
#   axes.xaxis.set_minor_locator(AutoMinorLocator())
#   if i in [1, 4]:
#     axes.yaxis.set_minor_locator(AutoMinorLocator())

# fig1.tight_layout()
# fig1.savefig('./time_snapshot.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()



# # Zoom-in profile and bottleneck with motion
# # t = 0.68, alpha1beta.6eta.01ms.015psi0
# # Resolution 4096
# x_stop = 2.201
# index_xstop = np.argmin(np.abs(one.x_sh - x_stop))
# vpva_start = one.v2 + one.va2 
# vpva_wrong_start = one.v2 + one.va2 + one.vsh
# pc_start = np.amax(one.pc_sh)
# vpva = one.vx_sh[one.ichk:index_xstop] + one.va_sh[one.ichk:index_xstop]
# vpva_wrong = one.vx_sh[one.ichk:index_xstop] + one.va_sh[one.ichk:index_xstop] + one.vsh
# pc_stair = pc_start*(vpva_start/vpva)**(gc)
# pc_stair_wrong = pc_start*(vpva_wrong_start/vpva_wrong)**(gc)

# pc_concat = np.zeros(np.size(one.x_sh))
# pc_concat_wrong = np.zeros(np.size(one.x_sh))

# pc_concat[:one.ichk] = pc_start
# pc_concat_wrong[:one.ichk] = pc_start
# pc_concat[one.ichk:index_xstop] = pc_stair
# pc_concat_wrong[one.ichk:index_xstop] = pc_stair_wrong 
# pc_concat[index_xstop:] = pc_stair[-1]
# pc_concat_wrong[index_xstop:] = pc_stair_wrong[-1]

# mach1 = one.v1/one.cs1 
# rho_post_shock = ((gg + 1.)*mach1**2/((gg - 1.)*mach1**2 + 2.))*one.rho1
# pg_post_shock = ((2.*gg*mach1**2 - (gg - 1.))/(gg + 1.))*one.pg1 

# latexify(columns=1)

# fig1 = plt.figure()
# fig2 = plt.figure()
# fig3 = plt.figure()
# fig = [fig1, fig2, fig3]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax3 = fig3.add_subplot(111)
# ax = [ax1, ax2, ax3]

# ax1.plot(one.x_sh, one.pc_sh)
# ax1.plot(one.x_sh, pc_concat, 'g--', label='w/ $v_\\mathrm{jump}$')
# # ax1.plot(np.ma.masked_array(one.x_sh, one.x_sh<one.start), np.ma.masked_array(pc_concat_wrong, one.x_sh<one.start), 'r--', label='w/o $v_\\mathrm{jump}$')
# # ax1.legend(frameon=False)

# ax2.plot(one.x_sh, one.rho_sh)
# ax2.hlines(one.rho1, one.start, one.check, colors='k', linestyle='--')
# ax2.hlines(rho_post_shock, one.start, one.check, colors='k', linestyle='--')

# ax3.plot(one.x_sh, one.pg_sh)
# ax3.hlines(one.pg1, one.start, one.check, colors='k', linestyle='--')
# ax3.hlines(pg_post_shock, one.start, one.check, colors='k', linestyle='--')

# lab = ['$P_c$', '$\\rho$', '$P_g$']

# for i, axes in enumerate(ax):
#   axes.axvspan(one.begin, one.start, alpha=0.1, color='yellow')
#   axes.axvspan(one.start, one.check, alpha=0.1, color='orange')
#   axes.axvspan(one.check, x_stop, alpha=0.1, color='red')
#   axes.axvspan(x_stop, one.end, alpha=0.1, color='#3776ab')
#   axes.margins(x=0)
#   axes.set_xlabel('$x$')
#   axes.set_ylabel(lab[i])
#   axes.xaxis.set_minor_locator(AutoMinorLocator())
#   axes.yaxis.set_minor_locator(AutoMinorLocator())

# for i, figu in enumerate(fig):
#   figu.tight_layout()
# fig1.savefig('./zoom_in_pc.png', dpi=300)
# fig2.savefig('./zoom_in_rho.png', dpi=300)
# fig3.savefig('./zoom_in_pg.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()



# # Zoom-in profile and bottleneck with motion (isothermal)
# # t = 0.68, alpha1beta.6eta.01ms.015psi0
# # Resolution 4096
# vpva_start = one.v1 + one.va1 
# vpva_wrong_start = one.v1 + one.va1 + one.vsh
# pc_start = np.amax(one.pc_sh)
# vpva = one.vx_sh[one.istr:one.ichk] + one.va_sh[one.istr:one.ichk]
# vpva_wrong = one.vx_sh[one.istr:one.ichk] + one.va_sh[one.istr:one.ichk] + one.vsh
# pc_stair = pc_start*(vpva_start/vpva)**(gc)
# pc_stair_wrong = pc_start*(vpva_wrong_start/vpva_wrong)**(gc)

# pc_concat = np.zeros(np.size(one.x_sh))
# pc_concat_wrong = np.zeros(np.size(one.x_sh))

# pc_concat[:one.istr] = pc_start
# pc_concat_wrong[:one.istr] = pc_start
# pc_concat[one.istr:one.ichk] = pc_stair
# pc_concat_wrong[one.istr:one.ichk] = pc_stair_wrong 
# pc_concat[one.ichk:] = pc_stair[-1]
# pc_concat_wrong[one.ichk:] = pc_stair_wrong[-1]

# mach1 = np.abs(one.v2)/one.cs2 
# rho_post_shock = ((gg + 1.)*mach1**2/((gg - 1.)*mach1**2 + 2.))*one.rho2

# latexify(columns=1)

# fig1 = plt.figure()
# fig2 = plt.figure()
# fig = [fig1, fig2]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax = [ax1, ax2]

# ax1.plot(one.x_sh, one.pc_sh)
# ax1.plot(one.x_sh, pc_concat, 'g--', label='w/ $v_\\mathrm{jump}$')
# ax1.plot(np.ma.masked_array(one.x_sh, one.x_sh<one.start), np.ma.masked_array(pc_concat_wrong, one.x_sh<one.start), 'r--', label='w/o $v_\\mathrm{jump}$')
# ax1.legend(frameon=False)

# ax2.plot(one.x_sh, one.rho_sh)
# ax2.hlines(one.rho2, one.start, one.check, colors='k', linestyle='--')
# ax2.hlines(rho_post_shock, one.start, one.check, colors='k', linestyle='--')

# lab = ['$P_c$', '$\\rho$']

# for i, axes in enumerate(ax):
#   axes.axvspan(one.begin, one.start, alpha=0.1, color='yellow')
#   axes.axvspan(one.start, one.check, alpha=0.1, color='orange')
#   axes.axvspan(one.check, one.end, alpha=0.1, color='red')
#   axes.margins(x=0)
#   axes.set_xlabel('$x$')
#   axes.set_ylabel(lab[i])
#   axes.xaxis.set_minor_locator(AutoMinorLocator())
#   axes.yaxis.set_minor_locator(AutoMinorLocator())

# for i, figu in enumerate(fig):
#   figu.tight_layout()
# fig1.savefig('./zoom_in_pc_iso.png', dpi=300)
# fig2.savefig('./zoom_in_rho_iso.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()



# # Early time evolution of the staircase
# latexify(columns=2)

# fig = plt.figure()
# grids = gs.GridSpec(2, 3, figure=fig)
# ax1 = fig.add_subplot(grids[0, 0])
# ax2 = fig.add_subplot(grids[0, 1])
# ax3 = fig.add_subplot(grids[0, 2])
# ax4 = fig.add_subplot(grids[1, 0])
# ax5 = fig.add_subplot(grids[1, 1])
# ax6 = fig.add_subplot(grids[1, 2])
# ax = [ax1, ax2, ax3, ax4, ax5, ax6]

# xl = 1. 
# xu = 6. 
# index_xl = np.argmin(np.abs(one.x1v - xl))
# index_xu = np.argmin(np.abs(one.x1v - xu))

# for i, axes in enumerate(ax):
#   axes.semilogy(one.x1v[index_xl:index_xu], one.ecr_array[i][index_xl:index_xu]*(gc - 1.), label='t={:.2f}'.format(one.time_array[i]))
#   axes.set_xlabel('$x$')
#   axes.set_ylabel('$P_c$')
#   axes.margins(x=0)
#   axes.legend(frameon=False)
#   axes.xaxis.set_minor_locator(AutoMinorLocator())

# fig.tight_layout()
# fig.savefig('./time_evo.png', dpi=300)

# plt.show()
# plt.close('all')
# plotdefault()



# # Visual comparison of plateau sizes for different alphas
# latexify(columns=1)

# fig1 = plt.figure() 
# ax1 = fig1.add_subplot(111)

# xl = 1. 
# xu = 6. 
# index_xl = np.argmin(np.abs(one.x1v - xl))
# index_xu = np.argmin(np.abs(one.x1v - xu))

# ax1.semilogy(one.x1v[index_xl:index_xu], one.ecr_array[1][index_xl:index_xu]*(gc - 1.))

# ax1.margins(x=0)
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$P_c$')
# ax1.xaxis.set_minor_locator(AutoMinorLocator())

# fig1.tight_layout()
# fig1.savefig('./pc_snapshot.png', dpi=300)

# plt.show() 
# plt.close('all')
# plotdefault()



# # Characteristic height
# n_alpha = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
# # n_alpha = np.array([1., 2., 3., 5., 6.])
# n_beta = np.array([0.6, 0.8, 1., 2., 3., 4.])
# n_eta = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1])

# h_alpha = np.array([0.2128, 0.3405, 0.3419, 0.3805, 0.3659, 0.2965, 0.2712, 0.2364, 0.1962, 0.1763, 0.1586, 0.1452, 0.1357, 0.1142, 0.1094])
# # h_alpha = np.array([0.2293, 0.2081, 0.1657, 0.1223, 0.1087])
# h_beta = np.array([0.3553, 0.3282, 0.2965, 0.3612, 0.3108, 0.3375])
# h_eta = np.array([0.2965, 0.3262, 0.3109, 0.3554, 0.3901, 0.4028])

# latexify(columns=1, square=False)

# logx_alpha = True
# logy_alpha = True
# logx_beta = False
# logy_beta = False
# logx_eta = True
# logy_eta = True

# fig1 = plt.figure()
# fig2 = plt.figure()
# fig3 = plt.figure()
# fig = [fig1, fig2, fig3]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax3 = fig3.add_subplot(111)
# ax = [ax1, ax2, ax3]

# ax1.scatter(n_alpha, h_alpha, s=8, c='b', marker='*')
# fit_start_alpha = 0
# fit_alpha = np.polyfit(np.log(n_alpha[fit_start_alpha:]), np.log(h_alpha[fit_start_alpha:]), deg=1)
# ax1.plot(n_alpha[fit_start_alpha:], np.exp(fit_alpha[1])*n_alpha[fit_start_alpha:]**(fit_alpha[0]), 'k--', label='index={:.2f}'.format(fit_alpha[0]))
# ax2.scatter(n_beta, h_beta, s=8, c='r', marker='*')
# ax3.scatter(n_eta, h_eta, s=8, c='g', marker='*')
# fit_eta = np.polyfit(np.log(n_eta), np.log(h_eta), deg=1)
# ax3.plot(n_eta, np.exp(fit_eta[1])*n_eta**(fit_eta[0]), 'k--', label='index={:.2f}'.format(fit_eta[0]))

# ax1.set_xlabel('$\\alpha_0$')
# ax1.set_ylabel('$h_*$')
# ax1.legend(frameon=False)
# if logx_alpha:
#   ax1.set_xscale('log')
# else:
#   ax1.xaxis.set_minor_locator(AutoMinorLocator())
# if logy_alpha:
#   ax1.set_yscale('log')
#   ax1.yaxis.set_minor_formatter(ticker.ScalarFormatter())
# else:
#   ax1.yaxis.set_minor_locator(AutoMinorLocator())

# ax2.set_xlabel('$\\beta_0$')
# ax2.set_ylabel('$h_*$')
# if logx_beta:
#   ax2.set_xscale('log')
# else:
#   ax2.xaxis.set_minor_locator(AutoMinorLocator())
# if logy_beta:
#   ax2.set_yscale('log')
#   ax2.yaxis.set_minor_formatter(ticker.ScalarFormatter())
# else:
#   ax2.yaxis.set_minor_locator(AutoMinorLocator())

# ax3.set_xlabel('$\\eta_0$')
# ax3.set_ylabel('$h_*$')
# ax3.legend(frameon=False)
# if logx_eta: 
#   ax3.set_xscale('log')
# else:
#   ax3.xaxis.set_minor_locator(AutoMinorLocator())
# if logy_eta:
#   ax3.set_yscale('log')
#   ax3.yaxis.set_minor_formatter(ticker.ScalarFormatter())
# else:
#   ax3.yaxis.set_minor_locator(AutoMinorLocator())

# for i, figu in enumerate(fig):
#   figu.tight_layout()
# fig1.savefig('./h_alpha.png', dpi=300)
# fig2.savefig('./h_beta.png', dpi=300)
# fig3.savefig('./h_eta.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()



# # Change in mdot, delta pc and delta fc for different alphas, betas and etas
# alpha = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5., 6., 7., 8.])
# beta = np.array([0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.3, 0.5, 0.6, 0.8, 1., 2., 3., 4.])
# eta = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1])

# mdot_alpha = np.array([0.969, 0.977, 1.063, 1.123, 1.175, 1.388, 1.713, 1.825, 1.861, 1.890, 1.901, 1.925, 1.944])
# dpc_alpha = np.array([1.120, 1.184, 1.207, 1.230, 1.234, 1.304, 1.269, 1.210, 1.186, 1.187, 1.175, 1.158, 1.141])
# dfc_alpha = np.array([0.947, 0.932, 0.911, 0.915, 0.899, 0.896, 0.852, 0.844, 0.844, 0.848, 0.846, 0.848, 0.843])

# mdot_beta = np.array([5.635, 4.318, 4.232, 3.943, 3.354, 3.078, 2.140, 1.680, 1.685, 1.466, 1.388, 1.091, 0.937, 0.896])
# dpc_beta = np.array([1.408, 1.393, 1.423, 1.376, 1.364, 1.666, 1.500, 1.463, 1.433, 1.352, 1.304, 1.117, 1.053, 1.036])
# dfc_beta = np.array([0.671, 0.739, 0.752, 0.727, 0.783, 0.858, 0.888, 0.919, 0.889, 0.908, 0.896, 0.864, 0.914, 0.953])

# mdot_eta = np.array([1.388, 1.378, 1.312, 1.209, 1.290, 1.211])
# dpc_eta = np.array([1.304, 1.299, 1.271, 1.271, 1.255, 1.260])
# dfc_eta = np.array([0.896, 0.879, 0.880, 0.899, 0.871, 0.884])

# mdot_beta_fit = np.polyfit(np.log(beta), np.log(mdot_beta), 1)

# latexify(columns=2)

# fig = plt.figure()
# grids = gs.GridSpec(3, 3, figure=fig)
# ax1 = fig.add_subplot(grids[0, 0])
# ax2 = fig.add_subplot(grids[0, 1])
# ax3 = fig.add_subplot(grids[0, 2])
# ax4 = fig.add_subplot(grids[1, 0])
# ax5 = fig.add_subplot(grids[1, 1])
# ax6 = fig.add_subplot(grids[1, 2])
# ax7 = fig.add_subplot(grids[2, 0])
# ax8 = fig.add_subplot(grids[2, 1])
# ax9 = fig.add_subplot(grids[2, 2])
# ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

# ax1.plot(alpha, mdot_alpha, '--o', color='C0', marker='o', markersize='5', fillstyle='none')
# ax2.plot(alpha, dpc_alpha, '--o', color='C1', marker='o', markersize='5', fillstyle='none')
# ax3.plot(alpha, dfc_alpha, '--o', color='C2', marker='o', markersize='5', fillstyle='none')

# ax4.plot(beta, mdot_beta, '--o', color='C0', marker='o', markersize='5', fillstyle='none')
# ax4.plot(beta[:-1], np.exp(mdot_beta_fit[0]*np.log(beta[:-1]) + mdot_beta_fit[1]), 'k:', label='index={:.3f}'.format(mdot_beta_fit[0]))
# ax5.plot(beta, dpc_beta, '--o', color='C1', marker='o', markersize='5', fillstyle='none')
# ax6.plot(beta, dfc_beta, '--o', color='C2', marker='o', markersize='5', fillstyle='none')

# ax7.plot(eta, mdot_eta, '--o', color='C0', marker='o', markersize='5', fillstyle='none')
# ax8.plot(eta, dpc_eta, '--o', color='C1', marker='o', markersize='5', fillstyle='none')
# ax9.plot(eta, dfc_eta, '--o', color='C2', marker='o', markersize='5', fillstyle='none')

# ax1.set_xlabel('$\\alpha_0$')
# ax2.set_xlabel('$\\alpha_0$')
# ax3.set_xlabel('$\\alpha_0$')

# ax4.set_xlabel('$\\beta_0$')
# ax5.set_xlabel('$\\beta_0$')
# ax6.set_xlabel('$\\beta_0$')

# ax7.set_xlabel('$\\eta_0$')
# ax8.set_xlabel('$\\eta_0$')
# ax9.set_xlabel('$\\eta_0$')

# ax1.set_ylabel('$\\langle\\dot{M}\\rangle/\\dot{M}_0$')
# ax2.set_ylabel('$\\langle\\Delta P_c\\rangle/\\Delta P_{c0}$')
# ax3.set_ylabel('$\\langle\\Delta F_c\\rangle/\\Delta F_{c0}$')

# ax4.set_ylabel('$\\langle\\dot{M}\\rangle/\\dot{M}_0$')
# ax5.set_ylabel('$\\langle\\Delta P_c\\rangle/\\Delta P_{c0}$')
# ax6.set_ylabel('$\\langle\\Delta F_c\\rangle/\\Delta F_{c0}$')

# ax7.set_ylabel('$\\langle\\dot{M}\\rangle/\\dot{M}_0$')
# ax8.set_ylabel('$\\langle\\Delta P_c\\rangle/\\Delta P_{c0}$')
# ax9.set_ylabel('$\\langle\\Delta F_c\\rangle/\\Delta F_{c0}$')

# # ax4.legend(frameon=False)

# for i, axes in enumerate(ax):
#   axes.set_xscale('log')
#   if axes != ax4: 
#     axes.yaxis.set_minor_locator(AutoMinorLocator())
#   else:
#     axes.set_yscale('log')
#     axes.yaxis.set_minor_formatter(ticker.ScalarFormatter())
#     axes.yaxis.set_major_formatter(ticker.NullFormatter())

# fig.tight_layout()
# # fig.savefig('./alpha_beta_eta_avg.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()



# # Resolution and reduced speed of light study
# latexify(columns=1)

# resolution = np.array([0.0588, 0.117, 0.233, 0.465, 0.926, 1.85, 3.70 , 7.41,  25.9])
# mdot = np.array([1.155, 1.282, 1.257, 1.355, 1.365, 1.388, 1.339, 1.449, 1.465])
# delpc = np.array([1.204, 1.270, 1.319, 1.339, 1.353, 1.304, 1.379, 1.407, 1.339])
# delfc = np.array([0.951, 0.963, 0.982, 0.955, 0.933, 0.896, 0.914, 0.924, 0.900])

# fig1 = plt.figure() 
# fig2 = plt.figure() 
# fig3 = plt.figure()  
# fig = [fig1, fig2, fig3]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111) 
# ax3 = fig3.add_subplot(111) 
# ax = [ax1, ax2, ax3] 

# ax1.semilogx(resolution, mdot, 'b--o', marker='o', markersize='5', fillstyle='none')
# ax2.semilogx(resolution, delpc, 'r--o', marker='o', markersize='5', fillstyle='none')
# ax3.semilogx(resolution, delfc, 'g--o', marker='o', markersize='5', fillstyle='none')

# lab = ['$\\langle\\dot{M}\\rangle/\\dot{M}_0$', '$\\langle\\Delta P_c\\rangle/\\Delta P_{c0}$', '$\\langle\\Delta F_c\\rangle/\\Delta F_{c0}$']

# for i, axes in enumerate(ax):
#   axes.set_xlabel('$\\langle l_\\mathrm{diff}\\rangle/\\Delta x$')
#   axes.set_ylabel(lab[i])
#   axes.yaxis.set_minor_locator(AutoMinorLocator())

# for i, figu in enumerate(fig):
#   figu.tight_layout()
# fig1.savefig('./reso_mdot.png', dpi=300)
# fig2.savefig('./reso_delpc.png', dpi=300)
# fig3.savefig('./reso_delfc.png', dpi=300)

# plt.show()
# plt.close('all')  

# plotdefault()



# # Visual comparison of pc profile for different resolutions
# with h5py.File('resolute_data.hdf5', 'a') as fp: 
#   # dset = fp.create_dataset('x_128', data=one.x1v)
#   # dset = fp.create_dataset('pc_128', data=one.ecr_array[0]*(gc - 1.))
#   # dset = fp.create_dataset('x_65536', data=one.x1v)
#   # dset = fp.create_dataset('pc_65536', data=one.ecr_array[0]*(gc - 1.))

# with h5py.File('resolute_data.hdf5', 'r') as fp:
#   x_128 = np.array(fp['x_128'])
#   x_65536 = np.array(fp['x_65536'])
#   pc_128 = np.array(fp['pc_128'])
#   pc_65536 = np.array(fp['pc_65536'])

# latexify(columns=1, square=True)

# fig1 = plt.figure() 
# ax1 = fig1.add_subplot(111)

# xu = 6. 
# index_xu_128 = np.argmin(np.abs(x_128 - xu))
# index_xu_65536 = np.argmin(np.abs(x_65536 - xu))

# ax1.semilogy(x_128[:index_xu_128], pc_128[:index_xu_128], label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 0.0588$')
# ax1.semilogy(x_65536[:index_xu_65536], pc_65536[:index_xu_65536], label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 25.9$')

# ax1.legend(frameon=False)
# ax1.margins(x=0)
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$P_c$')
# ax1.xaxis.set_minor_locator(AutoMinorLocator())

# fig1.tight_layout()
# fig1.savefig('./pc_resol.png', dpi=300)

# plt.show() 
# plt.close('all')
# plotdefault()



# # Comparison of distributions for different resolutions
# with h5py.File('./stampede2/cases/128/alpha1beta1eta.01ms.015psi0/resol_plots.hdf5', 'r') as fp:
#   width_ldiffusive_128 = np.array(fp['width_diffusive'])
#   height_crpress_128 = np.array(fp['height_crpress'])
#   plateau_128 = np.array(fp['plateau'])
# with h5py.File('./stampede2/cases/65536/alpha1beta1eta.01ms.015psi0/resol_plots.hdf5', 'r') as fp:
#   width_ldiffusive_65536 = np.array(fp['width_diffusive'])
#   height_crpress_65536 = np.array(fp['height_crpress'])
#   plateau_65536 = np.array(fp['plateau'])

# num_bins = 100 
# width_bin_128 = np.logspace(np.log10(np.amin(width_ldiffusive_128)), np.log10(np.amax(width_ldiffusive_128)), num_bins)
# width_bin_65536 = np.logspace(np.log10(np.amin(width_ldiffusive_65536)), np.log10(np.amax(width_ldiffusive_65536)), num_bins)
# height_bin_128 = np.logspace(np.log10(np.amin(height_crpress_128)), np.log10(np.amax(height_crpress_128)), num_bins)
# height_bin_65536 = np.logspace(np.log10(np.amin(height_crpress_65536)), np.log10(np.amax(height_crpress_65536)), num_bins)
# plateau_128 = np.delete(plateau_128, np.where(plateau_128==0)[0])
# plateau_bin_128 = np.logspace(np.log10(np.amin(plateau_128)), np.log10(np.amax(plateau_128)), num_bins)
# plateau_65536 = np.delete(plateau_65536, np.where(plateau_65536==0)[0])
# plateau_bin_65536 = np.logspace(np.log10(np.amin(plateau_65536)), np.log10(np.amax(plateau_65536)), num_bins)

# widths_128 = (width_bin_128[1:] - width_bin_128[:-1])
# width_hist_128 = np.histogram(width_ldiffusive_128, bins=width_bin_128)
# width_norm_128 = width_hist_128[0]/(widths_128*np.sum(width_hist_128[0]))

# widths_65536 = (width_bin_65536[1:] - width_bin_65536[:-1])
# width_hist_65536 = np.histogram(width_ldiffusive_65536, bins=width_bin_65536)
# width_norm_65536 = width_hist_65536[0]/(widths_65536*np.sum(width_hist_65536[0]))

# heights_128 = (height_bin_128[1:] - height_bin_128[:-1])
# height_hist_128 = np.histogram(height_crpress_128, bins=height_bin_128)
# height_norm_128 = height_hist_128[0]/(heights_128*np.sum(height_hist_128[0]))

# heights_65536 = (height_bin_65536[1:] - height_bin_65536[:-1])
# height_hist_65536 = np.histogram(height_crpress_65536, bins=height_bin_65536)
# height_norm_65536 = height_hist_65536[0]/(heights_65536*np.sum(height_hist_65536[0]))

# plateaus_128 = (plateau_bin_128[1:] - plateau_bin_128[:-1])
# plateau_hist_128 = np.histogram(plateau_128, bins=plateau_bin_128)
# plateau_norm_128 = plateau_hist_128[0]/(plateaus_128*np.sum(plateau_hist_128[0]))

# plateaus_65536 = (plateau_bin_65536[1:] - plateau_bin_65536[:-1])
# plateau_hist_65536 = np.histogram(plateau_65536, bins=plateau_bin_65536)
# plateau_norm_65536 = plateau_hist_65536[0]/(plateaus_65536*np.sum(plateau_hist_65536[0]))

# width_delete_128 = np.where(width_norm_128==0)[0]
# width_bin_fit_128 = np.delete(width_bin_128, width_delete_128)
# width_norm_fit_128 = np.delete(width_norm_128, width_delete_128)
# width_bin_fit_log_128 = np.log(width_bin_fit_128)
# width_norm_fit_log_128 = np.log(width_norm_fit_128)
# widths_fit_128 = np.exp(width_bin_fit_log_128[1:]) - np.exp(width_bin_fit_log_128[:-1])

# width_delete_65536 = np.where(width_norm_65536==0)[0]
# width_bin_fit_65536 = np.delete(width_bin_65536, width_delete_65536)
# width_norm_fit_65536 = np.delete(width_norm_65536, width_delete_65536)
# width_bin_fit_log_65536 = np.log(width_bin_fit_65536)
# width_norm_fit_log_65536 = np.log(width_norm_fit_65536)
# widths_fit_65536 = np.exp(width_bin_fit_log_65536[1:]) - np.exp(width_bin_fit_log_65536[:-1])

# height_delete_128 = np.where(height_norm_128==0)[0]
# height_bin_fit_128 = np.delete(height_bin_128, height_delete_128)
# height_norm_fit_128 = np.delete(height_norm_128, height_delete_128)
# height_bin_fit_log_128 = np.log(height_bin_fit_128)
# height_norm_fit_log_128 = np.log(height_norm_fit_128)
# heights_fit_128 = np.exp(height_bin_fit_log_128[1:]) - np.exp(height_bin_fit_log_128[:-1])

# height_delete_65536 = np.where(height_norm_65536==0)[0]
# height_bin_fit_65536 = np.delete(height_bin_65536, height_delete_65536)
# height_norm_fit_65536 = np.delete(height_norm_65536, height_delete_65536)
# height_bin_fit_log_65536 = np.log(height_bin_fit_65536)
# height_norm_fit_log_65536 = np.log(height_norm_fit_65536)
# heights_fit_65536 = np.exp(height_bin_fit_log_65536[1:]) - np.exp(height_bin_fit_log_65536[:-1])

# plateau_delete_128 = np.where(plateau_norm_128==0)[0]
# plateau_bin_fit_128 = np.delete(plateau_bin_128, plateau_delete_128)
# plateau_norm_fit_128 = np.delete(plateau_norm_128, plateau_delete_128)
# plateau_bin_fit_log_128 = np.log(plateau_bin_fit_128)
# plateau_norm_fit_log_128 = np.log(plateau_norm_fit_128)
# plateaus_fit_128 = np.exp(plateau_bin_fit_log_128[1:]) - np.exp(plateau_bin_fit_log_128[:-1])

# plateau_delete_65536 = np.where(plateau_norm_65536==0)[0]
# plateau_bin_fit_65536 = np.delete(plateau_bin_65536, plateau_delete_65536)
# plateau_norm_fit_65536 = np.delete(plateau_norm_65536, plateau_delete_65536)
# plateau_bin_fit_log_65536 = np.log(plateau_bin_fit_65536)
# plateau_norm_fit_log_65536 = np.log(plateau_norm_fit_65536)
# plateaus_fit_65536 = np.exp(plateau_bin_fit_log_65536[1:]) - np.exp(plateau_bin_fit_log_65536[:-1])

# latexify(columns=1, square=True)

# fig1 = plt.figure()
# fig2 = plt.figure() 
# fig3 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax3 = fig3.add_subplot(111)

# ax1.bar(np.exp(width_bin_fit_log_128[:-2]), np.exp(width_norm_fit_log_128[:-1]), widths_fit_128[:-1], align='edge', alpha=0.3, label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 0.0588$')
# ax1.bar(np.exp(width_bin_fit_log_65536[:-2]), np.exp(width_norm_fit_log_65536[:-1]), widths_fit_65536[:-1], align='edge', alpha=0.3, label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 25.9$')
# ax2.bar(np.exp(height_bin_fit_log_128[:-2]), np.exp(height_norm_fit_log_128[:-1]), heights_fit_128[:-1], align='edge', alpha=0.3, label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 0.0588$')
# ax2.bar(np.exp(height_bin_fit_log_65536[:-2]), np.exp(height_norm_fit_log_65536[:-1]), heights_fit_65536[:-1], align='edge', alpha=0.3, label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 25.9$')
# ax3.bar(np.exp(plateau_bin_fit_log_128[:-2]), np.exp(plateau_norm_fit_log_128[:-1]), plateaus_fit_128[:-1], align='edge', alpha=0.3, label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 0.0588$')
# ax3.bar(np.exp(plateau_bin_fit_log_65536[:-2]), np.exp(plateau_norm_fit_log_65536[:-1]), plateaus_fit_65536[:-1], align='edge', alpha=0.3, label='$\\langle l_\\mathrm{diff}\\rangle/\\Delta x = 25.9$')

# ax1.legend(frameon=False)
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xlabel('Width ($l_\\mathrm{diff}$)')
# ax1.set_ylabel('Distribution')
# x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
# ax1.xaxis.set_minor_locator(x_minor)

# ax2.legend(frameon=False)
# ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.set_xlabel('Height ($P_c$)')
# ax2.set_ylabel('Distribution')
# x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
# ax2.xaxis.set_minor_locator(x_minor)

# ax3.legend(frameon=False)
# ax3.set_xscale('log')
# ax3.set_yscale('log')
# ax3.set_xlabel('Plateau width H ($l_\\mathrm{diff}$)')
# ax3.set_ylabel('Distribution')
# x_minor = ticker.LogLocator(base = 10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
# ax3.xaxis.set_minor_locator(x_minor)

# fig1.tight_layout()
# fig2.tight_layout()
# fig3.tight_layout()
# fig1.savefig('./stair_width_reso.png', dpi=300)
# fig2.savefig('./stair_height_reso.png', dpi=300)
# fig3.savefig('./stair_plateau_reso.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()













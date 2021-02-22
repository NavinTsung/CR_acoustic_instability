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
import scipy.signal as signal
from scipy import interpolate
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
  def make_video(self, equi, save_path, logx=False, logy=False):
    filename = self.filename 

    # Choosing files for display
    file_array = self.file_array 
    time_array = np.zeros(np.size(file_array))

    # For no adaptive mesh refinement
    x1v = np.array(ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v'])

    # Equilibrium profile
    rho_eq = equi['rho']
    v_eq = equi['v']
    if not self.isothermal:
      pg_eq = equi['pg']
    if self.cr:
      pc_eq = equi['pc']
      fc_eq = equi['fc']

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
      rho_array = data['rho'][0, 0, :]
      v_array = data['vel1'][0, 0, :]
      if not self.isothermal:
        pg_array = data['press'][0, 0, :]
      if self.cr:
        ecr_array = data['Ec'][0, 0, :]
        fc_array = data['Fc1'][0, 0, :]*self.vmax
      if self.passive:
        r_array = data['r0'][0, 0, :]

      # Plot and save image 
      if (self.isothermal and (not self.cr)):
        fig = plt.figure(figsize=(6, 3))
        grids = gs.GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        lab = ['$\\rho$', '$v$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, v_array - v_eq, label='t={:.3f}'.format(time))
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
      elif ((not self.isothermal) and self.cr):
        fig = plt.figure(figsize=(12, 3))
        grids = gs.GridSpec(1, 5, figure=fig)
        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[0, 1])
        ax3 = fig.add_subplot(grids[0, 2])
        ax4 = fig.add_subplot(grids[0, 3])
        ax5 = fig.add_subplot(grids[0, 4])
        lab = ['$\\rho$', '$v$', '$P_c$', '$P_g$', '$F_c$']
        ax1.plot(x1v, rho_array - rho_eq, label='t={:.3f}'.format(time))
        ax2.plot(x1v, v_array - v_eq, label='t={:.3f}'.format(time))
        ax3.plot(x1v, ecr_array/3. - pc_eq, label='t={:.3f}'.format(time))
        ax4.plot(x1v, pg_array - pg_eq, label='t={:.3f}'.format(time))
        ax5.plot(x1v, fc_array - fc_eq, label='t={:.3f}'.format(time))
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
            axes.set_yscale('log')
          elif (axes != ax5):
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
  def staircase(self, plot=False, xlim=False, time_series=False, fit=False, inset=False):
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
      dx = x1v[1] - x1v[0]
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
      vx = data['vel1'][0, 0, index_xl:index_xu]
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
      L_equi = np.abs(self.profile_object.pc_sol/self.profile_object.dpcdx_sol)
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
      display_file = file_array[np.argmin(np.abs(self.time_array - time))]
      display_file_pc = file_array[np.argmin(np.abs(self.time_array - time_pc))]
      data = ar.athdf('./' + self.filename + '.out1.' + format(display_file_pc, '05d') + '.athdf')
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

      # fig1 = plt.figure(figsize=(13, 6))
      fig1 = plt.figure()
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

      ax4.plot(x1v_display, pc_display, 'o-')
      for i, x in enumerate(x1v_display[index_xl:index_xu]): 
        if self.stair_start_loc[display_file_pc][i] == 1: 
          ax4.axvline(x1v_display[index_xl+i], linestyle='--', color='g')
        if self.stair_end_loc[display_file_pc][i] == 1:
          ax4.axvline(x1v_display[index_xl+i], linestyle='--', color='r')

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


      lab = ['Width', 'Plateau width', '$P_c$ Height']

      for i, axes in enumerate(ax):
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel(lab[i])
        axes.set_ylabel('Distribution')
        
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

  def convexhull(self, xlim=False, plot=False, xlog=False, ylog=False):
    filename = self.filename 

    # Analyse only a specific region
    if xlim:
      xl = float(input('Enter starting x for staircase analysis: '))
      xu = float(input('Enter ending x for staircase analysis: '))
      xv_start = float(input('Start recording the velocity here: '))
      xv_end = float(input('End recording the velocity here: '))

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
      if self.with_v:
        b0 = data['Bcc1'][0, 0, 0]
        r = data['rho'][0, 0, index_xl:index_xu]
        vel = data['vel1'][0, 0, index_xl:index_xu]
        pg = data['press'][0, 0, index_xl:index_xu]
        if xlim:
          index_xv_start = np.argmin(np.abs(x1v - xv_start))
          index_xv_end = np.argmin(np.abs(x1v - xv_end))
          r_xv = r[index_xv_start:index_xv_end]
          pg_xv = pg[index_xv_start:index_xv_end]
          speed_sound = np.sqrt(gg*np.mean(pg_xv)/np.mean(r_xv))
          vel = vel + speed_sound

        rho = 1./(vel + b0/np.sqrt(r))
        plt.plot(x1v, rho)
        plt.show()
      else:
        rho = data['rho'][0, 0, index_xl:index_xu]
      pc = data['Ec'][0, 0, index_xl:index_xu]*(gc - 1)
      rho_max = np.amax(rho)
      rho_min = np.amin(rho)
      pc_max = np.amax(pc)
      if np.argmax(rho) != 0:
        rho[0] = rho_max
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
      if self.with_v:
        fit_pc = lambda pc0: np.sum(pc0*(rho_stair[fp]/rho_max)**(gc) - pc[loc_stair])**2
      else:
        fit_pc = lambda pc0: np.sum(pc0*(rho_stair[fp]/rho_max)**(gc/2.) - pc[loc_stair])**2
      pc_fit = opt.minimize(fit_pc, pc_max)
      # pc_stair[fp] = pc_fit.x[0]*(rho_stair[fp]/rho_max)**(gc/2.) if not self.with_v else pc_fit.x[0]*(rho_stair[fp]/rho_max)**(gc)
      pc_stair[fp] = pc_max*(rho_stair[fp]/rho_max)**(gc/2.) if not self.with_v else pc_max*(rho_stair[fp]/rho_max)**(gc)
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

      lab = ['$\\rho$', '$P_c$'] if not self.with_v else ['$(v + v_A)^{-1}$', '$P_c$']

      for i, axes in enumerate(ax):
        if xlog:
          axes.set_xscale('log')
        if ylog:
          axes.set_yscale('log')
        axes.legend(frameon=False)
        axes.set_xlabel('$x$')
        axes.set_ylabel(lab[i])
        if not xlog:
          axes.xaxis.set_minor_locator(AutoMinorLocator())
        if not ylog:
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

  def time_avg(self, plot=False, logx=False, logy=False, compare=False):
    filename = self.filename
    file_array = self.file_array 
    num_file = np.size(file_array)

    x_avg = ar.athdf('./' + filename + '.out1.' + format(file_array[0], '05d') + '.athdf')['x1v']
    rho_avg = np.zeros(np.size(x_avg))
    v_avg = np.zeros(np.size(x_avg))
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
      invariant = np.zeros(np.size(x_avg))

    if compare:
      rho0 = np.zeros(np.size(x_avg))
      v0 = np.zeros(np.size(x_avg))
      if not self.isothermal:
        pg0 = np.zeros(np.size(x_avg))
      if self.cr:
        pc0 = np.zeros(np.size(x_avg))
        fc0 = np.zeros(np.size(x_avg))
        va0 = np.zeros(np.size(x_avg))

    for i, fp in enumerate(file_array):
      print(fp)
      data = ar.athdf('./' + filename + '.out1.' + format(fp, '05d') \
        + '.athdf')
      rho = data['rho'][0, 0, :]
      vx = data['vel1'][0, 0, :]
      if not self.isothermal:
        pg = data['press'][0, 0, :]
      if self.cr:
        ec = data['Ec'][0, 0, :] 
        fc = data['Fc1'][0, 0, :]*self.vmax
        va = np.abs(data['Vc1'][0, 0, :])
        ss = data['Sigma_adv1'][0, 0, :]/self.vmax
        sd = data['Sigma_diff1'][0, 0, :]/self.vmax
        sc = 1./(1./ss + 1./sd)
      rho_avg += rho/num_file
      v_avg += vx/num_file
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
        cr_heating_avg += -(vx + va)*sc*(fc - gc*vx*ec)/num_file
        invariant += (ec*(gc - 1.))*(vx + va)**(gc)/num_file
      if i == 0:
        rho0 = data['rho'][0, 0, :]
        v0 = data['vel1'][0, 0, :]
        if not self.isothermal:
          pg0 = data['press'][0, 0, :]
        if self.cr:
          pc0 = data['Ec'][0, 0, :]*(gc - 1.)
          fc0 = data['Fc1'][0, 0, :]*self.vmax
          va0 = np.abs(data['Vc1'][0, 0, :])

    # Save data
    self.x_avg = x_avg 
    self.rho_avg = rho_avg 
    self.v_avg = v_avg 
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
      self.invariant = invariant

    if plot:
      fig = plt.figure()
      grids = gs.GridSpec(2, 3, figure=fig) if not self.isothermal else gs.GridSpec(2, 2, figure=fig)
      ax1 = fig.add_subplot(grids[0, 0])
      ax2 = fig.add_subplot(grids[0, 1])
      ax3 = fig.add_subplot(grids[1, 0])
      ax4 = fig.add_subplot(grids[1, 1])
      if not self.isothermal:
        ax5 = fig.add_subplot(grids[1, 2])

      lab = ['$\\rho$', '$v$', '$P_g$', '$P_c$', '$F_c$'] if not self.isothermal else ['$\\rho$', '$v$', '$F_c$', '$P_c$']

      ax1.plot(x_avg, rho_avg)
      ax2.plot(x_avg, v_avg)
      if not self.isothermal:
        ax3.plot(x_avg, pg_avg)
      elif self.cr :
        ax3.plot(x_avg, fc_avg)
      if self.cr:
        ax4.plot(x_avg, pc_avg)
        if not self.isothermal:
          ax5.plot(x_avg, fc_avg)

      if compare:
        ax1.plot(x_avg, rho0, 'k--')
        ax2.plot(x_avg, v0, 'k--')
        if not self.isothermal:
          ax3.plot(x_avg, pg0, 'k--')
        elif self.cr:
          ax3.plot(x_avg, fc0, 'k--')
        if self.cr:
          ax4.plot(x_avg, pc0, 'k--')
          if not self.isothermal:
            ax5.plot(x_avg, fc0, 'k--')

      for i, axes in enumerate(fig.axes):
        if logx and (axes != ax2) and (axes != ax5):
          axes.set_xscale('log')
        else:
          axes.xaxis.set_minor_locator(AutoMinorLocator())
        if logy and (axes != ax2) and (axes != ax5):
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
inputfile = 'athinput.cr_power'
file_array = np.array([0, 5]) 

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

# Background for v
profile_in = dict()
profile_in['alpha'] = 1.
profile_in['beta'] = 1.
profile_in['eta'] = 0.01
profile_in['ms'] = 0.08
profile_in['psi'] = 0.5

profile_in['rho0'] = 1. 
profile_in['pg0'] = 1. 
profile_in['r0'] = 1.

# # Background for v_iso
# profile_in = dict()
# profile_in['alpha'] = 1.
# profile_in['beta'] = 1.
# profile_in['eta'] = 0.01
# profile_in['ms'] = 0.01
# profile_in['psi'] = 0.

# profile_in['rho0'] = 1. 
# profile_in['cs0'] = 1. 
# profile_in['r0'] = 1.

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

# one = Plot1d(inputfile, file_array, video=False, staircase=False, profile_in=None, with_v=False)
# if one.passive:
#   fig, fig_pass = one.plot(logx=False, logy=True)
#   fig.savefig('./1dplot.png', dpi=300)
#   fig_pass.savefig('./1dplot_passive.png', dpi=300)
# else:
#   fig = one.plot(logx=False, logy=True)
#   fig.savefig('./1dplot.png', dpi=300)
# plt.show()
# plt.close('all')

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
# ax1.plot(one.x1v, one.vx_array[0, :], 'k--')
# ax1.plot(one.x1v, one.vx_array[1, :])
# # ax1.plot(x, ma.masked_array(amp, np.abs(amp)>0.002), 'k--')
# # ax1.plot(x, -ma.masked_array(amp, np.abs(amp)>0.002), 'k--')
# ax1.margins(x=0)
# ax1.set_xlim(1.35, 1.65)
# ax1.set_ylim(-0.0042, 0.006)
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$v$')
# fig.tight_layout()
# # fig.savefig('./amp_compare.png', dpi=300)
# fig.savefig('./alpha1beta1eta_01psi0_gauss.png', dpi=300)
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
# file_end = 1000

# total_num_file = file_end - file_start 
# each = total_num_file//size

# video_array = np.arange(file_start + rank*each, file_start + (rank+1)*each)
# video_path = '/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/power/results/sims/output/video/'
# video = Plot1d(inputfile, video_array, video=True, staircase=False, profile_in=None, with_v=True)
# video.make_video(equi, video_path, logx=False, logy=True)

################################
# # Staricase identification
# # stair_array = np.array([311])
# stair_array = np.arange(1001)
# stair = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=profile_in, with_v=True)
# latexify(columns=2)
# stairfig_stat, stairfig_pc, stairfig_time, stairfig_avgpc, stairfig_avgv = stair.staircase(plot=True, xlim=False, time_series=True, fit=True, inset=False)
# # plotdefault()
# stairfig_stat.savefig('./staircase_stat.png', dpi=300)
# stairfig_pc.savefig('./staircase_pc.png', dpi=300)
# stairfig_time.savefig('./staircase_time.png', dpi=300) # Need to comment out if time_series=False
# stairfig_avgpc.savefig('./staircase_avgpc.png', dpi=300) # Need to comment out if time_series=False
# stairfig_avgv.savefig('./staircase_avgv.png', dpi=300) # Need to comment out if time_series=False

# plt.show()
# plt.close('all')

###############################
# Construct convex hull for density and reconstruct Pc
stair_array = np.array([766])
stair2 = Plot1d(inputfile, stair_array, video=False, staircase=True, profile_in=profile_in, with_v=True)
latexify(columns=1, square=True)
stair2fig_rho, stair2fig_pc = stair2.convexhull(xlim=True, plot=True, xlog=False, ylog=True)

stair2fig_rho.savefig('./rho_hull.png', dpi=300)
stair2fig_pc.savefig('./pc_hull.png', dpi=300)
plotdefault()

plt.show()
plt.close('all')

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
# avg_array = np.arange(0, 1000)
# # avg_array = np.array([700])
# avg = Plot1d(inputfile, avg_array, video=False, staircase=False, history=False, avg=True, profile_in=None, with_v=True)
# avg.time_avg(plot=False)
# avg_fig = avg.time_avg(plot=True, logx=False, logy=True, compare=True)
# avg_fig.savefig('./time_avg.png', dpi=300)

# plt.show()
# plt.close()

# # Parallel
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# avg_total_array = np.arange(0, 1000)
# each = np.size(avg_total_array)//size

# avg_array = avg_total_array[rank*each:(rank+1)*each]
# avg = Plot1d(inputfile, avg_array, video=False, staircase=False, history=False, avg=True, profile_in=None, with_v=True)
# avg.time_avg(plot=False)

# # Combine values
# avg_rho = np.zeros(np.size(avg.x_avg))
# avg_v = np.zeros(np.size(avg.x_avg))
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
#   invari = np.zeros(np.size(avg.x_avg))

# comm.Reduce(avg.rho_avg*np.size(avg_array)/np.size(avg_total_array), avg_rho, op=MPI.SUM, root=0)
# comm.Reduce(avg.v_avg*np.size(avg_array)/np.size(avg_total_array), avg_v, op=MPI.SUM, root=0)
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
#   comm.Reduce(avg.invariant*np.size(avg_array)/np.size(avg_total_array), invari, op=MPI.SUM, root=0)

# # # video_path = '/Users/tsunhinnavintsung/Workspace/Codes/workspace/1dcr_v2_1/cr_acous/wind/results/sims/output/video/'
# # # avg.fc_compare(video_path, logx=False, logy=True)

# if rank == 0:
#   avg.rho_avg = avg_rho 
#   avg.v_avg = avg_v
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
#     avg.invariant = invari

#   x_vpva = (avg.v_avg + avg.va_avg)**(-gc)
#   smooth_vpva = signal.savgol_filter(x_vpva, 31, 3)
#   smooth_pc = signal.savgol_filter(avg.pc_avg, 31, 3)
#   # dlogpcdcoup = np.gradient(np.log(avg.pc_avg), np.log(x_vpva))
#   # dlogpcdcoup_delete = np.where(dlogpcdcoup < 0)[0]
#   dlogpcdcoup = np.gradient(np.log(smooth_pc), np.log(smooth_vpva))
#   dlogpcdcoup_delete = np.where(dlogpcdcoup < 0)[0]
#   smooth_vpva = np.delete(smooth_vpva, dlogpcdcoup_delete)
#   dlogpcdcoup = np.delete(dlogpcdcoup, dlogpcdcoup_delete)
#   # dlogpcdcoup_smooth = signal.savgol_filter(dlogpcdcoup, 31, 3)
#   fit = np.polyfit(np.log(smooth_vpva), dlogpcdcoup, deg=0)

#   # x_rho = avg.rho_avg**(gc/2.)
#   # dlogpcdcoup = np.gradient(np.log(avg.pc_avg), np.log(x_rho))
#   # dlogpcdcoup_delete = np.where(dlogpcdcoup < 0)[0]
#   # x_rho = np.delete(x_rho, dlogpcdcoup_delete)
#   # dlogpcdcoup = np.delete(dlogpcdcoup, dlogpcdcoup_delete)
#   # dlogpcdcoup_smooth = signal.savgol_filter(dlogpcdcoup, 31, 3)
#   # fit = np.polyfit(np.log(x_rho), dlogpcdcoup, deg=0)

#   plotdefault()

#   fig = plt.figure()
#   ax = fig.add_subplot(111)

#   ax.loglog(smooth_vpva, dlogpcdcoup)
#   # ax.loglog(x_rho, dlogpcdcoup)
#   # ax.loglog(x_vpva, dlogpcdcoup_smooth, 'r')
#   ax.axhline(1.0, color='k', linestyle='--')
#   ax.axhline(fit[0], color='brown', linestyle=':')

#   ax.margins(x=0)
#   ax.set_xlabel('$(v + v_A)^{-\\gamma_c}$')
#   # ax.set_xlabel('$\\rho^{\\gamma_c/2}$')
#   ax.set_ylabel('$\\mathrm{dlog} P_c/\\mathrm{dlog} (v + v_A)^{-\\gamma_c}$')
#   # ax.set_ylabel('$\\mathrm{dlog} P_c/\\mathrm{dlog} \\rho^{\\gamma_c/2}$')

#   fig.tight_layout()
#   plt.show()

#   # Pc invariant
#   # fit_invar = np.polyfit(avg.x_avg, avg.invariant, deg=1)
#   # fit_invar2 = np.polyfit(avg.x_avg, avg.pc_avg*(avg.v_avg + avg.va_avg)**(gc), deg=1)
#   # fit_push = np.polyfit(np.log(-np.gradient(avg.pc_avg, avg.x_avg)), np.log(-avg.cr_pushing_avg), deg=1)
#   # fit_heat = np.polyfit(np.log(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg)), np.log(-avg.cr_heating_avg), deg=1)

#   fig1 = plt.figure()
#   fig2 = plt.figure()
#   fig3 = plt.figure()
#   ax1 = fig1.add_subplot(111)
#   ax2 = fig2.add_subplot(121)
#   ax3 = fig2.add_subplot(122)
#   ax4 = fig3.add_subplot(111)

#   ax1.plot(avg.x_avg, avg.invariant, 'b', label='$\\langle P_c (v + v_A)^{\\gamma_c}\\rangle$')
#   ax1.plot(avg.x_avg, avg.pc_avg*(avg.v_avg + avg.va_avg)**(gc), 'r-o', label='$\\langle P_c\\rangle (\\langle v\\rangle + \\langle v_A\\rangle)^{\\gamma_c}$')
#   # ax1.plot(avg.x_avg, fit_invar[0]*avg.x_avg + fit_invar[1], 'k--', label='slope=${:.3f}$'.format(fit_invar[0]))
#   # ax1.plot(avg.x_avg, fit_invar2[0]*avg.x_avg + fit_invar2[1], color='brown', linestyle='--', label='slope=${:.3f}$'.format(fit_invar2[0]))

#   # ax2.loglog(-np.gradient(avg.pc_avg, avg.x_avg), -avg.cr_pushing_avg)
#   # ax2.loglog(-np.gradient(avg.pc_avg, avg.x_avg), np.exp(fit_push[1])*(-np.gradient(avg.pc_avg, avg.x_avg))**(fit_push[0]), 'k--', label='slope=${:.3f}$'.format(fit_push[0]))
#   ax2.loglog(avg.x_avg, -np.gradient(avg.pc_avg, avg.x_avg), 'k--', label='$-\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax2.loglog(avg.x_avg, -avg.cr_pushing_avg, label='$\\langle\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   # ax3.loglog(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg), -avg.cr_heating_avg) 
#   # ax3.loglog(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg), np.exp(fit_heat[1])*(-(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg))**(fit_heat[0]), \
#     # 'k--', label='slope=${:.3f}$'.format(fit_heat[0]))
#   ax3.loglog(avg.x_avg, -(avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg), 'k--', label='$-(\\langle v\\rangle + \\langle v_A\\rangle)\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax3.loglog(avg.x_avg, -avg.cr_heating_avg, label='$\\langle(v + v_A)\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')

#   ax4.loglog(avg.x_avg, avg.cr_heating_avg/((avg.v_avg + avg.va_avg)*np.gradient(avg.pc_avg, avg.x_avg)))

#   ax1.margins(x=0)
#   ax1.set_xlabel('$x$')
#   ax1.legend(frameon=False)
#   # ax1.set_ylabel('$\\langle P_c\\rangle (\\langle v\\rangle + \\langle v_A\\rangle)^{\\gamma_c}$')

#   ax2.legend(frameon=False)
#   ax3.legend(frameon=False)
#   ax2.margins(x=0)
#   ax3.margins(x=0)
#   ax4.margins(x=0)
#   # ax2.set_xlabel('$-\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   # ax3.set_xlabel('$-(\\langle v\\rangle + \\langle v_A\\rangle)\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')
#   ax2.set_xlabel('$x$')
#   ax3.set_xlabel('$x$')
#   ax4.set_xlabel('$x$')
#   # ax2.set_ylabel('$\\langle\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   # ax3.set_ylabel('$\\langle(v + v_A)\\sigma_c (F_c - \\gamma_c E_c v)\\rangle$')
#   ax4.set_ylabel('$\\langle(v + v_A)\\sigma_c (F_c - \\gamma_c E_c v)\\rangle/(\\langle v\\rangle + \\langle v_A\\rangle)\\mathrm{d}\\langle P_c\\rangle/\\mathrm{d} x$')

#   fig1.tight_layout()
#   fig2.tight_layout()
#   fig3.tight_layout()
#   fig1.savefig('./invar.png', dpi=300)
#   fig2.savefig('./source.png', dpi=300)
#   fig3.savefig('./suppressed.png', dpi=300)

#   plt.show()

# #########################
# # Plots for publication

# # Nonlinear evolution of staircase
# latexify(columns=2)

# fig = plt.figure()
# grids = gs.GridSpec(2, 3, figure=fig)
# ax1 = fig.add_subplot(grids[0, 0])
# ax2 = fig.add_subplot(grids[0, 1])
# ax3 = fig.add_subplot(grids[0, 2])
# ax4 = fig.add_subplot(grids[1, 0])
# ax5 = fig.add_subplot(grids[1, 1])
# ax6 = fig.add_subplot(grids[1, 2])

# lab = ['$\\rho$', '$v$', '$P_g$', '$P_c$', '$F_c$', '$T$']

# ax1.plot(one.x1v, one.rho_array[0], 'k--', label='t=0.0')
# ax2.plot(one.x1v, one.vx_array[0], 'k--', label='t=0.0')
# ax3.plot(one.x1v, one.pg_array[0], 'k--', label='t=0.0')
# ax4.plot(one.x1v, one.ecr_array[0]*(gc - 1.), 'k--', label='t=0.0')
# ax5.plot(one.x1v, one.fcx_array[0]*one.vmax, 'k--', label='t=0.0')
# ax6.plot(one.x1v, one.pg_array[0]/one.rho_array[0], 'k--', label='t=0.0')

# temp_min = 0.01
# temp = one.pg_array[1]/one.rho_array[1]
# ind_del = np.where(temp < temp_min)[0]
# x1v = np.delete(one.x1v, ind_del)

# ax1.plot(x1v, np.delete(one.rho_array[1], ind_del), label='t={:.2f}'.format(one.time_array[1]))
# ax2.plot(x1v, np.delete(one.vx_array[1], ind_del), label='t={:.2f}'.format(one.time_array[1]))
# ax3.plot(x1v, np.delete(one.pg_array[1], ind_del), label='t={:.2f}'.format(one.time_array[1]))
# ax4.plot(x1v, np.delete(one.ecr_array[1]*(gc - 1.), ind_del), label='t={:.2f}'.format(one.time_array[1]))
# ax5.plot(x1v, np.delete(one.fcx_array[1]*one.vmax, ind_del), label='t={:.2f}'.format(one.time_array[1]))
# ax6.plot(x1v, np.delete(temp, ind_del), label='t={:.2f}'.format(one.time_array[1]))

# for i, axes in enumerate(fig.axes):
#   axes.margins(x=0)
#   axes.set_xlabel('$x$')
#   axes.set_ylabel(lab[i])
#   axes.legend(frameon=False)
#   if (axes != ax2) and (axes != ax5):
#     axes.set_yscale('log')
#     axes.xaxis.set_minor_locator(AutoMinorLocator())
#   else:
#     axes.yaxis.set_minor_locator(AutoMinorLocator())
#     axes.xaxis.set_minor_locator(AutoMinorLocator())

# fig.tight_layout()
# fig.savefig('/Users/tsunhinnavintsung/Desktop/Publish/alpha1beta1eta_01psi0_gauss.png', dpi=300)

# plt.show()
# plt.close('all')

# plotdefault()

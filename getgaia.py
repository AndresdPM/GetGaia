#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from sklearn import mixture
from astropy import units as u, coordinates as c
from scipy import stats
from math import log10, floor

import random
from astropy.stats import sigma_clip, mad_std

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

from zero_point import zpt

def add_inner_title(ax, title, loc, size=None, color=None, rotation=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    if size is None:
        prop = dict(size=plt.rcParams['legend.fontsize'])
    else:
        prop = size
    if color is not None:
        prop['color'] = color
    if rotation is not None:
       prop['rotation'] = rotation
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
   """
   Correct the G-band fluxes and magnitudes for the input list of Gaia EDR3 data.
   
   Parameters
   ----------
   
   bp_rp: float, array_like
       The (BP-RP) colour listed in the Gaia EDR3 archive.
   astrometric_params_solved: int, array_like
       The astrometric solution type listed in the Gaia EDR3 archive.
   phot_g_mean_mag: float, array_like
       The G-band magnitude as listed in the Gaia EDR3 archive.
   phot_g_mean_flux: float, array_like
       The G-band flux as listed in the Gaia EDR3 archive.
       
   Returns
   -------
   
   The corrected G-band magnitudes and fluxes. The corrections are only applied to
   sources with a 6-parameter astrometric solution fainter than G=13, for which a
   (BP-RP) colour is available.
   
   Example
   
   gmag_corr, gflux_corr = correct_gband(bp_rp, astrometric_params_solved, 
                                         phot_g_mean_mag, phot_g_mean_flux)
   """
   if np.isscalar(bp_rp) or np.isscalar(astrometric_params_solved) or \
                   np.isscalar(phot_g_mean_mag) or np.isscalar(phot_g_mean_flux):
       bp_rp = np.float64(bp_rp)
       astrometric_params_solved = np.int64(astrometric_params_solved)
       phot_g_mean_mag = np.float64(phot_g_mean_mag)
       phot_g_mean_flux = np.float64(phot_g_mean_flux)
   
   if not (bp_rp.shape == astrometric_params_solved.shape \
                       == phot_g_mean_mag.shape == phot_g_mean_flux.shape):
       raise ValueError('Function parameters must be of the same shape!')
   
   do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<=13) | \
                                      (astrometric_params_solved != 95)
   bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>13) & \
                                  (phot_g_mean_mag<=16)
   faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
   bp_rp_c = np.clip(bp_rp, 0.25, 3.0)
   
   correction_factor = np.ones_like(phot_g_mean_mag)
   correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + 0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
   correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + 0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)
   
   gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)
   gflux_corrected = phot_g_mean_flux * correction_factor
   
   return gmag_corrected, gflux_corrected
 

def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
   """
   Calculate the corrected flux excess factor for the input Gaia EDR3 data.
   
   Parameters
   ----------
   
   bp_rp: float, array_like
       The (BP-RP) colour listed in the Gaia EDR3 archive.
   phot_bp_rp_flux_excess_factor: float, array_like
       The flux excess factor listed in the Gaia EDR3 archive.
       
   Returns
   -------
   
   The corrected value for the flux excess factor, which is zero for "normal" stars.
   
   Example
   -------
   
   phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, 
                                               phot_bp_rp_flux_excess_factor)
   """

   if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
       bp_rp = np.float64(bp_rp)
       phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
   
   if bp_rp.shape != phot_bp_rp_excess_factor.shape:
       raise ValueError('Function parameters must be of the same shape!')
   
   do_not_correct = np.isnan(bp_rp)
   bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
   greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
   redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)
   
   correction = np.zeros_like(bp_rp)
   correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange],2)
   correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange],2) - 0.005879*np.power(bp_rp[greenrange],3)
   correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
   
   return phot_bp_rp_excess_factor - correction


def statistics(table):
   """
   Weighted statistics, its error and the standard deviation.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   
   x_i = table.loc[:, var_cols]
   ex_i = get_errors(table, used_cols = var_cols)
   ex_i.columns = ex_i.columns.str.rstrip('_error')

   weighted_variance = (1./(1./ex_i**2).sum(axis = 0))
   weighted_avg = ((x_i.div(ex_i.values**2)).sum(axis = 0) * weighted_variance.values).add_suffix('_wmean')
   weighted_avg_error = np.sqrt(weighted_variance[~weighted_variance.index.duplicated()]).add_suffix('_wmean_error')

   avg = x_i.mean().add_suffix('_mean')
   avg_error = x_i.std().add_suffix('_mean_error')/np.sqrt(len(x_i))
   std = x_i.std().add_suffix('_std')

   #The formula is 1.25*sigma/sqrt(N) for medians of normal distributions. It gets better from there. This is the upper limmit of error.
   median_err = (1.25404*table.std()/np.sqrt(table.count())).add_suffix('_median_error')
   median = table.median().add_suffix('_median')

   return pd.concat([weighted_avg, weighted_avg_error, avg, avg_error, median, median_err, std])


def wcs2xy(ra, dec, ra0, dec0):
   """
   Compute x, y in degrees centred at the object
   """

   ra_rad = ra *np.pi/180.
   dec_rad = dec *np.pi/180.
   ra0_rad = ra0 *np.pi/180.
   dec0_rad = dec0 *np.pi/180.
   
   sin_ra_ra0 = np.sin(ra_rad-ra0_rad)
   cos_ra_ra0 = np.cos(ra_rad-ra0_rad)
   sin_dec = np.sin(dec_rad)
   cos_dec = np.cos(dec_rad)
   sin_dec0 = np.sin(dec0_rad)
   cos_dec0 = np.cos(dec0_rad)

   x = -cos_dec * sin_ra_ra0
   y =  sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_ra_ra0

   return x*180./np.pi, y*180./np.pi


def _gauss2d_mge(n, xc, yc, sx, sy, pos_ang):
   """
   2D Gaussian image with size N[0]xN[1], center (XC,YC),
   sigma (SX,SY) along the principal axes and position angle POS_ANG, measured
   from the positive Y axis to the Gaussian major axis (positive counter-clockwise).
   """

   ang = np.radians(pos_ang - 90.)
   x, y = np.ogrid[-xc:n[0] - xc, -yc:n[1] - yc]

   xcosang = np.cos(ang)/(np.sqrt(2.)*sx)*x
   ysinang = np.sin(ang)/(np.sqrt(2.)*sx)*y
   xsinang = np.sin(ang)/(np.sqrt(2.)*sy)*x
   ycosang = np.cos(ang)/(np.sqrt(2.)*sy)*y

   im = (xcosang + ysinang)**2 + (ycosang - xsinang)**2

   return np.exp(-im)


def _multi_gauss(pars, img, sigmaPSF, normPSF, xpeak, ypeak, theta):
   """
   Multi Gaussian expansion. Eq.(4,5) in Cappellari (2002)
   """

   sigmaPSF = np.atleast_1d(sigmaPSF)
   normPSF = np.atleast_1d(normPSF)

   lum, sigma, q = pars

   u = 0.
   for lumj, sigj, qj in zip(lum, sigma, q):
      for sigP, normP in zip(sigmaPSF, normPSF):
          sx = np.sqrt(sigj**2 + sigP**2)
          sy = np.sqrt((sigj*qj)**2 + sigP**2)
          g = _gauss2d_mge(img.shape, xpeak, ypeak, sx, sy, theta)
          u += lumj*normP/(2.*np.pi*sx*sy) * g

   return u


def sky_model(args, data, field = None, probabilities = None,  fit_pge = True, use_center = True, ellipse_plot = 'ellipse_plot.pdf', profiles_plot = 'profiles_plot.pdf'):
   from astropy.convolution import Gaussian2DKernel
   from astropy.convolution import convolve
   from mgefit.find_galaxy import find_galaxy
   from mgefit.sectors_photometry import sectors_photometry
   from mgefit.mge_fit_sectors import mge_fit_sectors

   entire_field = pd.concat([data, field])
   resolution = int(np.ptp(entire_field.x)*3600/20)
   bins = [np.linspace(entire_field.x.min(), entire_field.x.max(), resolution+1), np.linspace(entire_field.y.min(), entire_field.y.max(), resolution+1)]
   sigmapsf = resolution/50.
   normpsf = 1.
   radius_mask = args.field_radius*3600
   ngauss = 1000 # Number of gaussians for the fitting.

   try:
      entire_field = entire_field[entire_field.member]
      minlevel = 0.
   except:
      minlevel = len(field)/(args.field_area*3600**2)

   img, binx, biny = np.histogram2d(entire_field.x, entire_field.y, bins=bins)

   # Calculate center and position of the stars in pixels
   center = np.array(img.shape)/2

   # Deffine the pixel scale, min level, PSF, and number of Gaussians
   scale = (binx[1]-binx[0])*3600  # arcsec/pixel

   # Calculate the distance in pixels to the center of the image
   x, y = np.ogrid[-center[1]:img.shape[0] - center[1], -center[0]:img.shape[1] - center[0]]   # note yc before xc
   r = np.sqrt(x**2 + y**2)

   # Filter the image using NaN outside the observed area
   img[r > radius_mask/scale] = np.nan
   img = convolve(img, Gaussian2DKernel(x_stddev= sigmapsf))

   f = find_galaxy(img.T, fraction=0.02, binning=1, quiet = 1, plot = 1)
   plt.savefig(ellipse_plot, bbox_inches='tight')
   plt.close('all')

   # We mask pixels outside the obserd region
   mask = r < radius_mask/scale

   if use_center:
      center_x = center[0]
      center_y = center[1]
   else:
      center_x = f.xmed
      center_y = f.ymed

   # Create a Series with the properties of the ellipse:
   ellipse = pd.Series(data={'used_x':(center_x - center[0]) * scale /3600., 'used_y':(center_y - center[1]) * scale /3600., 'x':(f.xmed - center[0]) * scale /3600., 'y':(f.ymed - center[1]) * scale /3600., 'pa':f.theta, 'eps':f.eps, 'majoraxis':f.majoraxis * scale /3600.})
   
   if fit_pge:
      # Perform galaxy photometry
      s = sectors_photometry(img, f.eps, f.theta, center_x, center_y, minlevel=minlevel, mask=mask, plot=0)
      #plt.pause(1)  # Allow plot to appear on the screen

      plt.clf()
      m = mge_fit_sectors(s.radius, s.angle, s.counts, f.eps, ngauss=ngauss, sigmapsf=sigmapsf, scale=scale, bulge_disk=0, linear=1, quiet = 1, plot = 1)
      plt.savefig(profiles_plot, bbox_inches='tight')
      plt.close('all')

      solution = m.sol.T[m.sol[0]!=0].T
      model_sample = _multi_gauss(solution, img, sigmapsf, normpsf, center_x, center_y, f.theta)
      model_object = _multi_gauss(solution[:, :-1], img, sigmapsf, normpsf, center_x, center_y, f.theta)
      model_field = _multi_gauss(solution[:, -1:], img, sigmapsf, normpsf, center_x, center_y, f.theta)

      model_field = np.ones_like(img)*minlevel

      # Compute the positions of data stars within the bins
      posx = np.digitize(data.x, binx)
      posy = np.digitize(data.y, biny)

      # Select points within the histogram
      ind = (posx > 0) & (posx <= resolution) & (posy > 0) & (posy <= resolution)

      probabilities.loc[ind, "n_sample_sky"] = model_object[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
      probabilities.loc[ind, "n_object_sky"] = model_object[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
      probabilities.loc[ind, "n_field_sky"] = model_field[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
      probabilities.loc[ind, "n_minlevel_sky"] = minlevel

      return ellipse, probabilities
   else:
      return ellipse


def CalculateProbabilities(args, data, field, systematics = None, probs_filename = 'None', ellipse_filename = 'None'):
   """
   Compute maximum Likelihood based on sky position, cmd, PMs and parallax.
   """

   from mgefit.find_galaxy import find_galaxy
   from mgefit.sectors_photometry import sectors_photometry
   from mgefit.mge_fit_sectors import mge_fit_sectors
   from astropy.convolution import Gaussian2DKernel
   from astropy.convolution import convolve

   probabilities = pd.DataFrame(index = data.index)

   probabilities['source_id'] = data['source_id']

   if args.use_position_prob:
      print('Computing sky distribution...')
      try:
         ellipse, probabilities = sky_model(args, data, field = field, probabilities = probabilities, fit_pge = True, use_center = True, ellipse_plot = './%s/%s_sky_ellipse_1.pdf'%(args.name, args.name), profiles_plot = './%s/%s_mpg_profiles.pdf'%(args.name, args.name))
         ellipse.to_frame().T.to_csv(ellipse_filename, index = False)
         probabilities.to_csv(probs_filename, index = False)
      except:
         print('Failure to get sky distribution. Falling to CMD and PM')
         args.use_position_prob = False
         ellipse = []
   else:
      ellipse = []

   if systematics is not None:
      for col in systematics.columns:
         data.loc[:, col] = np.sqrt(data.loc[:, col]**2 + systematics.loc[:, col].values**2)
         field.loc[:, col] = np.sqrt(field.loc[:, col]**2 + systematics.loc[:, col].values**2)

   # calculate ratio of field sample area to data sample area
   area_diff = args.field_area/args.area

   print('Computing CMD distribution...')

   probabilities["n_object_cmd"] = [np.sum(stats.norm.pdf(data.loc[data.index != idx, "gmag"], data.loc[data.index == idx, "gmag"], data.loc[data.index != idx, "gmag_error"])\
                                 * stats.norm.pdf(data.loc[data.index != idx, "bp_rp"], data.loc[data.index == idx, "bp_rp"], data.loc[data.index != idx, "bp_rp_error"])) for idx in data.index]

   probabilities["n_field_cmd"] = [np.sum(stats.norm.pdf(field.loc[:, "gmag"], data.loc[data.index == idx, "gmag"], field.loc[:, "gmag_error"])\
                                 * stats.norm.pdf(field.loc[:, "bp_rp"], data.loc[data.index == idx, "bp_rp"], field.loc[:, "bp_rp_error"])) / area_diff for idx in data.index]

   print('Computing PMs and parallax distribution...')

   if args.use_parallax_prob:

      probabilities["n_object_pmp"] = [np.sum(stats.norm.pdf(data.loc[data.index != idx, "pmra"], data.loc[data.index == idx, "pmra"], data.loc[data.index != idx, "pmra_error"])\
                                    * stats.norm.pdf(data.loc[data.index != idx, "pmdec"], data.loc[data.index == idx, "pmdec"], data.loc[data.index != idx, "pmdec_error"])\
                                    * stats.norm.pdf(data.loc[data.index != idx, "parallax"], data.loc[data.index == idx, "parallax"], data.loc[data.index != idx, "parallax_error"])) for idx in data.index]

      probabilities["n_field_pmp"] = [np.sum(stats.norm.pdf(field.loc[:, "pmra"], data.loc[data.index == idx, "pmra"], field.loc[:, "pmra_error"])\
                                    * stats.norm.pdf(field.loc[:, "pmdec"], data.loc[data.index == idx, "pmdec"], field.loc[:, "pmdec_error"])\
                                    * stats.norm.pdf(field.loc[:, "parallax"], data.loc[data.index == idx, "parallax"], field.loc[:, "parallax_error"])) / area_diff for idx in data.index]

   else:

      probabilities["n_object_pmp"] = [np.sum(stats.norm.pdf(data.loc[data.index != idx, "pmra"], data.loc[data.index == idx, "pmra"], data.loc[data.index != idx, "pmra_error"])\
                                    * stats.norm.pdf(data.loc[data.index != idx, "pmdec"], data.loc[data.index == idx, "pmdec"], data.loc[data.index != idx, "pmdec_error"])) for idx in data.index]

      probabilities["n_field_pmp"] = [np.sum(stats.norm.pdf(field.loc[:, "pmra"], data.loc[data.index == idx, "pmra"], field.loc[:, "pmra_error"])\
                                    * stats.norm.pdf(field.loc[:, "pmdec"], data.loc[data.index == idx, "pmdec"], field.loc[:, "pmdec_error"])) / area_diff for idx in data.index]



   # number of cluster stars nearby in CMD and PMs+parallaxes
   probabilities["n_object_cmd"] = np.maximum(0, probabilities["n_object_cmd"] - probabilities["n_field_cmd"])
   probabilities["n_object_pmp"] = np.maximum(0, probabilities["n_object_pmp"] - probabilities["n_field_pmp"])
   
   if args.use_position_prob:
      # overall probability of being a member star
      total = probabilities["n_object_sky"] * probabilities["n_object_cmd"] * probabilities["n_object_pmp"]\
            + probabilities["n_object_sky"] * probabilities["n_field_cmd"]  * probabilities["n_object_pmp"]\
            + probabilities["n_object_sky"] * probabilities["n_object_cmd"] * probabilities["n_field_pmp"]\
            + probabilities["n_object_sky"] * probabilities["n_field_cmd"]  * probabilities["n_field_pmp"]\
            + probabilities["n_field_sky"]  * probabilities["n_object_cmd"] * probabilities["n_object_pmp"]\
            + probabilities["n_field_sky"]  * probabilities["n_field_cmd"]  * probabilities["n_object_pmp"]\
            + probabilities["n_field_sky"]  * probabilities["n_object_cmd"] * probabilities["n_field_pmp"]\
            + probabilities["n_field_sky"]  * probabilities["n_field_cmd"]  * probabilities["n_field_pmp"]

      probabilities["membership_prob"] = probabilities["n_object_sky"]*probabilities["n_object_cmd"]*probabilities["n_object_pmp"]/total
   else:
      # overall probability of being a member star
      total = probabilities["n_object_cmd"] * probabilities["n_object_pmp"]\
            + probabilities["n_field_cmd"]  * probabilities["n_object_pmp"]\
            + probabilities["n_object_cmd"] * probabilities["n_field_pmp"]\
            + probabilities["n_field_cmd"]  * probabilities["n_field_pmp"]\

      probabilities["membership_prob"] = probabilities["n_object_cmd"]*probabilities["n_object_pmp"]/total
   
   probabilities.to_csv(probs_filename, index = False)

   return ellipse, probabilities



def manual_select_from_cmd(args, data, ellipse = None):
   """
   Select stars based on their membership probabilities and cmd position
   """

   from matplotlib.widgets import Slider, Button, RadioButtons
   from matplotlib.path import Path
   from matplotlib.widgets import LassoSelector
   from matplotlib import path, patches

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.2):
         self.canvas = ax.figure.canvas
         self.collection = collection
         self.alpha_other = alpha_other

         self.xys = collection.get_offsets()
         self.Npts = len(self.xys)

         # Ensure that we have separate colors for each object
         self.fc = collection.get_facecolors()
         if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
         elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

         lineprops = {'color': 'k', 'linewidth': 1, 'alpha': 0.8}
         self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops)
         self.ind = []

      def onselect(self, verts):
         path = Path(verts)
         self.ind = np.nonzero(path.contains_points(self.xys))[0]
         self.selection = path.contains_points(self.xys)
         self.fc[:, -1] = self.alpha_other
         self.fc[self.ind, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

      def disconnect(self):
         self.lasso.disconnect_events()
         self.fc[:, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

   data_color = data.loc[:, 'bp_rp']
   data_mag = data.loc[:, 'gmag']
   data_prob = data.loc[:, 'membership_prob']
   data_x = data.loc[:, 'x']
   data_y = data.loc[:, 'y']
   data_pmra = data.loc[:, 'pmra']
   data_pmdec = data.loc[:, 'pmdec']

   if args.membership_prob is not None:
      members = data_prob >= args.membership_prob
   elif args.silent == True:
      members = data_prob >= 0.5
   else:
      try:
         #Plot ellipse
         ellipse_patch1 = patches.Ellipse(xy=(0, 0), width=2*ellipse.majoraxis, fill=False,
                                 height=2*ellipse.majoraxis*(1-ellipse.eps), angle=-ellipse.pa,
                                 color='k', linewidth=1)

         ellipse_patch2 = patches.Ellipse(xy=(0, 0), width=2*ellipse.majoraxis, fill=False,
                                 height=2*ellipse.majoraxis*(1-ellipse.eps), angle=-ellipse.pa,
                                 color='k', linewidth=1)

      except:
         pass

      if args.inside_core:
         sky_sel = ellipse1.contains_points(np.array([data_x.values, data_y.values]).T, radius=0)
      else:
         sky_sel = [True] * len(data)

      members = (data_prob >= 0.5) & sky_sel

      plt.close('all')
      subplot_kw = dict(autoscale_on = False)
      fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (11., 7.5), subplot_kw = subplot_kw)
      plt.subplots_adjust(right=0.85, wspace = 0.35, hspace = 0.1, bottom=0.15)
      axcolor = 'lightgoldenrodyellow'
      ax_prob = plt.axes([0.9, 0.15, 0.02, 0.72], facecolor=axcolor)

      sel_cmd = ax1.scatter(data_color.loc[members], data_mag.loc[members], c = '0.2', s=2, zorder = 1, alpha = 0.75)
      sel_pm = ax2.scatter(data_pmra.loc[members], data_pmdec.loc[members], c = '0.2', s=2, zorder = 1, alpha = 0.75)
      sel_sky = ax3.scatter(data_x.loc[members], data_y.loc[members], c = '0.2', s=2, zorder = 1, alpha = 0.75)

      rej_cmd = ax4.scatter(data_color.loc[~members], data_mag.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.75)
      rej_pm = ax5.scatter(data_pmra.loc[~members], data_pmdec.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.75)
      rej_sky = ax6.scatter(data_x.loc[~members], data_y.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.75)
      
      try:
         ax2.plot(args.pmra, args.pmdec, 'kx', ms = 3, zorder = 2)
         ax5.plot(args.pmra, args.pmdec, 'kx', ms = 3, zorder = 2)
      except:
         pass
      
      try:
         #Plot ellipse
         ax3.add_artist(ellipse_patch1)
         ax6.add_artist(ellipse_patch2)
      except:
         pass

      #Gal = ax1.annotate('%s'%args.name, xy=(np.nanmin(data_color)+0.1, np.nanmin(data_mag)+0.3), xycoords='data', color = 'k')
      #nstars = ax1.annotate('%i stars'%members.sum(), xy=(np.nanmax(data_color)-0.1, np.nanmin(data_mag)+0.3), xycoords='data', color = 'k')

      nstars = ax1.annotate('%i stars'%members.sum(), xy=(0.6, 0.93), xycoords='axes fraction', color = 'k')
      name = add_inner_title(ax1, '%s'%args.name, loc=2)


      ax1.set_xlim(-0.5, 4.0)
      ax1.set_ylim(20.5, 10)
      ax1.set_ylabel(r'$G$ [Mag]')
      ax1.axes.xaxis.set_ticklabels([])
      ax1.grid()

      ax4.set_xlim(-0.5, 4.0)
      ax4.set_ylim(20.5, 10)
      ax4.set_xlabel(r'$(G_{BP}-G_{RP})$ [Mag]')
      ax4.set_ylabel(r'$G$ [Mag]')
      ax4.grid()

      ax2.set_xlim(args.min_pmra, args.max_pmra)
      ax2.set_ylim(args.min_pmdec, args.max_pmdec)
      ax2.set_ylabel(r'$\mu_{\delta}{\rm{ [mas \ yr^{-1}}]}$')
      ax2.axes.xaxis.set_ticklabels([])
      ax2.grid()

      ax5.set_xlim(args.min_pmra, args.max_pmra)
      ax5.set_ylim(args.min_pmdec, args.max_pmdec)
      ax5.set_xlabel(r'$\mu_{\alpha\star}{\rm{ [mas \ yr^{-1}}]}$ ')
      ax5.set_ylabel(r'$\mu_{\delta}{\rm{ [mas \ yr^{-1}}]}$')
      ax5.grid()

      ax3.set_xlim(-args.max_search_radius, args.max_search_radius)
      ax3.set_ylim(-args.max_search_radius, args.max_search_radius)
      ax3.set_ylabel(r'$y {\rm{ [^\circ]}$')
      ax3.axes.xaxis.set_ticklabels([])
      ax3.grid()

      ax6.set_xlim(-args.max_search_radius, args.max_search_radius)
      ax6.set_ylim(-args.max_search_radius, args.max_search_radius)
      ax6.set_xlabel(r'$x {\rm{ [^\circ]}$')
      ax6.set_ylabel(r'$y {\rm{ [^\circ]}$')
      ax6.grid()

      s_prob = Slider(ax_prob, 'Prob', 0., 1.0, valinit=0.5, orientation='vertical')

      def update(val):
         members = (data_prob >= s_prob.val)  & sky_sel
         
         nstars.set_text('%i stars'%members.sum())
         
         sel_cmd.set_offsets(np.array([data_color.loc[members],  data_mag.loc[members]]).T)
         rej_cmd.set_offsets(np.array([data_color.loc[~members], data_mag.loc[~members]]).T)

         sel_sky.set_offsets(np.array([data_x.loc[members],  data_y.loc[members]]).T)
         rej_sky.set_offsets(np.array([data_x.loc[~members], data_y.loc[~members]]).T)

         sel_pm.set_offsets(np.array([data_pmra.loc[members],  data_pmdec.loc[members]]).T)
         rej_pm.set_offsets(np.array([data_pmra.loc[~members], data_pmdec.loc[~members]]).T)

         fig.canvas.draw_idle()

      s_prob.on_changed(update)

      def accept(event):
         if event.key == "enter":
            plt.close('all')

      fig.canvas.mpl_connect("key_press_event", accept)
      fig.suptitle("Use the slider to select probability and press enter to accept.")

      plt.show()

      members = (data_prob >= s_prob.val) & sky_sel

   if args.silent == False:
      use_lasso = str2bool(input("Would you like to further refine your CMD selection? "))
      
      if use_lasso:
         subplot_kw = dict(autoscale_on = False)
         fig, ax = plt.subplots(1, 1, figsize = (4, 4.5), subplot_kw = subplot_kw)

         pts = ax.scatter(data_color[members], data_mag[members], s=2, fc = 'red', zorder = 1)
         ax.scatter(data_color[~members], data_mag[~members], c = '0.2', s=1, zorder = 0, alpha = 0.25)

         ax.set_xlim(np.nanmin(data_color)-0.2, np.nanmax(data_color)+0.2)
         ax.set_ylim(np.nanmax(data_mag)+0.3, np.nanmin(data_mag)-0.3)
         ax.set_xlabel(data_color.name)
         ax.set_ylabel(data_mag.name)
         ax.grid()

         selector = SelectFromCollection(ax, pts)

         def accept(event):
            if event.key == "enter":
               selector.disconnect()
               plt.close('all')

         fig.canvas.mpl_connect("key_press_event", accept)
         fig.suptitle("Use your cursor to select likely member stars and press enter to accept.")

         plt.show()
         input("Once your selection is made, please press enter to continue.")

         members.loc[members == True] = selector.selection

   plt.close('all')
   subplot_kw = dict(autoscale_on = False)
   fig, (ax1, ax2) = plt.subplots(1,2, figsize = (7.5, 3.5), subplot_kw = subplot_kw)
   plt.subplots_adjust(wspace = 0.3, bottom=0.15)

   sel_cmd = ax1.scatter(data_color.loc[members], data_mag.loc[members], c = 'red', s=2, zorder = 1, alpha = 0.95)
   rej_cmd = ax1.scatter(data_color.loc[~members], data_mag.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.25)

   sel_sky = ax2.scatter(data_x.loc[members], data_y.loc[members], c = 'red', s=2, zorder = 1, alpha = 0.95)
   rej_sky = ax2.scatter(data_x.loc[~members], data_y.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.25)
   
   try:
      #Plot ellipse
      ellipse = patches.Ellipse(xy=(0, 0), width=2*ellipse.majoraxis, fill=False,
                              height=2*ellipse.majoraxis*(1-ellipse.eps), angle=-ellipse.pa,
                              color='k', linewidth=1.5)
      ax2.add_artist(ellipse)
   except:
      pass
   

   ax1.set_xlim(np.nanmin(data_color)-0.1, np.nanmax(data_color)+0.1)
   ax1.set_ylim(np.nanmax(data_mag)+0.05, np.nanmin(data_mag)-0.05)
   ax1.set_xlabel(data_color.name)
   ax1.set_ylabel(data_mag.name)
   ax1.grid()

   ax2.set_xlim(np.nanmin(data_x), np.nanmax(data_x))
   ax2.set_ylim(np.nanmax(data_y), np.nanmin(data_y))
   ax2.set_xlabel(data_x.name)
   ax2.set_ylabel(data_y.name)
   ax2.grid()
   
   plt.savefig('./%s/%s_selection.pdf'%(args.name, args.name), bbox_inches='tight')
   plt.close('all')

   try:
      return members
   except:
      return [True]*len(mag)


def plot_final_selection(args, data, ellipse = None, plot_prob = True):
   from matplotlib.path import Path
   from matplotlib import path, patches

   data_color = data.loc[:, 'bp_rp']
   data_mag = data.loc[:, 'gmag']
   data_x = data.loc[:, 'x']
   data_y = data.loc[:, 'y']
   data_pmra = data.loc[:, 'pmra']
   data_pmdec = data.loc[:, 'pmdec']

   members = data.loc[:, 'member']

   try:
      #Plot ellipse
      ellipse_patch1 = patches.Ellipse(xy=(0, 0), width=2*ellipse.majoraxis, fill=False,
                              height=2*ellipse.majoraxis*(1-ellipse.eps), angle=-ellipse.pa,
                              color='k', linewidth=1)

      ellipse_patch2 = patches.Ellipse(xy=(0, 0), width=2*ellipse.majoraxis, fill=False,
                              height=2*ellipse.majoraxis*(1-ellipse.eps), angle=-ellipse.pa,
                              color='k', linewidth=1)

   except:
      pass

   if args.inside_core:
      sky_sel = ellipse_patch1.contains_points(np.array([data_x.values, data_y.values]).T, radius=0)
   else:
      sky_sel = [True] * len(data)
   
   if plot_prob:
      data_prob = data.loc[:, 'membership_prob']
      color = data_prob.loc[members == True]
   else:
      color = '0.2'
   
   plt.close('all')
   subplot_kw = dict(autoscale_on = False)
   fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (12.0, 7.), subplot_kw = subplot_kw)
   plt.subplots_adjust(right=0.85, wspace = 0.35, hspace = 0.1, bottom=0.15)

   #nstars = ax1.annotate('%i stars'%members.sum(), xy=(np.nanmax(data_color)-1., np.nanmin(data_mag)+0.7), xycoords='data', color = 'k')
   nstars = add_inner_title(ax1, '%i stars'%members.sum(), loc=2)
   name = add_inner_title(ax1, '%s'%args.name, loc=1)

   sel_cmd = ax1.scatter(data_color.loc[members], data_mag.loc[members], c = color, s=2, zorder = 1, alpha = 0.75)
   sel_pm = ax2.scatter(data_pmra.loc[members], data_pmdec.loc[members], c = color, s=2, zorder = 1, alpha = 0.75)
   sel_sky = ax3.scatter(data_x.loc[members], data_y.loc[members], c = color, s=2, zorder = 1, alpha = 0.75)

   rej_cmd = ax4.scatter(data_color.loc[~members], data_mag.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.75)
   rej_pm = ax5.scatter(data_pmra.loc[~members], data_pmdec.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.75)
   rej_sky = ax6.scatter(data_x.loc[~members], data_y.loc[~members], c = '0.2', s=1, zorder = 0, alpha = 0.75)
   
   try:
      ax2.errorbar(args.pmra, args.pmdec, xerr = args.pmra, yerr= args.pmdec, fmt='ks', ms = 2, ecolor='k', label = 'Mc20', zorder = 2)
      ax5.errorbar(args.pmra, args.pmdec, xerr = args.pmra, yerr= args.pmdec, fmt='ks', ms = 2, ecolor='k', label = 'Mc20', zorder = 2)
   except:
      pass
   
   try:
      #Plot ellipse
      ax3.add_artist(ellipse_patch1)
      ax6.add_artist(ellipse_patch2)
   except:
      pass

   ax1.set_xlim(-0.5, 4.0)
   ax1.set_ylim(20.5, 10)
   ax1.set_ylabel(r'$G$ [Mag]')
   ax1.axes.xaxis.set_ticklabels([])
   ax1.grid()

   ax4.set_xlim(-0.5, 4.0)
   ax4.set_ylim(20.5, 10)
   ax4.set_xlabel(r'$(G_{BP}-G_{RP})$ [Mag]')
   ax4.set_ylabel(r'$G$ [Mag]')
   ax4.grid()

   ax2.set_xlim(args.min_pmra, args.max_pmra)
   ax2.set_ylim(args.min_pmdec, args.max_pmdec)
   ax2.set_ylabel(r'$\mu_{\delta}{\rm{ [mas \ yr^{-1}}]}$')
   ax2.axes.xaxis.set_ticklabels([])
   ax2.grid()

   ax5.set_xlim(args.min_pmra, args.max_pmra)
   ax5.set_ylim(args.min_pmdec, args.max_pmdec)
   ax5.set_xlabel(r'$\mu_{\alpha\star}{\rm{ [mas \ yr^{-1}}]}$ ')
   ax5.set_ylabel(r'$\mu_{\delta}{\rm{ [mas \ yr^{-1}}]}$')
   ax5.grid()

   ax3.set_xlim(-args.max_search_radius, args.max_search_radius)
   ax3.set_ylim(-args.max_search_radius, args.max_search_radius)
   ax3.set_ylabel(r'$y {\rm{ [^\circ]}$')
   ax3.axes.xaxis.set_ticklabels([])
   ax3.grid()

   ax6.set_xlim(-args.max_search_radius, args.max_search_radius)
   ax6.set_ylim(-args.max_search_radius, args.max_search_radius)
   ax6.set_xlabel(r'$x {\rm{ [^\circ]}$')
   ax6.set_ylabel(r'$y {\rm{ [^\circ]}$')
   ax6.grid()


   plt.savefig('./%s/%s_selection_final.pdf'%(args.name, args.name), bbox_inches='tight')


def members_prob(table, clf, vars, clipping_prob = 3, data_0 = None):
   """
   This routine will find probable members through scoring of a passed model (clf).
   """

   has_vars = table.loc[:, vars].notnull().all(axis = 1)

   data = table.loc[has_vars, vars]

   clustering_data = table.loc[has_vars, 'clustering_data'] == 1

   results = pd.DataFrame(columns = ['member_clustering_prob', 'member_clustering'], index = table.index)

   if (clustering_data.sum() > 1):

      if data_0 is None:
         data_0 = data.loc[clustering_data, vars].median().values

      data -= data_0

      data_std = data.loc[clustering_data, :].std().values

      clf.fit(data.loc[clustering_data, :] / data_std)

      log_prob = clf.score_samples(data / data_std)
      label_GMM = log_prob >= np.median(log_prob[clustering_data])-clipping_prob*np.std(log_prob[clustering_data])

      results.loc[has_vars, 'member_clustering_prob'] = log_prob
      results.loc[has_vars, 'member_clustering'] = label_GMM

   return results


def pm_cleaning_GMM_recursive(table, vars, data_0 = None, n_components = 1, covariance_type = 'full', clipping_prob = 3, plots = True, verbose = False, plot_name = ''):
   """
   This routine iteratively find members using a Gaussian mixture model.
   """

   clf = mixture.GaussianMixture(n_components = n_components, covariance_type = covariance_type, means_init = np.zeros((n_components, len(vars))))

   convergence = False
   iteration = 0
   while not convergence:
      if verbose:
         print("\rIteration %i, %i objects remain."%(iteration, table.clustering_data.sum()))

      clust = table.loc[:, vars+['clustering_data']]

      if iteration > 3:
         data_0 = None

      fitting = members_prob(clust, clf, vars, clipping_prob = clipping_prob,  data_0 = data_0)
      
      table['member_clustering'] = fitting.member_clustering
      table['member_clustering_prob'] = fitting.member_clustering_prob
      table['clustering_data'] = (table.clustering_data == 1) & (fitting.member_clustering == 1)
      
      if (iteration > 999):
         convergence = True
      elif iteration > 0:
         convergence = fitting.equals(previous_fitting)

      previous_fitting = fitting.copy()
      iteration += 1

   if plots == True:
      plt.close('all')
      fig, ax1 = plt.subplots(1, 1)
      ax1.plot(table.loc[table.clustering_data != 1, vars[0]], table.loc[table.clustering_data != 1, vars[1]], 'k.', ms = 0.5, zorder = 0)
      ax1.scatter(table.loc[table.clustering_data == 1 ,vars[0]].values, table.loc[table.clustering_data == 1 ,vars[1]].values, c = table.loc[table.clustering_data == 1, 'member_clustering_prob'].values, s = 1, zorder = 1)
      ax1.set_xlabel(r'$\mu_{\alpha*}$')
      ax1.set_ylabel(r'$\mu_{\delta}$')
      ax1.set_xlim(table.loc[:, vars[0]].mean()-5*table.loc[:, vars[0]].std(), table.loc[:, vars[0]].mean()+5*table.loc[:, vars[0]].std())
      ax1.set_ylim(table.loc[:, vars[1]].mean()-5*table.loc[:, vars[1]].std(), table.loc[:, vars[1]].mean()+5*table.loc[:, vars[1]].std())      
      ax1.grid()
      plt.savefig(plot_name, bbox_inches='tight')
      plt.close('all')

   return fitting.member_clustering


def get_uwe(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, norm_uwe = True):
   """
   Calculates the corresponding RUWE for Gaia stars.
   """

   uwe = np.sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al - 5.))

   if norm_uwe:
      #We make use of the normalization array from files table_u0_g_col.txt, table_u0_g.txt

      has_color = np.isfinite(bp_rp) & (np.isfinite(phot_g_mean_mag))
      u0gc = pd.read_csv('./DR2_RUWE_V1/table_u0_g_col.txt', header =0)

      #histogram
      dx = 0.01
      dy = 0.1
      bins = [np.arange(np.amin(u0gc['g_mag'])-0.5*dx, np.amax(u0gc['g_mag'])+dx, dx), np.arange(np.amin(u0gc[' bp_rp'])-0.5*dy, np.amax(u0gc[' bp_rp'])+dy, dy)]
      
      posx = np.digitize(phot_g_mean_mag[has_color], bins[0])
      posy = np.digitize(bp_rp[has_color], bins[1])

      posx[posx < 1] = 1
      posx[posx > len(bins[0])-1] = len(bins[0])-1
      posy[posy < 1] = 1
      posy[posy > len(bins[1])-1] = len(bins[1])-1

      u0_gc = np.reshape(np.array(u0gc[' u0']), (len(bins[0])-1, len(bins[1])-1))[posx, posy]
      uwe[has_color] /= np.array(u0_gc)

      if not all(has_color):
         u0g = pd.read_csv('./DR2_RUWE_V1/table_u0_g.txt', header =0)

         posx = np.digitize(phot_g_mean_mag[~has_color], bins[0])

         posx[posx < 1] = 1
         posx[posx > len(bins[0])-1] = len(bins[0])-1

         u0_c = u0g[' u0'][posx]
         uwe[~has_color] /= np.array(u0_c)

   return uwe


def get_real_error(table):
   """
   This routine calculates the excess of error that should be added to the listed Gaia errors. The lines are based on Fabricius 2020. Figure 21.
   """

   p5 = table.astrometric_params_solved == 31                                        # 5p parameters solved Brown et al. (2020)
   
   table.loc[:, ['parallax_error_old', 'pmra_error_old', 'pmdec_error_old']] = table.loc[:, ['parallax_error', 'pmra_error', 'pmdec_error']].values
   
   table.loc[p5, ['parallax_error', 'pmra_error', 'pmdec_error']] = np.sqrt(table.loc[p5, ['parallax_error', 'pmra_error', 'pmdec_error']].values**2 * 1.05**2)
   table.loc[~p5, ['parallax_error', 'pmra_error', 'pmdec_error']] = np.sqrt(table.loc[~p5, ['parallax_error', 'pmra_error', 'pmdec_error']].values**2 * 1.22**2)

   return table



def clean_astrometry(ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, use_5p = False):
   """
   Select stars with good astrometry in Gaia.
   """

   labels_ruwe = ruwe <= 1.4
   labels_harmonic_amplitude = ipd_gof_harmonic_amplitude <= 0.2                                    # Reject blended transits Fabricius et al. (2020)
   labels_visibility = visibility_periods_used >= 9                                                 # Lindengren et al. (2020)
   labels_excess_noise = astrometric_excess_noise_sig <= 2.0                                        # Lindengren et al. (2020)

   labels_astrometric = labels_ruwe & labels_harmonic_amplitude & labels_visibility & labels_excess_noise

   if use_5p:
      labels_params_solved = astrometric_params_solved == 31                                        # 5p parameters solved Brown et al. (2020)
      labels_astrometric = labels_astrometric & labels_params_solved

   return labels_astrometric


def clean_photometry(gmag, corrected_flux_excess_factor, sigma_flux_excess_factor = 3):
   """
   This routine select stars based on their flux_excess_factor. Riello et al.2020
   """
   from matplotlib.path import Path

   def sigma_corrected_C(gmag, sigma_flux_excess_factor):
      return sigma_flux_excess_factor*(0.0059898 + 8.817481e-12 * gmag ** 7.618399)

   mag_nodes = np.linspace(np.min(gmag)-0.1, np.max(gmag)+0.1, 100)
   
   up = [(gmag, sigma) for gmag, sigma in zip(mag_nodes, sigma_corrected_C(mag_nodes, sigma_flux_excess_factor))]
   down = [(gmag, sigma) for gmag, sigma in zip(mag_nodes[::-1], sigma_corrected_C(mag_nodes, -sigma_flux_excess_factor)[::-1])]

   path_C = Path(up+down, closed=True)
   
   labels_photometric = path_C.contains_points(np.array([gmag, corrected_flux_excess_factor]).T)

   return labels_photometric


def pre_clean_data(phot_g_mean_mag, corrected_flux_excess_factor, ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, sigma_flux_excess_factor = 3, use_5p = False):
   """
   This routine cleans the Gaia data from astrometrically and photometric bad measured stars.
   """
   
   labels_photometric = clean_photometry(phot_g_mean_mag, corrected_flux_excess_factor, sigma_flux_excess_factor = sigma_flux_excess_factor)
   
   labels_astrometric = clean_astrometry(ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, use_5p = use_5p)
   
   return labels_photometric & labels_astrometric


def remove_jobs():
   """
   This routine removes jobs from the Gaia archive server.
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_log_in(gaia_user = None, gaia_paswd = None):
   """
   This routine log in to the Gaia archive.
   """

   from astroquery.gaia import Gaia
   import getpass

   while True:
      try:
         Gaia.login(user=gaia_user, password=gaia_paswd)
         print("Welcome to the Gaia server!")
         break
      except:
         print("Please introduce username and password")
         gaia_user = input("Gaia username: ")
         gaia_paswd = getpass.getpass(prompt='Gaia password: ') 

   return Gaia


def gaia_query(Gaia, query, min_gmag, max_gmag, norm_uwe, test_mode, save_individual_queries, load_existing, name, n, n_total):
   """
   This routine launch the query to the Gaia archive.
   """

   query = query + " AND (phot_g_mean_mag > %.4f) AND (phot_g_mean_mag <= %.4f)"%(min_gmag, max_gmag)

   if not test_mode:
      
      individual_query_filename = './%s/individual_queries/%s_G_%.4f_%.4f.csv'%(name, name, min_gmag, max_gmag)

      if os.path.isfile(individual_query_filename) and load_existing:
         result = pd.read_csv(individual_query_filename)

      else:
         job = Gaia.launch_job_async(query)
         result = job.get_results()
         removejob = Gaia.remove_jobs([job.jobid])
         result = result.to_pandas()

         if save_individual_queries:
            result.to_csv('./%s/individual_queries/%s_G_%.4f_%.4f.csv'%(name, name, min_gmag, max_gmag), index = False)
   else:
      result = pd.DataFrame()
   
   print('\n')
   print('----------------------------')
   print('Table %i of %i: %i stars'%(n, n_total, len(result)))
   print('----------------------------')

   return result, query


def get_mag_bins(min_mag, max_mag, area, mag = None):
   """
   This routine generates logarithmic spaced bins for G magnitude.
   """

   num_nodes = np.max((1, np.round( ( (max_mag - min_mag) * max_mag ** 2 * area)*5e-5)))

   bins_mag = (1.0 + max_mag - np.logspace(np.log10(1.), np.log10(1. + max_mag - min_mag), num = int(num_nodes), endpoint = True))

   return bins_mag


def gaia_multi_query_run(args):
   """
   This routine pipes gaia_query into multiple threads.
   """

   return gaia_query(*args)


def columns_n_conditions(source_table, search_type, astrometric_cols, photometric_cols, quality_cols, ra, dec, min_radius = 0.5, max_radius = 1.0, width = 1.0, height = 1.0, max_gmag_error = 0.5, max_rpmag_error = 0.5, max_bpmag_error = 0.5, min_parallax = -2, max_parallax = 1, max_parallax_error = 1.0, min_pmra = -6, max_pmra = 6, max_pmra_error = 1.0, min_pmdec = -6, max_pmdec = 6, max_pmdec_error = 1.0):

   """
   This routine generates the columns and conditions for the query.
   """

   if 'dr3' in source_table:
      if 'ruwe' not in quality_cols:
         quality_cols = 'ruwe' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
   elif 'dr2' in source_table:
      if 'astrometric_n_good_obs_al' not in quality_cols:
         quality_cols = 'astrometric_n_good_obs_al' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
      if 'astrometric_chi2_al' not in quality_cols:
         quality_cols = 'astrometric_chi2_al' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
      if 'phot_bp_rp_excess_factor' not in quality_cols:
         quality_cols = 'phot_bp_rp_excess_factor' +  (', ' + quality_cols if len(quality_cols) > 1 else '')

   if search_type == 'box':
      search_area = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),BOX('ICRS',%.8f,%.8f,%.8f,%.8f))=1"%(ra, dec, width, height)
   elif search_type == 'anulus':
      search_area = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),CIRCLE('ICRS',%.8f,%.8f,%.8f))=1"%(ra, dec, max_radius) +" AND CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec), CIRCLE('ICRS',%.8f,%.8f,%.8f))=0"%(ra, dec, min_radius)
   else:
      search_area = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),CIRCLE('ICRS',%.8f,%.8f,%.8f))=1"%(ra, dec, max_radius)

   conditions = search_area + ' AND b*b > 16 AND (pmra > %.4f) AND (pmra < %.4f) AND (pmra_error < %.4f) AND (pmdec > %.4f) AND (pmdec < %.4f) AND (pmdec_error < %.4f) AND (parallax > %.4f) AND (parallax < %.4f) AND (parallax_error < %.4f) AND ((1.09*phot_g_mean_flux_error/phot_g_mean_flux) < %.4f) AND ((1.09*phot_bp_mean_flux_error/phot_bp_mean_flux) < %.4f) AND ((1.09*phot_rp_mean_flux_error/phot_rp_mean_flux) < %.4f)'%(min_pmra, max_pmra, max_pmra_error, min_pmdec, max_pmdec, max_pmdec_error, min_parallax, max_parallax, max_parallax_error, max_gmag_error, max_bpmag_error, max_rpmag_error)

   columns = (", " + astrometric_cols if len(astrometric_cols) > 1 else '') + (", " + photometric_cols if len(photometric_cols) > 1 else '') +  (", " + quality_cols if len(quality_cols) > 1 else '')

   query = "SELECT source_id " + columns + " FROM " + source_table + " WHERE " + conditions

   return query, quality_cols
   

def incremental_query(query, area, min_gmag = 10.0, max_gmag = 19.5, norm_uwe = True, use_parallel = True, test_mode = False, save_individual_queries = False, load_existing = False, name = 'output', gaia_user = None, gaia_paswd = None):

   """
   This routine search the Gaia archive and downloads the stars using parallel workers.
   """

   from multiprocessing import Pool, cpu_count

   if not test_mode:
      Gaia = gaia_log_in(gaia_user = gaia_user, gaia_paswd = gaia_paswd)
   else:
      Gaia = None

   mag_nodes = get_mag_bins(min_gmag, max_gmag, area)
   n_total = len(mag_nodes)
   
   if (n_total > 1) and use_parallel:

      print("Executing %s jobs."%(n_total-1))

      nproc = int(np.min((n_total, 20, cpu_count()*2)))

      pool = Pool(nproc-1)

      args = []
      for n, node in enumerate(range(n_total-1)):
         args.append((Gaia, query, mag_nodes[n+1], mag_nodes[n], norm_uwe, test_mode, save_individual_queries, load_existing, name, n, n_total))

      tables_gaia_queries = pool.map(gaia_multi_query_run, args)

      tables_gaia = [results[0] for results in tables_gaia_queries]
      queries = [results[1] for results in tables_gaia_queries]

      result_gaia = pd.concat(tables_gaia)

      pool.close()

   else:
      result_gaia, queries = gaia_query(Gaia, query, min_gmag, max_gmag, norm_uwe, test_mode, save_individual_queries, load_existing, name, 1, 1)

   if not test_mode:
      Gaia.logout()

   return result_gaia, queries


def get_object_properties(args):
   """
   This routine will try to obtain all the required object properties from Simbad or from the user.
   """

   #Try to get object:
   if (args.ra is None) or (args.dec is None):
      try:
         from astroquery.simbad import Simbad
         import astropy.units as u
         from astropy.coordinates import SkyCoord

         customSimbad = Simbad()
         customSimbad.add_votable_fields('distance', 'propermotions', 'dim', 'fe_h')

         object_table = customSimbad.query_object(args.name)
         print('Object found:', object_table['MAIN_ID'])

         coo = SkyCoord(ra = object_table['RA'], dec = object_table['DEC'], unit=(u.hourangle, u.deg))

         args.ra = float(coo.ra.deg)
         args.dec = float(coo.dec.deg)

         #Try to get radius
         if ((args.search_type == 'anulus') or (args.search_type == 'cone')) and args.max_search_radius is None:
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.max_search_radius = max(2.0* np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               if not args.silent:
                  try:
                     args.max_search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
                  except:
                     args.max_search_radius = 1.0
                  
         if (args.search_type == 'anulus') and (args.min_search_radius is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.min_search_radius = max(0.5 * np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               if not args.silent:
                  try:
                     args.min_search_radius = float(input('Inner radius of the anulus search not defined, please enter the inner radius in degrees (Press enter to adopt the default value of 0.5 deg): '))
                  except:
                     args.min_search_radius = 0.5

         if (args.search_type == 'box') and (args.search_height is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.search_height = max(2.0* np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               if not args.silent:
                  try:
                     args.search_height = float(input('Height of search not defined, please enter the width in degrees (Press enter to adopt the default value of 0.5 deg): '))
                  except:
                     args.search_height = 0.5


         if (args.search_type == 'box') and (args.search_width is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.search_width = max(2.0 * np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1) / np.cos(np.deg2rad(args.dec))
            else:
               if not args.silent:
                  try:
                     args.search_width = float(input('Width of search not defined, please enter the width in degrees (Press enter to adopt the default value of 0.5 deg): '))
                  except:
                     args.search_width = 0.5 / np.cos(np.deg2rad(args.dec))

         #We try to get PMs:
         if any((args.min_pmra == None, args.max_pmra == None)):
            if (object_table['PMRA'].mask == False):
               args.pmra = float(object_table['PMRA'])
               args.max_pmra = float(object_table['PMRA']) + 5
               args.min_pmra = float(object_table['PMRA']) - 5
            else:
               if not args.silent:
                  try:
                     args.max_pmra = float(input('Max PMRA not defined, please enter pmra in mas/yr (Press enter to adopt the default value of 5 m.a.s./yr): '))
                  except:
                     args.max_pmra = 5.0
                  try:
                     args.min_pmra = float(input('Min PMRA not defined, please enter pmra in mas/yr (Press enter to adopt the default value of  -5 m.a.s./yr): '))
                  except:
                     args.min_pmra = -5.0

         if any((args.min_pmdec == None, args.max_pmdec == None)):
            if (object_table['PMDEC'].mask == False):
               args.pmdec = float(object_table['PMDEC'])
               args.max_pmdec = float(object_table['PMDEC']) + 5
               args.min_pmdec = float(object_table['PMDEC']) - 5
            else:
               if not args.silent:
                  try:
                     args.max_pmdec = float(input('Max PMDEC not defined, please enter pmdec in mas/yr (Press enter to adopt the default value of 5 m.a.s./yr): '))
                  except:
                     args.max_pmdec = 5.0
                  try:
                     args.min_pmdec = float(input('Min PMDEC not defined, please enter pmdec in mas/yr (Press enter to adopt the default value of  -5 m.a.s./yr): '))
                  except:
                     args.min_pmdec = -5.0

         if (args.min_parallax is None) and not args.silent:
            try:
               args.min_parallax = float(input('Min parallax not defined, please enter parallax in mas/yr (Press enter to adopt the default value of -2 m.a.s.): '))
            except:
               args.min_parallax = -2.0

         if (args.max_parallax is None) and not args.silent:
            try:
               args.max_parallax = float(input('Max parallax not defined, please enter parallax in mas/yr (Press enter to adopt the default value of 1 m.a.s.): '))
            except:
               args.max_parallax = 1.0

      except:
         if args.ra is None:
            args.ra = float(input('R.A. not defined, please enter R.A. in degrees: '))
         if args.dec is None:
            args.dec = float(input('Dec not defined, please enter Dec in degrees: '))

         if not args.silent:
            #Try to get radius
            if args.max_search_radius is None:
               try:
                  args.max_search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
               except:
                  args.max_search_radius = 1.0

            if (args.search_type == 'anulus') and (args.min_search_radius is None):
               try:
                  args.min_search_radius = float(input('Inner radius of the anulus search not defined, please enter the inner radius in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.min_search_radius = 0.5

            if (args.search_type == 'box') and (args.search_height is None):
               try:
                  args.search_height = float(input('Height of search not defined, please enter the width in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.search_height = 0.5

            if (args.search_type == 'box') and (args.search_width is None):
               try:
                  args.search_width = float(input('Width of search not defined, please enter the width in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.search_width = 0.5

   if args.pmra is None:
      args.pmra = 0
   if args.pmdec is None:
      args.pmdec = 0

   if args.max_pmra is None:
      args.max_pmra = args.pmra+5
   if args.min_pmra is None:
      args.min_pmra = args.pmra-5

   if args.max_pmdec is None:
      args.max_pmdec = args.pmdec+5
   if args.min_pmdec is None:
      args.min_pmdec = args.pmdec-5

   if args.min_parallax is None:
      args.min_parallax = -2.0
   if args.max_parallax is None:
      args.max_parallax = 1.0
   if args.parallax is None:
      args.parallax = 0.

   if args.max_search_radius is None:
      args.max_search_radius = 0.5
   if args.min_search_radius is None:
      args.min_search_radius = 0.25

   if (args.search_type == 'box'):
      if (args.search_height is None):
         args.search_height = 0.5
      if (args.search_width is None):
         args.search_width = 0.5
   
   setattr(args, 'area', get_area(args.search_type, args.max_search_radius, args.min_search_radius, args.search_width, args.search_height, args.dec))

   if args.field_radius is None:
      args.field_radius = 1.*np.sqrt(args.area / np.pi + args.max_search_radius**2)

   setattr(args, 'field_area', get_area("anulus", args.field_radius, args.max_search_radius, args.search_width, args.search_height, args.dec))

   if args.use_members:
      setattr(args, 'download_radius', args.field_radius)
   else:
      setattr(args, 'download_radius', args.max_search_radius)

   args.name = args.name.replace(" ", "_")

   args.logfile = './%s/%s.log'%(args.name, args.name)

   print('\n')
   print(' USED PARAMETERS '.center(42, '*'))
   print('- (ra, dec) = (%s, %s) deg.'%(round(args.ra, 5), round(args.dec, 5)))
   print('- pmra = [%s, %s] m.a.s./yr.'%(round(args.min_pmra, 5), round(args.max_pmra, 5)))
   print('- pmdec = [%s, %s] m.a.s./yr.'%(round(args.min_pmdec, 5), round(args.max_pmdec, 5)))
   print('- parallax = [%s, %s] m.a.s.'%(round(args.min_parallax, 5), round(args.max_parallax, 5)))
   print('- radius = %s deg.'%args.max_search_radius)
   print('*'*42+'\n')

   return args


def get_area(search_type, max_radius, min_radius, width, height, dec):
   """
   This routine calculates the covered area.
   """

   if search_type == 'box':
      area = height * width * np.abs(np.cos(np.deg2rad(dec)))
   elif search_type == 'anulus':
      area = np.pi*max_radius**2 - np.pi*min_radius**2
   else:
      area = np.pi*max_radius**2

   return area


def get_coo_split(args, table):
   
   center = c.SkyCoord(args.ra*u.deg, args.dec*u.deg)
   coords = c.SkyCoord(table.ra*u.deg, table.dec*u.deg)

   table['r'] = center.separation(coords).to("arcsec").value
   #table["phi"] = center.position_angle(coords).value
   table['x'], table['y'] = wcs2xy(table.ra, table.dec, args.ra, args.dec)

   data = table.loc[table.r <= (args.max_search_radius*u.deg).to("arcsec").value, :]
   field = table.loc[table.r > (args.max_search_radius*u.deg).to("arcsec").value, :]

   return data, field


def create_dir(path):
   """
   This routine creates directories.
   """
   
   try:
      os.mkdir(path)
   except OSError:  
      print ("Creation of the directory %s failed" % path)
   else:  
      print ("Successfully created the directory %s " % path)


def round_significant(x, ex, sig=1):
   """
   This routine returns a quantity rounded to its error significan figures.
   """
   
   try:
      significant = sig-int(floor(log10(abs(ex))))-1
   except:
      significant = 4

   return round(x, significant), round(ex, significant)


def select_conditions(args, table):
   """
   Select table based on simple conditions
   """

   conditions_astrometry = (table.pmra >= args.min_pmra) & (table.pmra <= args.max_pmra)\
                         & (table.pmdec >= args.min_pmdec) & (table.pmdec <= args.max_pmdec)\
                         & (table.parallax >= args.min_parallax) & (table.parallax <= args.max_parallax)
   
   conditions_photometry = (table.bp_rp >= args.min_bp_rp) & (table.bp_rp <= args.max_bp_rp)\
                         & (table.gmag >= args.min_gmag) & (table.gmag <= args.max_gmag)\
                         & (table.bpmag <= args.max_gmag) & (table.rpmag <= args.max_rpmag)\

   conditions_astrometric_error = (table.pmra_error <= args.max_pmra_error) & (table.pmdec_error <= args.max_pmdec_error) & (table.parallax_error <= args.max_parallax_error)

   conditions_photometric_error = (table.bpmag_error <= args.max_bpmag_error) & (table.rpmag_error <= args.max_rpmag_error) & (table.gmag_error <= args.max_gmag_error)

   table = table[conditions_astrometry & conditions_photometry & conditions_astrometric_error & conditions_photometric_error]
   
   return table


def flatten_tuple(tuples):
   return tuple([element for tupl in tuples for element in tupl])


def get_errors(data, used_cols = None):
   """
   This routine finds the corresponding error columns for the columns in "used_cols"
   """

   errors = pd.DataFrame(index = data.index)

   cols = data.columns
   if not used_cols:
      used_cols = cols

   for col in used_cols:
      if '%s_error'%col in cols:
         errors['%s_error'%col] = data['%s_error'%col]
      else:
         errors['%s_error'%col] = np.ones_like(data[col])*1e-32

   return errors


def median(table):
   """
   This routine calculates the median and the error of the median.
   The formula is 1.25*sigma/sqrt(N) for medians of normal distributions. This is the upper limmit of error.
   """

   var_err = (1.25404*table.std()/np.sqrt(table.count())).add_prefix('median_').add_suffix('_error')
   var_median = table.median().add_prefix('median_')

   return pd.concat([var_median, var_err])


def weighted_avg_err(table):
   """
   This routine calculates the error-weighted mean and its error.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   x_i = table.loc[:, var_cols]

   ex_i = get_errors(table, used_cols = var_cols)

   var_variance = 1./(1./ex_i**2).sum(axis = 0)

   var_avg = (x_i.div(ex_i.values**2)).sum(axis = 0) * var_variance.values

   return pd.concat([var_avg.add_prefix('mean_'), np.sqrt(var_variance[~var_variance.index.duplicated()]).add_prefix('mean_')])


def sigma_clip_cell(table, var_cols = None, sigma = 3):
   """
   This routine performs sigma clipping of the data whithin each Vornoi Cell
   """

   filtered_data = sigma_clip(table.loc[:, var_cols], sigma=sigma, maxiters=None, stdfunc=mad_std)

   index_filt = table[~filtered_data.mask.any(axis = 1)].index
   
   return index_filt


def voronoi_stats(table, coo, name_vor_bin, n_stars = 50, sigma = None):
   """
   This routine calls voronoi_binning using as coordinates those in "coo" and
   compute the mean and the median values for all columns in the table within these cells.
   """
   
   if len(coo) == 3:
      from voronoi_3d_binning import voronoi_3d_binning as voronoi_binning
   else:
      from voronoi_2d_binning import voronoi_2d_binning as voronoi_binning

   var_cols = [x for x in table.columns if not '_error' in x]

   if sigma is not None:
      table_it = table.loc[:, coo+list(set(var_cols) - set(coo))].copy()
      convergence = False
      while not convergence:

         vor_bin, vor_n_stars = voronoi_binning(table_it.loc[:,coo].values, n_stars, quiet =1, wvt = True, cvt = True)

         table_it[name_vor_bin] = vor_bin

         clean_indexes = table_it.groupby([name_vor_bin]).apply(sigma_clip_cell, var_cols = list(set(var_cols) - set(coo)), sigma = sigma)
         clean_index = list(set([index for clean_index in clean_indexes for index in clean_index]))
         
         table_clean = table_it.iloc[table_it.index.isin(clean_index), :]

         convergence = table_it.equals(table_clean)

         table_it = table_clean.copy()

      sigma_clip_used = table_it.index
      sigma_clip_rejected = table.index[~table.index.isin(table_it.index)]
      
      del [table_it, table_clean]

      table = table[table.index.isin(sigma_clip_used)].copy()
   else:
      vor_bin, vor_n_stars = voronoi_binning(table.loc[:,coo].values, n_stars, quiet =1, wvt = True, cvt = True)

      sigma_clip_used = table.index
      sigma_clip_rejected = []

   table[name_vor_bin] = vor_bin

   grouped_table_mean = table.groupby([name_vor_bin]).apply(weighted_avg_err).reset_index()
   grouped_table_median = table.groupby([name_vor_bin])[var_cols].apply(median).reset_index()

   results = grouped_table_mean.drop(columns=['mean_%s'%name_vor_bin, 'mean_%s_error'%name_vor_bin, ]).merge(grouped_table_median, on = [name_vor_bin], how = 'left')
   results['vor_n_stars'] = vor_n_stars
   
   return table, results, vor_bin, sigma_clip_used, sigma_clip_rejected


def voronoi_stats_launcher(args):
   """
   This routine pipes voronoi_stats arguments into multiple threads.
   """

   return voronoi_stats(*args)


def voronoi_stats_parallel(table, voronoi_vars, voronoi_coos, voronoi_names, voronoi_nstars, sigma = None, use_parallel = False):
   """
   This routine launch the Voronoi tesellation
   """

   from multiprocessing import Pool, cpu_count

   args = []
   for voronoi_coo, voronoi_name, voronoi_nstar in zip(voronoi_coos, voronoi_names, voronoi_nstars):
      args.append((table.loc[:,voronoi_vars], voronoi_coo, voronoi_name, voronoi_nstar, sigma))

   if use_parallel:
      pool = Pool(min(cpu_count(), voronoi_coos))
      Y = pool.map(voronoi_stats_launcher, args)
   else:
      Y = []
      for arg in args:
         Y.append(voronoi_stats_launcher(arg))

   tables_filtered = [results[0] for results in Y]
   tables_voronoi = [results[1] for results in Y]
   cells_voronoi = [results[2] for results in Y]
   sigma_clip_used = [results[3] for results in Y]
   sigma_clip_rejected = [results[4] for results in Y]

   return tables_filtered, tables_voronoi, cells_voronoi, sigma_clip_used, sigma_clip_rejected


def str2bool(v):
   """
   This routine converts ascii input to boolean.
   """

   if v.lower() in ('yes', 'true', 't', 'y', '1'):
       return True
   elif v.lower() in ('no', 'false', 'f', 'n', '0'):
       return False
   else:
       raise argparse.ArgumentTypeError('Boolean value expected.')



def main(argv):  
   """
   Inputs
   """
   parser = argparse.ArgumentParser(description="This script asynchronously download Gaia DR2 data, cleans it from poorly measured sources.")
   parser.add_argument('--name', type=str, default = 'Output', help='Name for the Output table.')
   parser.add_argument('--silent', type=str2bool, default = False, help='Accept all default values without asking. Default is False.')
   parser.add_argument('--use_members', type=str2bool, default=True, help='Whether to use only member stars for the epochs alignment or to use all available stars.')
   parser.add_argument('--gaia_user', type=str, default = None, help='Gaia username. Useful for automatization of the script.')
   parser.add_argument('--gaia_paswd', type=str, default = None, help='Gaia password. Useful for automatization of the script.')
   parser.add_argument('--ra', type=float, default = None, help='Central R.A. of the object.')
   parser.add_argument('--dec', type=float, default = None, help='Central Dec. of the object.')
   parser.add_argument('--distance', type=float, default = None, help='Distance to the object.')
   parser.add_argument('--radial_velocity', type=float, default = None, help='Radial_velocity to the object.')
   parser.add_argument('--search_type', type=str, default = 'cone', help='Shape of the area to search. Options are "box", "cone" or "anulus". The "box" size is controlled by the "search_width" and "search_height" parameters. The "cone" radius is controlled by the "search_radius" parameter.')
   parser.add_argument('--search_width', type=float, default = None, help='Width for the cone search in degrees.')
   parser.add_argument('--search_height', type=float, default = None, help='Height for the cone search in degrees.')
   parser.add_argument('--max_search_radius', type=float, default = None, help='Radius of search in degrees.')
   parser.add_argument('--min_search_radius', type=float, default = None, help='Inner radius for the cone search in degrees. Useful for anulus search.')
   parser.add_argument('--field_radius', type=float, default = None, help='Outer radius for the annular region used as control sample. By default is "max_search_radius" + 0.25 deg.')
   parser.add_argument('--pmra', type=float, default= None, help='Proper motion of the object in R.A., if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_pmra" and "max_pmra"')
   parser.add_argument('--pmra_error', type=float, default= None, help='Proper motion of the object in R.A., if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_pmra" and "max_pmra"')
   parser.add_argument('--min_pmra', type=float, default= None, help='Min pmra in mas.')
   parser.add_argument('--max_pmra', type=float, default = None, help='Max pmra in mas.')
   parser.add_argument('--max_pmra_error', type=float, default = 1.0, help='Max error in pmra in mas.')
   parser.add_argument('--pmdec', type=float, default= None, help='Proper motion in Dec. of the object, if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_pmra" and "max_pmra"')
   parser.add_argument('--pmdec_error', type=float, default= None, help='Proper motion in Dec. of the object, if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_pmra" and "max_pmra"')
   parser.add_argument('--min_pmdec', type=float, default= None, help='Min pmdec in mas.')
   parser.add_argument('--max_pmdec', type=float, default = None, help='Max pmdec in mas.')
   parser.add_argument('--max_pmdec_error', type=float, default = 1.0, help='Max error in pmra in mas.')
   parser.add_argument('--parallax', type=float, default=None, help='Parallax of the object, if known, in mas. Default will try to find the info in Simbad or use the middle value between "min_parallax" and "max_parallax"')
   parser.add_argument('--min_parallax', type=float, default= None, help='Min parallax in mas.')
   parser.add_argument('--max_parallax', type=float, default = None, help='Max parallax in mas.')
   parser.add_argument('--max_parallax_error', type=float, default = 1.0, help='Max error in parallax in mas.')
   parser.add_argument('--max_gmag', type=float, default = 20.5, help='Fainter G magnitude')
   parser.add_argument('--max_bpmag', type=float, default = 20.5, help='Fainter G magnitude')
   parser.add_argument('--max_rpmag', type=float, default = 20.5, help='Fainter G magnitude')
   parser.add_argument('--min_gmag', type=float, default = 10.0, help='Brighter G magnitude')
   parser.add_argument('--min_bp_rp', type=float, default = -0.5, help='Bluest color.')
   parser.add_argument('--max_bp_rp', type=float, default = 3.0, help='Reddest color')
   parser.add_argument('--max_gmag_error', type=float, default = 0.01, help='Max error in G magnitude.')
   parser.add_argument('--max_rpmag_error', type=float, default = 0.1, help='Max error in RP magnitude.')
   parser.add_argument('--max_bpmag_error', type=float, default = 0.1, help='Max error in BP magnitude.')
   parser.add_argument('--clean_uwe', type = str2bool, default = True)
   parser.add_argument('--norm_uwe', type = str2bool, default = True)
   parser.add_argument('--clipping_prob_pm', type=float, default=3., help='Sigma used for clipping pm and parallax. Default is 3.')
   parser.add_argument('--clipping_prob_pm_voronoi', type = float, default = 3, help = 'Sigma used for clipping pm and parallax. Default is 3. If 0, no rejection is made.')
   parser.add_argument('--pm_n_components', type=int, default=1, help='Number of Gaussian componnents for pm and parallax clustering. Default is 1.')
   parser.add_argument('--inside_core', type = str2bool, default = False, help='Use only stars inside the core of the galaxy. Default is True.')
   parser.add_argument('--use_parallel', type = str2bool, default = True, help='Use parallelized computation when possible. Default is True.')
   parser.add_argument('--test_mode', type = str2bool, default = False)
   parser.add_argument('--source_table', type = str, default = 'gaiaedr3.gaia_source', help='Gaia source table. Default is gaiaedr3.gaia_source.')
   parser.add_argument('--save_individual_queries', type = str2bool, default = True, help='If True, the code will save the individual queries.')
   parser.add_argument('--load_existing', type = str2bool, default = False, help='If True, the code will try to resume the previous search loading previous individual queries. It should be set to False if a new table is being downloaded. True when a specific search is failing due to connection problems.')
   parser.add_argument('--remove_quality_cols', type = str2bool, default = False, help='If True, the code will remove all quality columns from the final table, except "clean_label".')
   parser.add_argument('--plots', type=str2bool, default=True, help='Create sanity plots. Default is True.')
   parser.add_argument('--clean_data', type = str2bool, default = True, help = 'Screen out bad measurements based on Gaia DR2 quality flags. Default is False.')
   parser.add_argument('--do_first_pass', type = str2bool, default = True, help = 'Performs a first selection using maximum likelihood.')
   parser.add_argument('--use_position_prob', type = str2bool, default = False, help = 'Calculate probabilities of a star being a member using Multi Gaussian Expansion.')
   parser.add_argument('--use_parallax_prob', type = str2bool, default = True, help = 'Calculate probabilities of a star being a member using parallax.')
   parser.add_argument('--membership_prob', type = float, default = None, help = 'Performs a first selection using maximum likelihood.')
   parser.add_argument('--prepare_for_clustering', type = float, default = 0, help = 'Preselect sources before the last clustering')
   parser.add_argument('--sigma_flux_excess_factor', type=float, default=3., help='Sigma used for clipping in flux_excess_factor. Default is 3.')
   parser.add_argument('--only_5p_solutions', type = str2bool, default = False, help='If True, only 5p solution stars will be used. Default is False.')
   parser.add_argument('--voronoi', type = str2bool, default = False, help='If True, the code performs Voronoi sigma clipping. Default False.')

   args = parser.parse_args(argv)
   
   if not any(x == args.search_type for x in ['box', 'cone', 'anulus']):
      print('Pease use a correct "search_type" option: "box" or "cone".')
      sys.exit()

   args = get_object_properties(args)

   create_dir(args.name)
   
   if args.save_individual_queries:
      create_dir(args.name+'/individual_queries')

   astrometric_cols = 'l, b, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, dr2_radial_velocity, dr2_radial_velocity_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr'

   photometric_cols = 'phot_g_mean_flux, phot_g_mean_mag AS gmag, (1.086*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, phot_bp_mean_mag AS bpmag, (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, phot_rp_mean_mag AS rpmag, (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, bp_rp, sqrt( power( (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux), 2) + power( (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux), 2) ) as bp_rp_error'

   quality_cols = 'ecl_lat, pseudocolour, nu_eff_used_in_astrometry, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, astrometric_n_good_obs_al, astrometric_chi2_al, phot_bp_rp_excess_factor, ruwe, (phot_bp_n_blended_transits+phot_rp_n_blended_transits) *1.0 / (phot_bp_n_obs + phot_rp_n_obs) AS beta, ipd_gof_harmonic_amplitude, phot_bp_n_contaminated_transits, phot_rp_n_contaminated_transits'

   query, quality_cols = columns_n_conditions(args.source_table, args.search_type, astrometric_cols, photometric_cols, quality_cols, args.ra, args.dec,
                                              args.min_search_radius, args.download_radius, args.search_width, args.search_height,
                                              max_gmag_error = args.max_gmag_error, max_rpmag_error = args.max_rpmag_error,
                                              max_bpmag_error = args.max_bpmag_error, min_parallax = args.min_parallax, max_parallax = args.max_parallax,
                                              max_parallax_error = args.max_parallax_error, min_pmra = args.min_pmra, max_pmra = args.max_pmra,
                                              max_pmra_error = args.max_pmra_error, min_pmdec = args.min_pmdec, max_pmdec = args.max_pmdec, max_pmdec_error = args.max_pmdec_error)
   try:
      table = pd.read_csv("./%s/%s_raw.csv"%(args.name, args.name))
   except:
      table, queries = incremental_query(query, args.area, min_gmag = args.min_gmag, max_gmag = args.max_gmag, norm_uwe = args.norm_uwe, use_parallel = args.use_parallel,
                                         test_mode = args.test_mode, save_individual_queries = args.save_individual_queries, name = args.name, gaia_user = args.gaia_user, gaia_paswd = args.gaia_paswd)

      table.to_csv("./%s/%s_raw.csv"%(args.name, args.name), index = False)

      f = open("./%s/%s_queries.txt"%(args.name, args.name), 'w+')
      if type(queries) is list:
         for query in queries:
            f.write('%s\n'%query)
            f.write('\n')
      else:
         f.write('%s\n'%queries)
      f.write('\n')
      f.close()
   
   
   # We fix some of the variables using the codes published with EDR3
   zpt.load_tables()
   corrected_parallax = table['parallax'] - zpt.get_zpt(table['gmag'], table['nu_eff_used_in_astrometry'], table['pseudocolour'], table['ecl_lat'], table['astrometric_params_solved'])
   corrected_gmag, corrected_phot_g_mean_flux = correct_gband(table['bp_rp'], table['astrometric_params_solved'], table['gmag'], table['phot_g_mean_flux'])
   corrected_flux_excess_factor = correct_flux_excess_factor(table['bp_rp'], table['phot_bp_rp_excess_factor'])
   
   table['corrected_flux_excess_factor'] = correct_flux_excess_factor(table['bp_rp'], table['phot_bp_rp_excess_factor'])

   clean_label = pre_clean_data(table['gmag'], table['corrected_flux_excess_factor'], table['ruwe'], table['ipd_gof_harmonic_amplitude'], table['visibility_periods_used'], table['astrometric_excess_noise_sig'], table['astrometric_params_solved'], sigma_flux_excess_factor = args.sigma_flux_excess_factor, use_5p = True)

   table['clean_label'] = clean_label

   table['original_parallax'] = table['parallax'].values
   table['parallax'] = corrected_parallax
   
   table['original_gmag'] = table['gmag'].values
   table['gmag'] = corrected_gmag

   table = get_real_error(table)
   table = select_conditions(args, table)

   table.to_csv("./%s/%s_raw_selection.csv"%(args.name, args.name))

   if args.clean_data:
      table = table[table.clean_label == True]

   if args.use_members:

      if args.do_first_pass:
         data, field = get_coo_split(args, table)

         try: 
            probabilities = pd.read_csv("./%s/%s_probabilities.csv"%(args.name, args.name))
            ellipse = pd.read_csv("./%s/%s_ellipse_1.csv"%(args.name, args.name))
            ellipse.majoraxis = ellipse.majoraxis
            ellipse.eps = ellipse.eps
            ellipse.pa = ellipse.pa
            data.loc[:, 'membership_prob'] = probabilities.loc[:, 'membership_prob'].values

         except:
            # Likelihood calculation
            systematics = pd.DataFrame(data={'pmra_error': 0.034, 'pmdec_error': 0.034, 'parallax_error': 0.028, 'gmag_error': 0.002, 'bp_rp_error': 0.02}, index =[0])
            ellipse, probabilities = CalculateProbabilities(args, data.copy(), field.copy(),
                                                   systematics = systematics, probs_filename = "./%s/%s_probabilities.csv"%(args.name, args.name), ellipse_filename="./%s/%s_ellipse_1.csv"%(args.name, args.name))

            data.loc[:, 'membership_prob'] = probabilities.loc[:, 'membership_prob'].values

         field.loc[:, 'membership_prob'] = 0.0
         field.loc[:, 'member'] = False

         data.loc[:, 'member_1_pass'] = manual_select_from_cmd(args, data, ellipse = ellipse)

      else:
         cols = ['source_id', 'l', 'b', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
                  'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                  'dr2_radial_velocity', 'dr2_radial_velocity_error', 'ra_dec_corr',
                  'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
                  'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
                  'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'gmag',
                  'gmag_error', 'bpmag', 'bpmag_error', 'rpmag', 'rpmag_error', 'bp_rp',
                  'bp_rp_error', 'astrometric_n_good_obs_al', 'astrometric_chi2_al',
                  'phot_bp_rp_excess_factor', 'ruwe', 'beta',
                  'ipd_gof_harmonic_amplitude', 'phot_bp_n_contaminated_transits',
                  'phot_rp_n_contaminated_transits', 'clean_label', 'membership_prob', 'member_1_pass', 'x', 'y']
                     
         data = pd.read_csv("./%s/%s_%s.csv"%(args.name, args.name, args.com_option), usecols = cols)

      try:
         data.loc[:, 'clustering_data'] = data.loc[:, 'member_1_pass']
      except:
         data.loc[:, 'clustering_data'] = True

      """
      Perform the selection in the PM-parallax space.
      """
      if (args.use_parallax_prob):
         clustering_vars = ['pmra', 'pmdec', 'parallax']
         if (args.pmra is not None) and (args.pmdec is not None):
            clustering_zero = [args.pmra, args.pmdec, args.parallax]
         else:
            clustering_zero = None
      else:
         clustering_vars = ['pmra', 'pmdec']
         if (args.pmra is not None) and (args.pmdec is not None):
            clustering_zero = [args.pmra, args.pmdec]
         else:
            clustering_zero = None

      if (args.prepare_for_clustering is not None) and (args.pmra is not None) and (args.pmdec is not None):
         pmra_select = np.abs(data.pmra - args.pmra)-data.pmra_error  < args.prepare_for_clustering
         pmdec_select = np.abs(data.pmdec - args.pmdec)-data.pmdec_error  < args.prepare_for_clustering
         data.loc[:, 'clustering_data'] = pmra_select & pmdec_select & (data.clustering_data == True)

      data.loc[:, 'member_2_pass'] = pm_cleaning_GMM_recursive(data.copy(), clustering_vars, data_0 = clustering_zero, n_components = args.pm_n_components, clipping_prob = args.clipping_prob_pm, plots = args.plots, plot_name = './'+args.name+'/PM_selection')

      gaia_selection_vars = ['member_1_pass', 'member_2_pass']
      
      data.loc[:, 'member'] = (data.loc[:, gaia_selection_vars] == True).all(axis = 1)

   else:
      data.loc[:, 'member'] = True

   if args.voronoi:
      try:
         data_filt, data_voronoi, voronoi_cells, index_sel, index_rej = voronoi_stats_parallel(data[data.member == True], ['x', 'y', 'pmra', 'pmdec'], [['x','y']], ['voronoi_cell'], [max(5, int(np.sqrt(len(data[data.member == True]))))], sigma = args.clipping_prob_pm_voronoi)
         data.loc[data.member == True, 'member'] = data.loc[data.member == True].index.isin(index_sel[0])
      except:
         print('Voronoi failed!')
         pass

   stats = statistics(data.loc[data.member, ['pmra', 'pmdec', 'pmra_error', 'pmdec_error']])
   
   args.pmra_new = stats['pmra_median']
   args.pmdec_new = stats['pmdec_median']

   try:
      ellipse = sky_model(args, data, field = field, fit_pge = False, use_center = False, ellipse_plot = './%s/%s_sky_ellipse_2.pdf'%(args.name, args.name))
   except:
      try:
         ellipse = pd.read_csv('./%s/%s_ellipse_1.csv'%(args.name, args.name))
      except:
         ellipse = None


   data.to_csv("./%s/%s.csv"%(args.name, args.name), index = False)

   try:
      stats['Delta_x'] = ellipse.x
      stats['Delta_y'] = ellipse.y
      stats['majoraxis'] = ellipse.majoraxis
      stats['eps'] = ellipse.eps
      stats['pa'] = ellipse.pa
   except:
      pass

   stats['nstars'] = data.member.sum()
   stats.to_frame().T.to_csv("./%s/%s_resumen.csv"%(args.name, args.name), index = False)

   try:
      field.to_csv("./%s/%s_field.csv"%(args.name, args.name), index = False)
      probabilities.to_csv("./%s/%s_probabilities.csv"%(args.name, args.name), index = False)
   except:
      pass

   plot_final_selection(args, data, ellipse = ellipse, plot_prob = True)

   if args.remove_quality_cols:
      try:
         data.drop(columns = [x.strip() for x in quality_cols.split(',')], inplace = True)
      except:
         pass

   logresults = ' RESULTS '.center(82, '-')+'\n ' \
               +' - A total of %i stars were used.\n'%data.member.sum() \
               +' - Error-weighted mean, mean, and median absolute PM of used stars: \n' \
               +' pmra = %s+-%s , pmdec = %s+-%s\n'%flatten_tuple((round_significant(stats['pmra_wmean'], stats['pmra_wmean_error']), round_significant(stats['pmdec_wmean'], stats['pmdec_wmean_error']))) \
               +' pmra = %s+-%s , pmdec = %s+-%s\n'%flatten_tuple((round_significant(stats['pmra_mean'], stats['pmra_mean_error']), round_significant(stats['pmdec_mean'], stats['pmdec_mean_error']))) \
               +' pmra = %s+-%s , pmdec = %s+-%s\n'%flatten_tuple((round_significant(stats['pmra_median'], stats['pmra_median_error']), round_significant(stats['pmdec_median'], stats['pmdec_median_error']))) \
               +'-'*82 + '\n \n Execution ended.\n'

   print('\n')
   print(logresults)

   f = open(args.logfile, 'w+')
   f.write(logresults)
   f.close()

   print('\nDone!.\n')

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)


"""
Andres del Pino Molina
"""

#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
import sys
import os


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


def clean_astrometry(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, ipd_gof_harmonic_amplitude, uwe = None, norm_uwe = True):
   """
   Select stars with good astrometry in Gaia.
   """

   b = 1.2 * np.maximum(np.ones_like(phot_g_mean_mag), np.exp(-0.2*(phot_g_mean_mag-19.5)))

   if uwe is None:
      uwe = get_uwe(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, norm_uwe = norm_uwe)
   
   if norm_uwe:
      labels_uwe = uwe < 1.4
   else:
      labels_uwe = uwe < 1.95
   
   labels_harmonic_amplitude = ipd_gof_harmonic_amplitude <= 0.1 # Fabricius et al. (2020)
   
   labels_astrometric = (uwe < b) & labels_uwe & labels_harmonic_amplitude

   return labels_astrometric, uwe


def clean_photometry(bp_rp, phot_bp_rp_excess_factor):
   """
   Select stars with good photometry in Gaia.
   """

   labels_photometric = (1.0 + 0.015*bp_rp**2 < phot_bp_rp_excess_factor) & (1.5*(1.3 + 0.06*bp_rp**2) > phot_bp_rp_excess_factor)

   return labels_photometric


def pre_clean_data(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, phot_bp_rp_excess_factor, ipd_gof_harmonic_amplitude, uwe = None, norm_uwe = True):
   """
   This routine cleans the Gaia data from astrometrically and photometric bad measured stars.
   """

   labels_photometric = clean_photometry(bp_rp, phot_bp_rp_excess_factor)
   
   labels_astrometric, uwe = clean_astrometry(phot_g_mean_mag, bp_rp, astrometric_chi2_al, astrometric_n_good_obs_al, ipd_gof_harmonic_amplitude, uwe = uwe, norm_uwe = norm_uwe)
   
   return labels_photometric & labels_astrometric, uwe


def remove_jobs():
   """
   This routine removes jobs from the Gaia archive server.
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_log_in():
   """
   This routine log in to the Gaia archive.
   """

   from astroquery.gaia import Gaia
   import getpass
   
   while True:
      user_loging = input("Gaia username: ")
      user_paswd = getpass.getpass(prompt='Gaia password: ') 
      try:
         Gaia.login(user=user_loging, password=user_paswd)
         print("Welcome to the Gaia server!")
         break

      except:
         print("Incorrect username or password!")

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

         try:
            uwe = result['ruwe']
         except:
            uwe = None

         result['clean_label'], uwe = pre_clean_data(result['gmag'], result['bpmag'] - result['rpmag'], result['astrometric_chi2_al'], result['astrometric_n_good_obs_al'], result['phot_bp_rp_excess_factor'], result['ipd_gof_harmonic_amplitude'], uwe = uwe, norm_uwe = norm_uwe)

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



def incremental_query(query, area, min_gmag = 10.0, max_gmag = 19.5, norm_uwe = True, use_parallel = True, test_mode = False, save_individual_queries = False, load_existing = False, name = 'output'):

   """
   This routine search the Gaia archive and downloads the stars using parallel workers.
   """

   from multiprocessing import Pool, cpu_count

   if not test_mode:
      Gaia = gaia_log_in()
   else:
      Gaia = None
   
   mag_nodes = get_mag_bins(min_gmag, max_gmag, area)
   n_total = len(mag_nodes)
   
   if (n_total > 1) and use_parallel:

      print("Executing %s jobs."%(n_total-1))

      nproc = int(np.min((n_total, 20, cpu_count()*2)))

      pool = Pool(nproc)

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
               try:
                  args.max_search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 1 deg): '))
               except:
                  args.max_search_radius = 1.0
                  
         if (args.search_type == 'anulus') and (args.min_search_radius is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.min_search_radius = max(0.5 * np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               try:
                  args.min_search_radius = float(input('Inner radius of the anulus search not defined, please enter the inner radius in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.min_search_radius = 0.5

         if (args.search_type == 'box') and (args.search_height is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.search_height = max(2.0* np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)
            else:
               try:
                  args.search_height = float(input('Height of search not defined, please enter the width in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.search_height = 0.5


         if (args.search_type == 'box') and (args.search_width is None):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.search_width = max(2.0 * np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1) / np.cos(np.deg2rad(args.dec))
            else:
               try:
                  args.search_width = float(input('Width of search not defined, please enter the width in degrees (Press enter to adopt the default value of 0.5 deg): '))
               except:
                  args.search_width = 0.5 / np.cos(np.deg2rad(args.dec))

         #We try to get PMs:
         if any((args.min_pmra == None, args.max_pmra == None)):
            if (object_table['PMRA'].mask == False):
               args.max_pmra = float(object_table['PMRA']) + 1.5
               args.min_pmra = float(object_table['PMRA']) - 1.5
            else:
               try:
                  args.max_pmra = float(input('Max PMRA not defined, please enter pmra in mas/yr (Press enter to adopt the default value of -2 m.a.s.): '))
                  args.min_pmra = float(input('Min PMRA not defined, please enter pmra in mas/yr (Press enter to adopt the default value of  2 m.a.s.): '))
               except:
                  args.max_pmra = 2.0
                  args.min_pmra = -2.0

         if any((args.min_pmdec == None, args.max_pmdec == None)):
            if (object_table['PMDEC'].mask == False):
               args.max_pmdec = float(object_table['PMDEC']) + 1.5
               args.min_pmdec = float(object_table['PMDEC']) - 1.5
            else:
               try:
                  args.max_pmdec = float(input('Max PMDEC not defined, please enter pmdec in mas/yr (Press enter to adopt the default value of -2 m.a.s.): '))
                  args.min_pmdec = float(input('Min PMDEC not defined, please enter pmdec in mas/yr (Press enter to adopt the default value of  2 m.a.s.): '))
               except:
                  args.max_pmdec = 2.0
                  args.min_pmdec = -2.0

         if args.min_parallax is None:
            args.min_parallax = -2.0

         if args.max_parallax is None:
            args.max_parallax = 1.0

      except:
         if args.ra is None:
            args.ra = float(input('R.A. not defined, please enter R.A. in degrees: '))
         if args.dec is None:
            args.dec = float(input('Dec not defined, please enter Dec in degrees: '))

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


   if args.max_pmra is None:
      args.max_pmra = 2.0
   if args.min_pmra is None:
      args.min_pmra = -2.0

   if args.max_pmdec is None:
      args.max_pmdec = 2.0
   if args.min_pmdec is None:
      args.min_pmdec = -2.0

   if args.min_parallax is None:
      args.min_parallax = -2.0
   if args.max_parallax is None:
      args.max_parallax = 1.0

   if args.max_search_radius is None:
      args.max_search_radius = 1.0
   if args.min_search_radius is None:
      args.min_search_radius = 0.5

   print('\n')
   print(' USED PARAMETERS '.center(42, '*'))
   print('- (ra, dec) = (%s, %s) deg.'%(round(args.ra, 5), round(args.dec, 5)))
   print('- pmra = [%s, %s] m.a.s./yr.'%(round(args.min_pmra, 5), round(args.max_pmra, 5)))
   print('- pmdec = [%s, %s] m.a.s./yr.'%(round(args.min_pmdec, 5), round(args.max_pmdec, 5)))
   print('- parallax = [%s, %s] m.a.s.'%(round(args.min_parallax, 5), round(args.max_parallax, 5)))
   print('- radius = %s deg.'%args.max_search_radius)
   print('*'*42+'\n')

   return args


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

   conditions = search_area + ' AND (pmra > %.4f) AND (pmra < %.4f) AND (pmra_error < %.4f) AND (pmdec > %.4f) AND (pmdec < %.4f) AND (pmdec_error < %.4f) AND (parallax > %.4f) AND (parallax < %.4f) AND (parallax_error < %.4f) AND ((1.09*phot_g_mean_flux_error/phot_g_mean_flux) < %.4f) AND ((1.09*phot_bp_mean_flux_error/phot_bp_mean_flux) < %.4f) AND ((1.09*phot_rp_mean_flux_error/phot_rp_mean_flux) < %.4f)'%(min_pmra, max_pmra, max_pmra_error, min_pmdec, max_pmdec, max_pmdec_error, min_parallax, max_parallax, max_parallax_error, max_gmag_error, max_bpmag_error, max_rpmag_error)

   columns = (", " + astrometric_cols if len(astrometric_cols) > 1 else '') + (", " + photometric_cols if len(photometric_cols) > 1 else '') +  (", " + quality_cols if len(quality_cols) > 1 else '')

   query = "SELECT source_id " + columns + " FROM " + source_table + " WHERE " + conditions

   return query, quality_cols
   

def main(argv):  
   """
   Inputs
   """
   parser = argparse.ArgumentParser(description="This script asynchronously download Gaia DR2 data, cleans it from poorly measured sources.")
   parser.add_argument('--name', type=str, default = 'Output', help='Name for the Output table.')
   parser.add_argument('--ra', type=float, default = None, help='Central R.A.')
   parser.add_argument('--dec', type=float, default = None, help='Central Dec.')
   parser.add_argument('--search_type', type=str, default = 'cone', help='Shape of the area to search. Options are "box", "cone" or "anulus". The "box" size is controlled by the "search_width" and "search_height" parameters. The "cone" radius is controlled by the "search_radius" parameter.')
   parser.add_argument('--search_width', type=float, default = None, help='Width for the cone search in degrees.')
   parser.add_argument('--search_height', type=float, default = None, help='Height for the cone search in degrees.')
   parser.add_argument('--max_search_radius', type=float, default = None, help='Radius of search in degrees.')
   parser.add_argument('--min_search_radius', type=float, default = None, help='Inner radius for the cone search in degrees. Useful for anulus search.')
   parser.add_argument('--min_pmra', type=float, default= None, help='Min pmra in mas.')
   parser.add_argument('--max_pmra', type=float, default = None, help='Max pmra in mas.')
   parser.add_argument('--max_pmra_error', type=float, default = 1.5, help='Max error in pmra in mas.')
   parser.add_argument('--min_pmdec', type=float, default= None, help='Min pmdec in mas.')
   parser.add_argument('--max_pmdec', type=float, default = None, help='Max pmdec in mas.')
   parser.add_argument('--max_pmdec_error', type=float, default = 1.5, help='Max error in pmra in mas.')
   parser.add_argument('--min_parallax', type=float, default= None, help='Min parallax in mas.')
   parser.add_argument('--max_parallax', type=float, default = None, help='Max parallax in mas.')
   parser.add_argument('--max_parallax_error', type=float, default = 1., help='Max error in parallax in mas.')
   parser.add_argument('--max_gmag', type=float, default = 19.5, help='Fainter G magnitude')
   parser.add_argument('--min_gmag', type=float, default = 10.0, help='Brighter G magnitude')
   parser.add_argument('--max_gmag_error', type=float, default = 0.25, help='Max error in G magnitude.')
   parser.add_argument('--max_rpmag_error', type=float, default = 0.5, help='Max error in RP magnitude.')
   parser.add_argument('--max_bpmag_error', type=float, default = 0.5, help='Max error in BP magnitude.')
   parser.add_argument('--clean_uwe', type = str2bool, default = True)
   parser.add_argument('--norm_uwe', type = str2bool, default = True)
   parser.add_argument('--use_parallel', type = str2bool, default = True)
   parser.add_argument('--test_mode', type = str2bool, default = False)
   parser.add_argument('--source_table', type = str, default = 'gaiaedr3.gaia_source', help='Gaia source table. Default is gaiadr3.gaia_source.')
   parser.add_argument('--save_individual_queries', type = str2bool, default = True, help='If True, the code will save the individual queries.')
   parser.add_argument('--load_existing', type = str2bool, default = False, help='If True, the code will try to resume the previous search loading previous individual queries. It should be set to False if a new table is being downloaded. True when a specific search is failing due to connection problems.')
   parser.add_argument('--remove_quality_cols', type = str2bool, default = False, help='If True, the code will remove all quality columns from the final table, except "clean_label".')
   args = parser.parse_args(argv)

   if not any(x == args.search_type for x in ['box', 'cone', 'anulus']):
      print('Pease use a correct "search_type" option: "box" or "cone".')
      sys.exit()

   args = get_object_properties(args)

   create_dir(args.name)
   
   if args.save_individual_queries:
      create_dir(args.name+'/individual_queries')

   astrometric_cols = 'ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, dr2_radial_velocity, dr2_radial_velocity_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr'
   
   photometric_cols = 'phot_g_mean_mag AS gmag, (1.09*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, phot_bp_mean_mag AS bpmag, (1.09*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, phot_rp_mean_mag AS rpmag, (1.09*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, bp_rp, sqrt( power( (1.086*phot_bp_mean_flux_error/phot_bp_mean_flux), 2) + power( (1.086*phot_rp_mean_flux_error/phot_rp_mean_flux), 2) ) as bp_rp_error'

   quality_cols = 'astrometric_n_good_obs_al, astrometric_chi2_al, phot_bp_rp_excess_factor, ruwe, (phot_bp_n_blended_transits+phot_rp_n_blended_transits) *1.0 / (phot_bp_n_obs + phot_rp_n_obs) AS beta, ipd_gof_harmonic_amplitude, phot_bp_n_contaminated_transits, phot_rp_n_contaminated_transits'
   
   area = get_area(args.search_type, args.max_search_radius, args.min_search_radius, args.search_width, args.search_height, args.dec)
   
   query, quality_cols = columns_n_conditions(args.source_table, args.search_type, astrometric_cols, photometric_cols, quality_cols, args.ra, args.dec,
                                              args.min_search_radius, args.max_search_radius, args.search_width, args.search_height,
                                              max_gmag_error = args.max_gmag_error, max_rpmag_error = args.max_rpmag_error,
                                              max_bpmag_error = args.max_bpmag_error, min_parallax = args.min_parallax, max_parallax = args.max_parallax,
                                              max_parallax_error = args.max_parallax_error, min_pmra = args.min_pmra, max_pmra = args.max_pmra,
                                              max_pmra_error = args.max_pmra_error, min_pmdec = args.min_pmdec, max_pmdec = args.max_pmdec, max_pmdec_error = args.max_pmdec_error)

   table, queries = incremental_query(query, area, min_gmag = args.min_gmag, max_gmag = args.max_gmag, norm_uwe = args.norm_uwe, use_parallel = args.use_parallel,
                                      test_mode = args.test_mode, save_individual_queries = args.save_individual_queries, name = args.name)
   
   if args.remove_quality_cols:
      table.drop(columns = [x.strip() for x in quality_cols.split(',')], inplace = True)

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

   print('\nDone!.\n')

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)


"""
Andres del Pino Molina
"""


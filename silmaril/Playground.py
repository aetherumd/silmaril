import astropy
import random as r
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math as m
import silmaril_plus as s
import os
from importlib.resources import files
import sample_IMF

folder_path = "C:/Users/josep/Documents/GitHub/silmaril/silmaril/data/relics-glafic-data"
scale = 1000
spread = 375
N = 10**(3)

def pos1(x):
  if(x != 0):
    return (scale*m.exp(-(x/spread)**2))*(x/abs(x))
  else:
    return 0
  # return (scale*m.exp((-(2*(1/scale)*x)**2)))*(x/abs(x))

with open(folder_path + "/starburst.txt", 'w') as file:
  file.write("# 		ID		CurrentAges[MYr]		X[pc]		Y[pc]		Z[pc]		mass[Msun]		t_sim[Myr], z, ctr(code), ctr(pc)")
  IDs = np.empty
  ages= np.empty
  xs = np.empty
  ys = np.empty
  zs = np.empty
  masses = sample_IMF.sample_massive_stars(N)
  t_sims = np.empty
  reds = np.empty
  for i in range (N):
    IDs = np.append(IDs, 0)
    ages = np.append(ages, r.randrange(1, 10**2, 1))
    xs = np.append(xs, pos1(r.randrange(-scale, scale, 1)))
    ys = np.append(ys, pos1(r.randrange(-scale, scale, 1)))
    zs = np.append(zs, pos1(r.randrange(-scale, scale, 1)))
    file.write(str(IDs[i]) + " " + str(ages[i]) + " " + str(xs[i]) + " " + str(ys[i]) + " " + str(zs[i]) + " " + str(masses[i]) + "\n")

wcs = astropy.wcs.WCS(fits.open(folder_path + '/hlsp_relics_model_model_whl0137-08_glafic_v1_x-arcsec-deflect.fits')[0].header)
x_deflections = s.open_fits(folder_path + "/hlsp_relics_model_model_whl0137-08_glafic_v1_x-arcsec-deflect.fits")
y_deflections = s.open_fits(folder_path + "/hlsp_relics_model_model_whl0137-08_glafic_v1_y-arcsec-deflect.fits")
lens = s.Lens(x_deflections,y_deflections,wcs,redshift=0.566,unit='arcsec')
detector = s.Detector(resolution=0.031,fov=30,center=astropy.coordinates.SkyCoord(0.0, 0.0,unit="deg"),psf_fwhm=2.065)
galaxy = s.Galaxy(folder_path + "/starburst.txt",redshift=6.2,size=200,center=astropy.coordinates.SkyCoord(0.0, 0.0,unit="deg"))
galaxy.plot(resolution=1000)
observation = s.Observation(detector,lens,galaxy)
observation.plot(background=3e-11,noise=5e-13,source_resolution=1000,star_by_star=True)
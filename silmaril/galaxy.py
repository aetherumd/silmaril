import numpy as np
import numpy as np
import pandas as pd
import os
from astropy.cosmology import FlatLambdaCDM
from imaging import Grid


class Galaxy():
    def __init__(self,filename,redshift,size):
        # load particle data
        self.data = np.loadtxt(filename)
        self.data_columns = ["ID", "CurrentAges[MYr]", "X[pc]",	"Y[pc]", "Z[pc]", "mass[Msun]", "t_sim[Myr]", "z", "ctr(code)", "ctr(pc)"]
        self.redshift = redshift
        self.size = size
        self.angular_size = ang_size(self.size, self.redshift)

        # compute luminosity distance in pc
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.luminosity_distance = cosmo.luminosity_distance(redshift).value*1e6
    
    def pixel_scale(self,resolution, zoom_factor=1):
        """
        Computes the pixel scale of an image of the galaxy at a given resolution and zoom factor.

        :param resolution: number of pixels on each side of the image
        :type resolution: int
        :param zoom_factor: zoom factor of the image, defaults to 1
        :type zoom_factor: float, optional
        """
        return (2*self.angular_size)/resolution*zoom_factor
    
    def grid(self,resolution,center,zoom_factor=1):
        """
        Returns a grid of points on the sky at a given resolution and zoom factor.
        """
        return Grid(center,resolution,self.pixel_scale(resolution,zoom_factor))

    def as_array(self, resolution, zoom_factor=1):
        """
        Returns an image of the galaxy as a 2d array of fluxes in Jy.

        :param resolution: number of pixels on each side of the image
        :type resolution: int
        :param zoom_factor: zoom factor of the image, defaults to 1
        :type zoom_factor: float, optional

        :return: image of the galaxy
        :rtype: numpy.ndarray
        """
        # convert position to arcseconds
        x_viewed = ang_size(self.data[:, 2], self.redshift)
        y_viewed = ang_size(self.data[:, 3], self.redshift)

        # compute flux using lookup table
        flux = zshifted_flux_jy(
            lum_look_up_table(
                stellar_ages=self.data[:, 1],
                table_link="l1500_inst_e.txt",
                column_idx=1,
                log=True,
            ),
            self.luminosity_distance,
        )

        lums, xedges, yedges = np.histogram2d(
            x_viewed,
            y_viewed,
            bins=resolution,
            weights=flux,
            range=[[-self.angular_size, self.angular_size], [-self.angular_size, self.angular_size]],
        )

        return lums.T*zoom_factor


def lum_to_appmag_ab(lum, lum_dist, redshft):
    """
    Convert point luminosity to point absolute magnitude as detected
    need luminosity of individual star to be in eg/s/Angstrom
    and lumdistance in pc
    """
    abs_magab = -15.65 - 2.54 * np.log10(lum / 10**39)
    app_magab = abs_magab + 5 * np.log10(lum_dist / 100e9) + 50
    return app_magab


def ang_size(phys_size, redshift):
    """
    angsize in arcesconds
    """

    # compute luminosity distance in pc
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lum_dist = cosmo.luminosity_distance(redshift).value*1e6
    size_dist = lum_dist / (1 + redshift) ** 2
    return (phys_size / size_dist) * (2.06e5)


def zshifted_flux_jy(lum, lum_dis, wav_angs=1500):
    """
    need lum distance in parsecs
    """
    return 7.5e11 * (wav_angs / 1500) ** 2 * (lum / (4 * np.pi * (lum_dis * 3e18) ** 2))

def lum_look_up_table(stellar_ages, table_link, column_idx: int, log=True):
    """
    given stsci link and ages, returns likely (log) luminosities
    does this via residuals
    Here are some tables.
    https://www.stsci.edu/science/starburst99/docs/table-index.html
    Data File Format:
    Column 1 : Time [yr]
    Column 2 : Solid Line
    Column 3 : Long Dashed Line
    Column 4 : Short Dashed Line

    M = 10^6 M_sun
    Mlow = 1 M_sun

    Solid line:
    alpha = 2.35, Mup = 100 M

    Long-dashed line:
    alpha = 3.30, Mup = 100 M

    Short-dashed line:
    alpha = 2.35, Mup = 30 M
    """
    if "www" in table_link:
        df = pd.read_csv(table_link, delim_whitespace=True, header=None)
        data = df.to_numpy().astype(float)
    else:
        data = np.loadtxt(table_link)

    look_up_times = data[:, 0]  # yr

    if log is True:
        look_up_lumi = 10 ** data[:, column_idx]
    else:
        look_up_lumi = data[:, column_idx]

    ages_mask = np.ones(stellar_ages.size)
    for i, a in enumerate(stellar_ages):
        closest_age_idx = np.argmin(np.abs(look_up_times - a))
        ages_mask[i] = closest_age_idx
    luminosities = look_up_lumi[np.array(ages_mask, dtype="int")]
    
    return luminosities
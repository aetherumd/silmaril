"""
Module containing the Galaxy class along with methods for working with filters and luminosity values
"""
import numpy as np
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM
from .imaging import Grid
from importlib.resources import files

class Galaxy:
    """Class representing a galaxy defined using particle data

    Parameters
    ----------
    filename : str
        name of the file containing the particle data
    data_format : str
        format of the file containing particle data (options are "fid" and "pos"), defaults to "pos"
    center : astropy.coordinates.SkyCoord
        coordinates of the center of the galaxy
    redshift : float
        redshift of the galaxy
    size : float
        physical size of the galaxy in pc

    Attributes
    ----------
    data
        particle data
    data_format
        format of the file containing particle data
    data_columns
        names of the columns of the particle data
    ages
        array of ages in Myr
    positions
        array of (x, y, z) coordinates for particle data
    center
        coordinates of the center of the galaxy
    redshift
        redshift of the galaxy
    size
        physical size of the galaxy in pc
    angular_size
        angular size of the galaxy in arcseconds
    luminosity_distance
        luminosity distance of the galaxy in pc
    """
    def __init__(self, filename,center, redshift, size,data_format="pos"):
        # load particle data
        self.data = np.loadtxt(filename)
        self.data_format = data_format
        if data_format == "pos":
            self.data_columns = [
                "ID",
                "CurrentAges[MYr]",
                "X[pc]",
                "Y[pc]",
                "Z[pc]",
                "mass[Msun]",
                "t_sim[Myr]",
                "z",
                "ctr(code)",
                "ctr(pc)",
            ]
            self.ages = data[:,1]
            self.positions = data[:,2:5]
        elif data_format == "fid":
            self.data_columns = [
                "t_sim[Myr]",
                "z",
                "ctr(code)",
                "ctr(pc)",
                "ID",
                "CurrentAges[Myr]",
                "log10UV(150nm)Lum[erg/s]",
                "X[pc]",
                "Y[pc]",
                "Z[pc]",
                "Vx[km/s]",
                "Vy[km/s]",
                "Vz[km/s]",
                "mass[Msun]"
            ]
            self.ages = data[:,0]
            self.positions = data[:,7:10]
        else:
            raise ValueError("Invalid format "+str(format))
        
        self.center = center
        self.redshift = redshift
        self.size = size
        self.angular_size = ang_size(self.size, self.redshift)

        # compute luminosity distance in pc
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.luminosity_distance = cosmo.luminosity_distance(redshift).value * 1e6

    def pixel_scale(self, resolution, zoom_factor=1):
        """Computes the pixel scale of an image of the galaxy at a given resolution and zoom factor.

        Parameters
        ----------
        resolution : int
            number of pixels on each side of the image
        zoom_factor : float, optional
            zoom factor of the image, defaults to 1

        Returns
        -------
        float
            pixel scale of the image
        """
        return (2 * self.angular_size) / resolution * zoom_factor

    def grid(self, resolution, zoom_factor=1):
        """Returns a grid of points on the sky at a given resolution and zoom factor.
        
        Parameters
        ----------
        resolution : int
            number of pixels on each side of the image
        zoom_factor : float, optional
            zoom factor of the image, defaults to 1

        Returns
        -------
        imaging.Grid
            grid of points on the sky
        """
        return Grid(self.center, resolution, self.pixel_scale(resolution, zoom_factor))

    def create_image(self, resolution, filter_name="F200W", zoom_factor=1):
        """Returns an image of the galaxy as a 2d array of fluxes in Jy.

        Parameters
        ----------
        resolution : int
            number of pixels on each side of the image
        filter_name : str
            name of the JWST filter to use (uses luminosity lookup table if set to None), defaults to "F200W"
        zoom_factor : float, optional
            zoom factor of the image, defaults to 1

        Returns
        -------
        numpy.ndarray
            image of the galaxy
        """
        pixel_scale = self.pixel_scale(resolution, zoom_factor)

        # convert position to arcseconds
        x_viewed = ang_size(self.positions[:,0], self.redshift)
        y_viewed = ang_size(self.positions[:,1], self.redshift)

        ages = self.ages
        ages=np.where(ages>0., ages, 0.)
        # compute flux using lookup table
        if filter_name is None:
            flux = zshifted_flux_jy(
                lum_look_up_table(
                    stellar_ages=ages,
                    table_link="l1500_inst_e.txt",
                    column_idx=1,
                    log=False,
                ),
                self.luminosity_distance,
            )
        else:
            flux = zshifted_flux_jy(
                lum_lookup_filtered(
                    stellar_ages=ages,
                    z=self.redshift,
                    table_file=None,
                    filter_name = filter_name
                ),
                self.luminosity_distance,
            )

        flux = flux/pixel_scale**2

        lums, xedges, yedges = np.histogram2d(
            x_viewed,
            y_viewed,
            bins=resolution,
            weights=flux,
            range=[
                [-self.angular_size, self.angular_size],
                [-self.angular_size, self.angular_size],
            ],
        )

        return lums.T * zoom_factor

    def plot(self, resolution, norm=None, zoom_factor=1):
        """Plots the galaxy at a given resolution and zoom factor.

        Parameters
        ----------
        resolution : int
            number of pixels on each side of the image
        norm : matplotlib.colors.Normalize, optional
            normalization of the image, defaults to None
        zoom_factor : float, optional
            zoom factor of the image, defaults to 1

        Returns
        -------
        matplotlib.pyplot.figure, matplotlib.pyplot.axes
            figure and axes of the plot
        """
        wcs = self.grid(resolution, zoom_factor).wcs

        if norm is None:
            norm = LogNorm()

        fig = plt.figure()
        ax = fig.add_subplot(projection=wcs)
        im = ax.imshow(
            self.create_image(resolution, zoom_factor), cmap="inferno", norm=norm
        )
        ax.set_facecolor("black")
        ra = ax.coords["ra"]
        ra.set_ticklabel(exclude_overlapping=True)
        ra.set_format_unit("deg")
        # ax.coords.grid(color='white', alpha=0.5, linestyle='solid')
        fig.colorbar(im)

        return fig, ax


def lum_to_appmag_ab(lum, lum_dist, redshift):
    """
    Convert point luminosity to point absolute magnitude as detected

    Parameters
    ----------
    lum : float
        luminosity in eg/s/Angstrom
    lum_dist : float
        luminosity distance in pc
    redshift : float
        redshift

    Return
    ------
    float
        absolute magnitude
    """
    abs_magab = -15.65 - 2.54 * np.log10(lum / 10**39)
    app_magab = abs_magab + 5 * np.log10(lum_dist / 100e9) + 50
    return app_magab


def ang_size(phys_size, redshift):
    """Computes angular size in arcseconds given physical size in pc and redshift

    Parameters
    ----------
    phys_size : float
        physical size in pc
    redshift : float
        redshift

    Returns
    -------
    float
        angular size in arcseconds
    """

    # compute luminosity distance in pc
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lum_dist = cosmo.luminosity_distance(redshift).value * 1e6
    size_dist = lum_dist / (1 + redshift) ** 2
    return (phys_size / size_dist) * (2.06e5)


def zshifted_flux_jy(lum, lum_dis, pivot_wav=1500):
    """
    Computes redshifted flux in Jy

    Parameters
    ----------
    lum : float
        luminosity
    lum_dis : float
        luminosity distance
    pivot_wav : float
        filter pivot wavelength, defaults to 1500

    Returns
    -------
        Redshifted flux in Jy
    """
    return 7.5e10 * (pivot_wav / 1500) ** 2 * (lum / (4 * np.pi * (lum_dis * 3e18) ** 2))

def pivot_wavelength(filter_name,z):
    """
    Computes the pivot wavelength of the given filter blueshifted by z

    Parameters
    ----------
    filter_name : str
        name of JWST filter to use
    z : float
        blueshift

    Returns
    -------
    pivot wavelength
    """
    # load filter throughput curve
    filter_data = np.loadtxt(str(files('silmaril.data.mean_throughputs').joinpath(filter_name + "_mean_system_throughput.txt")),skiprows=1)
    wav_angs = filter_data[:,0]*1e4/(1+z) # convert microns to angstroms and blueshift
    pivot_wav = np.sqrt(np.trapz(wav_angs*filter_data[:,1],wav_angs)/np.trapz(filter_data[:,1]/wav_angs,wav_angs))
    return pivot_wav


def lum_lookup_filtered(
    stellar_ages: float,
    z,
    table_file: str,
    filter_name = "F200W",
    stellar_masses=10,
    m_gal=1e6,
):
    """
    Computes luminosities from galaxy spectrum data using the given filter.

    Parameters
    ----------
    stellar_ages : float
        ages fo the stars in years
    z : float
        redshift of the galaxy
    filter_name : str
        name of JWST filter to use, defaults to "F200W"
    table_file : str
        filepath to the table of spectrum data
    stellar_masses : float
        mass of the individual stars
    m_gal : TYPE, optional
        mass of the galaxy [Msun] from the starburst model. Default is 10^6 Msun

    Returns
    -------
    luminosities : array
        returns the luminosity of the individual stars, default UV luminosity

    """
    filter_data = np.loadtxt(str(files('silmaril.data.mean_throughputs').joinpath(filter_name + "_mean_system_throughput.txt")),skiprows=1)
    wav_angs = filter_data[:,0]*1e4/(1+z) # convert microns to angstroms and blueshift

    ages = np.concatenate((range(1,20),range(20,100,10),range(100,1000,100))) # in Myr

    if table_file is None:
        np.loadtxt(str(files('silmaril.data').joinpath("fig7e.dat")),skiprows=1)
    else:
        starburst = np.loadtxt(table_file, skiprows=3) # load starburst data

    starburst[:,1:] = np.power(10,starburst[:,1:]) # convert from log to linear

    mean_phot_rate = np.zeros(len(ages)) # initialize empty array

    for i in range(len(ages)):
        lum = np.interp(wav_angs,starburst[:,0],starburst[:,i+1])
        mean_phot_rate[i] = np.trapz(wav_angs*lum*filter_data[:,1],wav_angs)/np.trapz(wav_angs*filter_data[:,1],wav_angs)

    lookup = scipy.interpolate.CubicSpline(ages,mean_phot_rate)

    return lookup(stellar_ages)*(stellar_masses / m_gal)
    

def lum_look_up_table(
    stellar_ages: float,
    stellar_masses=10,
    table_link: str = os.path.join("..", "starburst", "l1500_inst_e.txt"),
    column_idx: int = 1,
    log=False,
    m_gal=1e6,
):
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


    Parameters
    ----------
    stellar_ages : float
        ages fo the stars in years
    table_link : str
        link, either URL or filepath to the table
    column_idx : int
        column index to use for the tables
    log : TYPE, optional
        return log10 luminosities? The default is False.
    m_gal : TYPE, optional
        mass of the galaxy [Msun] from the starburst model. Default is 10^6 Msun

    Returns
    -------
    luminosities : array
        returns the luminosity of the individual stars, default UV luminosity

    """

    if "www" in table_link:
        df = pd.read_csv(table_link, delim_whitespace=True, header=None)
        data = df.to_numpy().astype(float)
    else:
        data = np.loadtxt(table_link)
    look_up_times = data[:, 0]  # yr

    if log is True:
        look_up_lumi = data[:, column_idx]
    else:
        look_up_lumi = 10 ** data[:, column_idx]

    # vectorized but need big memoery requirement for big array
    # residuals = np.abs(look_up_times - stellar_ages[:, np.newaxis])
    # closest_match_idxs = np.argmin(residuals, axis=1)
    # luminosities = look_up_lumi[closest_match_idxs]

    # loop, helps with memory allocation
    ages_mask = np.ones(stellar_ages.size)
    for i, a in enumerate(stellar_ages):
        closest_age_idx = np.argmin(np.abs(look_up_times - a))
        ages_mask[i] = closest_age_idx
    luminosities = look_up_lumi[np.array(ages_mask, dtype="int")]

    if log is True:
        lum_scaled = luminosities + np.log10(stellar_masses / m_gal)
    else:
        lum_scaled = luminosities * (stellar_masses / m_gal)

    return lum_scaled


def unpack_pop_ii_data(
    path: str,
    lum_scaling=1e-5,
    lum_link="../particle_data/luminosity_look_up_tables/l1500_inst_e.txt",
    table_column_idx=1,
    return_ids=False,
    return_z=False,
):
    r"""
    Depends on the lookup table function.
    given path or link, gives you look up table luminosities and cleans them up
    sample: https://www.stsci.edu/science/starburst99/data/l1500_inst_e.dat
    Parameters
    ----------
    path
        path to file
    lum_scaling
        scaling factor for luminosity, see stsci tables
    lum_link
        link to the lookup table, can be file path or url to csv

    Returns
    -------
    star_positions
        (x,y,z) positions of stars
    scaled_stellar_lums
        corresponding stellar luminosities
    masses
        masses in M_sun
    ages

    t_myr
        current time in Myr
    """

    pop_2_data = np.loadtxt(path)
    # birth_epochs = pop_2_data[:,0] *1e6
    ages = pop_2_data[:, 1] * 1e6  # convert to myr
    ages[ages < 1e6] = 1e6  # set minimum age
    t_myr = pop_2_data[0, 6]  # current simulation time
    z = pop_2_data[1, 6]
    masses = pop_2_data[:, 5]  # msun

    # use look up table; current bottle neck
    stellar_lums = lum_look_up_table(
        stellar_ages=ages, table_link=lum_link, column_idx=table_column_idx, log=True
    )

    scaled_stellar_lums = stellar_lums * lum_scaling
    star_positions = pop_2_data[:, 2:5]  # (x,y,z)

    if return_ids is True:
        if return_z is True:
            return (
                star_positions,
                scaled_stellar_lums,
                masses,
                ages,
                (t_myr, z),
                pop_2_data[:, 0],
            )
        else:
            return (
                star_positions,
                scaled_stellar_lums,
                masses,
                ages,
                t_myr,
                pop_2_data[:, 0],
            )
    else:
        if return_z is True:
            return star_positions, scaled_stellar_lums, masses, ages, (t_myr, z)
        else:
            return star_positions, scaled_stellar_lums, masses, ages, t_myr

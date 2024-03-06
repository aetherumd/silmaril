import numpy as np
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM
from imaging import Grid

class Galaxy:
    """Class representing a galaxy defined using particle data

    Parameters
    ----------
    filename : str
        name of the file containing the particle data
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
    data_columns
        names of the columns of the particle data
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
    def __init__(self, filename, center, redshift, size):
        # load particle data
        self.data = np.loadtxt(filename)
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

    def create_image(self, resolution, zoom_factor=1):
        """Returns an image of the galaxy as a 2d array of fluxes in Jy.

        Parameters
        ----------
        resolution : int
            number of pixels on each side of the image
        zoom_factor : float, optional
            zoom factor of the image, defaults to 1

        Returns
        -------
        numpy.ndarray
            image of the galaxy
        """
        pixel_scale = self.pixel_scale(resolution, zoom_factor)

        # convert position to arcseconds
        x_viewed = ang_size(self.data[:, 2], self.redshift)
        y_viewed = ang_size(self.data[:, 3], self.redshift)

        ages = self.data[:, 1]
        ages=np.where(ages>0., ages, 0.)
        # compute flux using lookup table
        flux = zshifted_flux_jy(
            lum_look_up_table(
                stellar_ages=ages,
                table_link="l1500_inst_e.txt",
                column_idx=1,
                log=False,
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

        return 20 * lums.T * zoom_factor

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


def lum_to_appmag_ab(lum, lum_dist, redshft):
    """Convert point luminosity to point absolute magnitude as detected
    need luminosity of individual star to be in eg/s/Angstrom
    and lumdistance in pc
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


def zshifted_flux_jy(lum, lum_dis, wav_angs=1500):
    """need lum distance in parsecs"""
    return 7.5e10 * (wav_angs / 1500) ** 2 * (lum / (4 * np.pi * (lum_dis * 3e18) ** 2))

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

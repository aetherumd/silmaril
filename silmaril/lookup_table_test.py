import numpy as np
import os
import scipy
from importlib.resources import files

def lum_lookup_filtered(
    stellar_ages: float,
    z,
    table_file: str,
    filter_name="F200W",
    stellar_masses=10,
    m_gal=1e6,
):
    """
    Computes luminosities from galaxy spectrum data using the given filter.

    Parameters
    ----------
    stellar_ages : float
        ages of the stars in Myr
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
    filter_data = np.loadtxt(
        str(
            files("silmaril.data.mean_throughputs").joinpath(
                filter_name + "_mean_system_throughput.txt"
            )
        ),
        skiprows=1,
    )
    wav_angs = filter_data[:, 0] * 1e4 / (1 + z)  # convert microns to angstroms and blueshift

    ages = np.concatenate((range(1, 20), range(20, 100, 10), range(100, 1000, 100)))  # in Myr

    if table_file is None:
        starburst = np.loadtxt(str(files("silmaril.data").joinpath("fig7e.dat")), skiprows=3)
    else:
        starburst = np.loadtxt(table_file, skiprows=3)  # load starburst data

    starburst[:, 1:] = np.power(10, starburst[:, 1:])  # convert from log to linear

    mean_phot_rate = np.zeros(len(ages))  # initialize empty array

    for i in range(len(ages)):
        lum = np.interp(wav_angs, starburst[:, 0], starburst[:, i + 1])
        mean_phot_rate[i] = np.trapz(wav_angs * lum * filter_data[:, 1], wav_angs) / np.trapz(
            wav_angs * filter_data[:, 1], wav_angs
        )

    lookup = scipy.interpolate.CubicSpline(ages, mean_phot_rate)

    return lookup(stellar_ages) * (stellar_masses / m_gal)


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

stellar_ages = 0
stellar_masses = 0
table_link = "table.csv"

# choose any of these
filter_name = "F200W"
m_gal = 1e6

# display lookup table
print(lum_look_up_table(stellar_ages, stellar_masses, table_link, filter_name, m_gal))

# cross validate table data with entries from muspelheim

# Magnitudes
# These zipped archives contain AB magnitudes in a selection 
# of JWST (NIRCam and MIRI) and HST filters for stars at the 
# same selection ages for which spectra are available. Magnitudes 
# have been computed at redshifts z=1-20, in redshifts increments 
# of 1.0, but please don't be shy to reach out if you need higher 
# redshift resolution, redshifts outside this range or predictions 
# for different filters. All magnitudes are computed under the 
# assumption of an ΩM=0.3, ΩΛ=0.7, H0=70 km s-1 Mpc-1 cosmology. 
# At z≥5.8, all flux shortward of the Lyman-α limit is assumed to 
# be absorbed by the neutral intergalactic medium. At lower redshifts, 
# no absorption shortward of the Lyman-α limit is applied.

# 10% solar tracks 9-575 M Szecsi et al. (2022) [v1.0 NIRCam]
# https://www.astro.uu.se/~ez/muspelheim/models/ABmag_Muspelheim_v1.0_BoOST22_SMC_grid_interpolation_NIRCam.zip
"""
Module containing utility methods for manipulating images and grids
"""

import numpy as np
from scipy.ndimage import map_coordinates, rotate, shift
from astropy.io import fits
from astropy.wcs import WCS
import yt
import re
from unyt import unyt_array


class Grid:
    """
    Grid of points on the sky

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        coordinates of the center of the grid
    num_pix : int
        number of pixels on each side of the grid
    scale : float
        pixel scale of the grid in arcseconds

    Attributes
    ----------
    center
        Coordinates of the center of the grid
    n
        number of pixels on each side of the grid
    scale
        pixel scale of the grid in arcseconds
    fov
        field of view of the grid in arcseconds
    x
        RA coordinates of the grid
    y
        DEC coordinates of the grid
    grid
        grid of coordinates in meshgrid format
    wcs
        astropy.WCS object of the grid
    """

    def __init__(self, center, num_pix, scale):
        self.center, self.n, self.scale = center, num_pix, scale
        self.fov = num_pix * scale

        # convert fov and pixel scale to degrees
        fov_deg = self.fov / 3600
        scale_deg = scale / 3600

        # get ra and dec of the center in degrees
        ra = center.ra.deg
        dec = center.dec.deg

        # generate x and y coordinates for the center of each pixel
        self.x = np.linspace(
            ra + fov_deg / 2 - scale_deg / 2, ra - fov_deg / 2 + scale_deg / 2, self.n
        )
        self.y = np.linspace(
            dec - fov_deg / 2 + scale_deg / 2, dec + fov_deg / 2 - scale_deg / 2, self.n
        )
        self.grid = np.meshgrid(self.x, self.y)

        # create wcs containing position information
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [(self.n + 1) / 2, (self.n + 1) / 2]
        self.wcs.wcs.cdelt = [-scale_deg, scale_deg]
        self.wcs.wcs.crval = np.array([ra, dec])
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    def as_2d_array(self):
        """Returns the grid as a 2d array of (ra,dec) coordinate pairs.

        Returns
        -------
        numpy.ndarray
            array of shape (n,n,2)
        """
        return np.transpose(self.grid, (1, 2, 0))

    def as_list_of_points(self):
        """Flattens the grid into a list of (ra,dec) coordinate pairs.

        Returns
        -------
        numpy.ndarray
            array of shape (n^2,2)
        """
        return np.reshape(self.as_2d_array(), (-1, 2))


def open_fits(file_name):
    """
    Returns the data from a fits file as a 2d numpy array

    Parameters
    ----------
    file_name : str
        name of the fits file

    Returns
    -------
    data : numpy.ndarray
        data array
    """
    return fits.open(file_name)[0].data


def transform_image(image, pixel_scale, angle=0, center=(0, 0)):
    """Rotates and shifts an image.

    Parameters
    ----------
    image : numpy.ndarray
        image to transform
    angle : float
        angle by which to rotate the image given in degrees
    center : tuple
        new center of the image
    pixel_scale : float
        pixel scale of the image

    Returns
    -------
    numpy.ndarray
        transformed image
    """
    pixel_shift = -center[1] / pixel_scale, center[0] / pixel_scale
    transformed_image = image
    transformed_image = rotate(transformed_image, angle, order=0, reshape=False)
    transformed_image = shift(transformed_image, pixel_shift, order=0)
    # return al.Array2D.no_mask(transformed_image,pixel_scales=scale)
    return transformed_image


def fast_griddata(points, values, xi):
    """A faster version of scipy.interpolate.griddata implemented using scipy.ndimage.map_coordinates.

    Parameters
    ----------
    points : numpy.ndarray
        list of points at which the values are defined
    values : numpy.ndarray
        values to interpolate
    xi : numpy.ndarray
        list of points at which to interpolate

    Returns
    -------
    numpy.ndarray
        interpolated values
    """
    n = np.sqrt(points.shape[0]).astype(int)
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    i = (y_max - xi[:, 1]) / (y_max - y_min) * (n - 1)
    j = (xi[:, 0] - x_min) / (x_max - x_min) * (n - 1)

    values_2d = np.reshape(values, (n, n))
    return map_coordinates(values_2d, np.array([i, j]), order=1)


def list_of_points_from_grid(grid):
    """Flattens a grid into a list of points.

    Parameters
    ----------
    grid : numpy.ndarray
        grid to flatten

    Returns
    -------
    numpy.ndarray
        flattened grid
    """
    grid_points = np.transpose(grid, (1, 2, 0))
    grid_points = np.reshape(grid_points, (-1, 2))

    return grid_points

def convert_to_text(filepath, 
    epf = [
        ("particle_family", "b"),
        ("particle_tag", "b"),
        ("particle_birth_epoch", "d"),
        ("particle_metallicity", "d"),
    ]
):
    """Converts a simulation output file into a usable text file for galaxy

    Parameters
    ----------
    filepath: str
        path to simulation output file
    epf: list(str, str)
        list of extra particle fields used for loading simulation in yt

    Returns
    -------
    str
        name of the output text file
    """
    # Constants used in the function
    headers = ["#", "ID", "CurrentAges[MYr]", "X[pc]", "Y[pc]", "Z[pc]", "mass[Msun]", "t_sim[MYr], z, ctr(code), ctr(pc)"]
    cgs_yr = 3.1556926e7  # 1yr (in s)
    cgs_pc = 3.08567758e18  # pc (in cm)

    ds = yt.load(filepath, extra_particle_fields=epf)

    time = ds.current_time.in_units("Myr").value
    z = ds.current_redshift

    # center of mass for stars in code units
    ad = ds.all_data()
    x1 = ad["star", "particle_position_x"]
    y1 = ad["star", "particle_position_y"]
    z1 = ad["star", "particle_position_z"]

    center_code = np.array([x1.mean(), y1.mean(), z1.mean()])
    center_pc = unyt_array(center_code * ds.length_unit.in_units("pc").value, "pc")

    h_0 = ds.hubble_constant * 100 # hubble parameter (km/s/Mpc)
    h_0_invsec = h_0 * 1e5 / (1e6 * cgs_pc)  # hubble constant h [km/s Mpc-1]->[1/sec]
    h_0inv_yr = 1 / h_0_invsec / cgs_yr  # 1/h_0 [yr]
    stellar_ages = np.array(ad["star", "particle_birth_epoch"]) * h_0inv_yr / 1e6 + 13.787 * 1e3

    x2 = x1.in_units("pc") - center_pc[0]
    y2 = y1.in_units("pc") - center_pc[1]
    z2 = z1.in_units("pc") - center_pc[2]

    # The 8 values in the last column are time, z, center_code(x,y,z), center_pc(x,y,z)
    last_col = [time, z, center_code[0], center_code[1], center_code[2], center_pc[0].value, center_pc[1].value, center_pc[2].value]

    pattern = r".*info_(\d{5})\.txt"
    file_name = "output_" + re.search(pattern, filepath).group(1) + ".txt"

    with open(file_name, "w") as f:
        # Formats the header row
        f.write("\t\t".join(headers) + "\n")
        ###
        # ID is right now 0, need code to randomly assign unique IDs to stars. The star masses are also
        # defaulted to 10 MSuns, might want to investigate randomly assigning masses through Dr. Ricotti's code. 
        ###
        for i in range(len(stellar_ages)):
            # In the last column, only the first 8 rows have information, the rest are 0s.
            if i >= 8:
                last_col_info = 0
            else:
                last_col_info = last_col[i]

            r_i = [0, stellar_ages[i], x2[i].value, y2[i].value, z2[i].value, 10, last_col_info]
            r_sci = ["{:.18e}".format(n) for n in r_i]
            r_str = " ".join(r_sci)
            f.write(r_str + "\n")

    return file_name


# def grid(center, num_pix, pixel_scale):
#     """Create a 2D grid of coordinates.

#     Parameters
#     ----------
#     center : tuple
#         center of the grid
#     fov : float
#         field of view of the grid
#     pixel_scale : float
#         pixel scale of the grid

#     Returns
#     -------
#     numpy.ndarray
#         2D grid of coordinates
#     """
#     x_center = center[0]
#     y_center = center[1]
#     fov = num_pix * pixel_scale
#     x = np.linspace(
#         x_center + fov / 2 - pixel_scale / 2,
#         x_center - fov / 2 + pixel_scale / 2,
#         int(fov / pixel_scale),
#     )
#     y = np.linspace(
#         y_center - fov / 2 + pixel_scale / 2,
#         y_center + fov / 2 - pixel_scale / 2,
#         int(fov / pixel_scale),
#     )
#     return np.meshgrid(x, y)

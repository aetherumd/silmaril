"""
Module containing methods and classes for generating simulated observations
"""

import numpy as np
from astropy.wcs import WCS
from numba import jit
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import path
from .utilities import *
from scipy.ndimage import gaussian_filter


class Grid:
    """Grid of points on the sky.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        Coordinates of the center of the grid
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


class Detector:
    """Class representing a detector.

    Parameters
    ----------
    resolution : float
        Resolution of the detector in arcseconds per pixel
    fov : float
        Field of view of the detector in arcseconds
    center : astropy.coordinates.SkyCoord
        Coordinates of the center of the image
    psf : np.ndarray
        Point spread function of the detector

    Attributes
    ----------
    resolution
        Resolution of the detector in arcseconds per pixel
    fov
        Field of view of the detector in arcseconds
    center
        Coordinates of the center of the image
    psf
        Point spread function of the detector
    num_pix
        Number of pixels on each side of the image
    grid
        Grid of coordinates of the image plane
    wcs
        astropy.WCS object of the image plane grid
    """

    def __init__(self, resolution, fov, center, psf_fwhm):
        self.resolution = resolution
        self.fov = fov
        self.center = center
        self.psf_fwhm = psf_fwhm
        self.num_pix = int(fov / resolution)
        self.grid = Grid(center, self.num_pix, resolution)
        self.wcs = self.grid.wcs


class Observation:
    """Class representing an observation of a lensed galaxy using a detector

    Parameters
    ----------
    detector : Detector
        detector used to take the observation
    lens : Lens
        lensing cluster
    galaxy : Galaxy
        source galaxy

    Attributes
    ----------
    detector
        detector used to take the observation
    lens
        lensing cluster
    galaxy
        source galaxy
    traced_pixel_corners
        grid of source plane coordinates of the corners of each pixel on
        the image plane
    """

    def __init__(self, detector, lens, galaxy):
        self.detector, self.lens, self.galaxy = detector, lens, galaxy

        # perform ray tracing and save traced grid
        image_pixel_corners = Grid(
            self.detector.center, self.detector.num_pix + 1, self.detector.resolution
        )
        traced_pixel_corners = self.lens.trace_grid(image_pixel_corners, self.galaxy.redshift)
        self.traced_pixel_corners = np.transpose(traced_pixel_corners, (1, 2, 0))

    def simulate_observation(
        self,
        background,
        noise,
        source_resolution,
        source_center=(0, 0),
        source_rotation=0,
        zoom_factor=1,
    ):
        """Simulates an observation of the lensed galaxy.

        Parameters
        ----------
        background : float
            background level of the image
        noise : float
            standard deviation of the noise
        source_resolution : int
            number of pixels on each side of the source image
        source_center : tuple, optional
            coordinate offset in arcseconds of the center of the source
            image, defaults to (0,0)
        source_rotation : float, optional
            rotation in degrees of the source image, defaults to 0
        zoom_factor : float, optional
            zoom factor of the source image, defaults to 1

        Returns
        -------
        numpy.ndarray
            simulated observation as a 2d array of luminosity values
        """
        # compute lensed image
        nonempty_pixels, arc_pixels, polygons, luminosities = self.trace_pixels(
            source_resolution, source_center, source_rotation, zoom_factor
        )
        lensed_image = np.zeros((self.detector.num_pix, self.detector.num_pix))
        for i, p in enumerate(nonempty_pixels):
            lensed_image[p] = luminosities[i] * self.detector.resolution**2

        lensed_image = gaussian_filter(
            lensed_image, self.detector.psf_fwhm / 2.355
        ) + np.random.normal(background, noise, lensed_image.shape)

        return lensed_image

    def save_to_fits(
        self,
        filename,
        background,
        noise,
        source_resolution,
        source_center=(0, 0),
        source_rotation=0,
        zoom_factor=1,
    ):
        """Simulates an observation of the lensed galaxy and saves it to a fits file with WCS information.

        Parameters
        ----------
        filename : str
            name of the file to save the observation to
        background : float
            background level of the image
        noise : float
            standard deviation of the noise
        source_resolution : int
            number of pixels on each side of the source image
        source_center : tuple, optional
            coordinate offset in arcseconds of the center of the source
            image, defaults to (0,0)
        source_rotation : float, optional
            rotation in degrees of the source image, defaults to 0
        zoom_factor : float, optional
            zoom factor of the source image, defaults to 1
        """
        lensed_image = self.simulate_observation(
            background,
            noise,
            source_resolution,
            source_center,
            source_rotation,
            zoom_factor,
        )
        hdu = fits.PrimaryHDU(lensed_image)
        hdu.header = self.detector.wcs.to_header()
        hdu.writeto(filename, overwrite=True)

    def trace_pixels(
        self, source_resolution, source_center=(0, 0), source_rotation=0, zoom_factor=1, star_by_star=False
    ):
        """"""
        # create source image
        galaxy_image = self.galaxy.create_image(source_resolution, zoom_factor)
        galaxy_pixel_scale = self.galaxy.pixel_scale(source_resolution, zoom_factor)
        # rotate source image
        transformed_galaxy_image = transform_image(
            galaxy_image, galaxy_pixel_scale, source_rotation, source_center
        )
        # create source grid
        source_grid = self.galaxy.grid(source_resolution, zoom_factor)

        nonempty_pixels, arc_pixels = get_arc_pixels(
            source_grid, self.traced_pixel_corners, self.detector.grid
        )
        polygons = get_traced_pixels(source_grid, self.traced_pixel_corners, nonempty_pixels)

        # if star_by_star:
        #     x_viewed = self.galaxy.positions[:,0]
        #     y_viewed = self.galaxy.positions[:,1]
        #     flux = self.galaxy
        luminosities = get_traced_luminosities(
            transformed_galaxy_image,
            source_grid,
            self.traced_pixel_corners,
            nonempty_pixels,
        )

        return nonempty_pixels, arc_pixels, polygons, luminosities

    def plot(
        self,
        background,
        noise,
        source_resolution,
        source_center=(0, 0),
        source_rotation=0,
        zoom_factor=1,
        norm=None,
    ):
        """Plot the lensed galaxy as seen by the detector.

        Parameters
        ----------
        background : float
            background level of the image
        noise : float
            standard deviation of the noise
        source_resolution : int
            number of pixels on each side of the source image
        source_center : tuple, optional
            coordinate offset in arcseconds of the center of the source
            image, defaults to (0,0)
        source_rotation : float, optional
            rotation in degrees of the source image, defaults to 0
        zoom_factor : float, optional
            zoom factor of the source image, defaults to 1
        norm : matplotlib.colors.Normalize, optional
            normalization of the image, defaults to None

        Returns
        -------
        tuple(matplotlib.figure.Figure,matplotlib.axes.Axes)
            matplotlib figure and axes objects
        """
        wcs = self.detector.grid.wcs

        if norm is None:
            norm = LogNorm()

        fig = plt.figure()
        ax = fig.add_subplot(projection=wcs)
        im = ax.imshow(
            self.simulate_observation(
                background,
                noise,
                source_resolution,
                source_center,
                source_rotation,
                zoom_factor,
            ),
            cmap="gray",
            norm=norm,
        )
        ax.set_facecolor("black")
        ra = ax.coords["ra"]
        ra.set_ticklabel(exclude_overlapping=True)
        ra.set_format_unit("deg")
        # ax.coords.grid(color='white', alpha=0.5, linestyle='solid')
        fig.colorbar(im)

        return fig, ax


@jit(nopython=True)
def nonempty_pixel_indices(traced_pixel_corners, source_x_range, source_y_range):
    """Returns a list of pixel indices on the image plane corresponding to pixels
    that fall within the source image when traced back to the source plane.

    Parameters
    ----------
    traced_corners_grid : numpy.ndarray
        grid of source plane coordinates of the corners of each pixel on
        the image plane
    source_x_range : list
        min and max x coordinates of the source image
    source_y_range : list
        min and max y coordinates of the source image

    Returns
    -------
    np.ndarray
        list of pixel indices
    """
    image_pix = traced_pixel_corners.shape[0] - 1  # number of pixels on each side of the image
    nonempty_pixels = []
    # loop over all pixels on the image plane
    for i in range(image_pix):
        for j in range(image_pix):
            # check if top left corner of traced pixel is within the source image grid
            bottom_left = traced_pixel_corners[i, j]
            if (
                (bottom_left[1] < source_y_range[1])
                and (bottom_left[1] > source_y_range[0])
                and (bottom_left[0] < source_x_range[1])
                and (bottom_left[0] > source_x_range[0])
            ):
                nonempty_pixels.append((i, j))

    return nonempty_pixels


def get_arc_pixels(source_grid, traced_pixel_corners, image_plane, nonempty_pixels=None):
    """Returns the list of pixels on the image plane that fall within the source
    image when traced back to the source plane.

    Parameters
    ----------
    source_grid : numpy.ndarray
        grid of source plane coordinates
    traced_corners_grid : numpy.ndarray
        grid of source plane coordinates of the corners of each pixel on
        the image plane
    image_grid : numpy.ndarray
        grid of image plane coordinates

    Returns
    -------
    tuple(numpy.ndarray,numpy.ndarray)
        two element tuple consisting of a list of pixel indices and a
        list of pixel coordinates
    """

    if nonempty_pixels is None:
        source_x_range = [np.min(source_grid.x), np.max(source_grid.x)]
        source_y_range = [np.min(source_grid.y), np.max(source_grid.y)]
        nonempty_pixels = nonempty_pixel_indices(
            traced_pixel_corners,
            nb.typed.List(source_x_range),
            nb.typed.List(source_y_range),
        )

    image_plane_grid = image_plane.as_2d_array()

    return nonempty_pixels, np.array([image_plane_grid[index] for index in nonempty_pixels])


def get_traced_pixels(source_grid, traced_pixel_corners, nonempty_pixels=None):
    """Returns a list of polygons on the source plane corresponding to pixels on the image plane.
    Each polygon is represented as an array of shape (4,2) where each row is a vertex of the polygon.

    Parameters
    ----------
    source_grid : Grid
        grid of source plane coordinates
    traced_corners_grid : numpy.ndarray
        grid of source plane coordinates of the corners of each pixel on
        the image plane

    Returns
    -------
    numpy.ndarray
        list of polygons on the source plane as an array of shape
        (n,4,2)
    """
    if nonempty_pixels is None:
        source_x_range = [np.min(source_grid.x), np.max(source_grid.x)]
        source_y_range = [np.min(source_grid.y), np.max(source_grid.y)]
        nonempty_pixels = nonempty_pixel_indices(
            traced_pixel_corners,
            nb.typed.List(source_x_range),
            nb.typed.List(source_y_range),
        )

    polygons = np.zeros((len(nonempty_pixels), 4, 2))

    for p in range(len(nonempty_pixels)):
        i, j = nonempty_pixels[p]
        top_left = traced_pixel_corners[i + 1, j]
        top_right = traced_pixel_corners[i + 1, j + 1]
        bottom_right = traced_pixel_corners[i, j + 1]
        bottom_left = traced_pixel_corners[i, j]
        polygons[p] = np.array([top_left, top_right, bottom_right, bottom_left])

    return polygons


def get_traced_luminosities(source_image, source_grid, traced_pixel_corners, nonempty_pixels=None):
    """
    Lenses the given source image and returns a 2d array of luminosity values corresponding to
    pixels on the image plane.

    Parameters
    ----------
    source_image : numpy.ndarray
        image of the source
    source_grid : Grid
        grid of source plane coordinates
    traced_pixel_corners : numpy.ndarray
        grid of source plane coordinates of the corners of each pixel on
        the image plane
    nonempty_pixels : np.ndarray
        list of indices of pixels on the image plane that fall on the 
        source image when traced back to the source plane

    Returns
    -------
    numpy.ndarray
        array of luminosity values in Jy/arcsec^2
    """
    x = source_grid.x
    y = source_grid.y
    source_plane_2d = source_grid.as_2d_array()

    if nonempty_pixels is None:
        source_x_range = [np.min(source_grid.x), np.max(source_grid.x)]
        source_y_range = [np.min(source_grid.y), np.max(source_grid.y)]
        nonempty_pixels = nonempty_pixel_indices(
            traced_pixel_corners,
            nb.typed.List(source_x_range),
            nb.typed.List(source_y_range),
        )

    lum = np.zeros(len(nonempty_pixels))

    for r in range(len(nonempty_pixels)):
        i, j = nonempty_pixels[r]
        top_left = traced_pixel_corners[i + 1, j]
        top_right = traced_pixel_corners[i + 1, j + 1]
        bottom_right = traced_pixel_corners[i, j + 1]
        bottom_left = traced_pixel_corners[i, j]

        vertices = [top_left, top_right, bottom_right, bottom_left]

        traced_pixel = path.Path(vertices)

        # compute bounding box
        x_min = min([v[0] for v in vertices])
        y_min = min([v[1] for v in vertices])
        x_max = max([v[0] for v in vertices])
        y_max = max([v[1] for v in vertices])

        y_index_range = [
            np.where(y_min - y > 0, y_min - y, np.inf).argmin(),
            np.where(y - y_max > 0, y - y_max, np.inf).argmin(),
        ]
        x_index_range = [
            np.where(x - x_max > 0, x - x_max, np.inf).argmin(),
            np.where(x_min - x > 0, x_min - x, np.inf).argmin(),
        ]

        image_slice = source_image[
            y_index_range[0] : y_index_range[1] + 1,
            x_index_range[0] : x_index_range[1] + 1,
        ]

        if image_slice.size == 0:
            lum[r] = 0
        else:
            source_plane_slice = source_plane_2d[
                y_index_range[0] : y_index_range[1] + 1, x_index_range[0] : x_index_range[1] + 1
            ]
            source_points = np.reshape(source_plane_slice, (-1, 2))

            index = traced_pixel.contains_points(source_points)
            luminosity_values = image_slice.flatten()[index]
            if len(luminosity_values) == 0:
                lum[r] = 0
            else:
                lum[r] = np.mean(luminosity_values)

    return lum


def traced_luminosities_from_stars(x_viewed,y_viewed,flux,source_grid,traced_pixel_corners, nonempty_pixels=None):
    """
    Lenses the given source galaxy directly from particle data and returns a 2d array of luminosity values corresponding to
    pixels on the image plane.

    Parameters
    ----------
    x_viewed : numpy.ndarray
        array of particle x coordinates in arcseconds
    y_viewed : numpy.ndarray
        array of particle y coordinates in arcseconds
    flux : numpy.ndarray
        array of particle fluxes in Jy
    traced_pixel_corners : numpy.ndarray
        grid of source plane coordinates of the corners of each pixel on
        the image plane
    nonempty_pixels : np.ndarray
        list of indices of pixels on the image plane that fall on the 
        source image when traced back to the source plane

    Returns
    -------
    numpy.ndarray
        array of luminosity values in Jy/arcsec^2
    """
    x = source_grid.x
    y = source_grid.y

    if nonempty_pixels is None:
        source_x_range = [np.min(source_grid.x), np.max(source_grid.x)]
        source_y_range = [np.min(source_grid.y), np.max(source_grid.y)]
        nonempty_pixels = nonempty_pixel_indices(
            traced_pixel_corners,
            nb.typed.List(source_x_range),
            nb.typed.List(source_y_range),
        )

    source_x_range = [np.min(x),np.max(x)]
    source_y_range = [np.min(y),np.max(y)]

    # transform coordinates
    source_center = [np.sum(source_x_range)/2,np.sum(source_y_range)/2] 
    x_viewed = x_viewed + source_center[0]
    y_viewed = y_viewed + source_center[1]
    
    lum = np.zeros(len(nonempty_pixels))

    for r in range(len(nonempty_pixels)):
        i, j = nonempty_pixels[r]
        top_left = traced_pixel_corners[i,j]
        top_right = traced_pixel_corners[i+1,j]
        bottom_right = traced_pixel_corners[i+1,j+1]
        bottom_left = traced_pixel_corners[i,j+1]

        vertices = [top_left,top_right,bottom_right,bottom_left]

        traced_pixel = path.Path(vertices)

        # compute bounding box
        x_min = min([v[0] for v in vertices])
        y_min = min([v[1] for v in vertices])
        x_max = max([v[0] for v in vertices])
        y_max = max([v[1] for v in vertices])

        condition = (x_viewed > x_min) & (x_viewed < x_max) & (y_viewed > y_min) & (y_viewed < y_max)

        if condition.sum() == 0:
            lum[r] = 0
        else:
            source_points = np.column_stack((x_viewed[condition], y_viewed[condition]))

            index = traced_pixel.contains_points(source_points)
            pixel_fluxes = flux[condition]

            luminosity_values = pixel_fluxes[index]
            if len(luminosity_values) == 0:
                lum[r] = 0
            else:
                polygon_area = abs(np.linalg.det([top_left-bottom_left,bottom_right-bottom_left]))
                lum[r] = np.sum(luminosity_values)/polygon_area
    
    return lum

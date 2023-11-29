import numpy as np
from scipy.ndimage import map_coordinates, rotate, shift
from astropy.io import fits


def open_fits(file_name):
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


def grid(center, num_pix, pixel_scale):
    """Create a 2D grid of coordinates.

    Parameters
    ----------
    center : tuple
        center of the grid
    fov : float
        field of view of the grid
    pixel_scale : float
        pixel scale of the grid

    Returns
    -------
    numpy.ndarray
        2D grid of coordinates
    """
    x_center = center[0]
    y_center = center[1]
    fov = num_pix * pixel_scale
    x = np.linspace(
        x_center + fov / 2 - pixel_scale / 2,
        x_center - fov / 2 + pixel_scale / 2,
        int(fov / pixel_scale),
    )
    y = np.linspace(
        y_center - fov / 2 + pixel_scale / 2,
        y_center + fov / 2 - pixel_scale / 2,
        int(fov / pixel_scale),
    )
    return np.meshgrid(x, y)

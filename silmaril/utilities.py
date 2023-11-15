import numpy as np
from scipy.ndimage import map_coordinates, rotate, shift
from astropy.io import fits

def open_fits(file_name):
    return fits.open(file_name)[0].data

def transform_image(image,pixel_scale,angle=0,center=(0,0)):
    """
    Rotates and shifts an image.

    :param image: image to transform
    :type image: numpy.ndarray
    :param angle: angle by which to rotate the image given in degrees
    :type angle: float
    :param center: new center of the image
    :type center: tuple
    :param pixel_scale: pixel scale of the image
    :type pixel_scale: float

    :return: transformed image
    :rtype: numpy.ndarray
    """
    pixel_shift = -center[1]/pixel_scale, center[0]/pixel_scale
    transformed_image = image
    transformed_image = rotate(transformed_image,angle,order=0,reshape=False)
    transformed_image = shift(transformed_image,pixel_shift,order=0)    
    # return al.Array2D.no_mask(transformed_image,pixel_scales=scale)
    return transformed_image

def fast_griddata(points,values,xi):
    """
    A faster version of scipy.interpolate.griddata implemented using scipy.ndimage.map_coordinates.

    :param points: list of points at which the values are defined
    :type points: numpy.ndarray
    :param values: values to interpolate
    :type values: numpy.ndarray
    :param xi: list of points at which to interpolate
    :type xi: numpy.ndarray

    :return: interpolated values
    :rtype: numpy.ndarray
    """
    n = np.sqrt(points.shape[0]).astype(int)
    y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
    x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
    i = (y_max-xi[:,1])/(y_max-y_min)*(n-1)
    j = (xi[:,0]-x_min)/(x_max-x_min)*(n-1)

    values_2d = np.reshape(values, (n,n))
    return map_coordinates(values_2d, np.array([i, j]), order=1)

def list_of_points_from_grid(grid):
    """
    Flattens a grid into a list of points.

    :param grid: grid to flatten
    :type grid: numpy.ndarray

    :return: flattened grid
    :rtype: numpy.ndarray
    """
    grid_points = np.transpose(grid,(1,2,0))
    grid_points = np.reshape(grid_points,(-1,2))

    return grid_points


def grid(center,num_pix,pixel_scale):
    """
    Create a 2D grid of coordinates.

    :param center: center of the grid
    :type center: tuple
    :param fov: field of view of the grid
    :type fov: float
    :param pixel_scale: pixel scale of the grid
    :type pixel_scale: float

    :return: 2D grid of coordinates
    :rtype: numpy.ndarray
    """
    x_center = center[0]
    y_center = center[1]
    fov = num_pix*pixel_scale
    x = np.linspace(x_center + fov / 2 - pixel_scale / 2, x_center - fov / 2 + pixel_scale / 2, int(fov / pixel_scale))
    y = np.linspace(y_center - fov / 2 + pixel_scale / 2 , y_center + fov / 2 - pixel_scale / 2, int(fov / pixel_scale))
    return np.meshgrid(x, y)


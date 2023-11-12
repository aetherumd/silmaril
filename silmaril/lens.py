from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import map_coordinates
from utilities import *

class Lens():
    """
    Class representing a lensing cluster.

    :param x_deflections: x-component of the deflection angles
    :type x_deflections: np.ndarray
    :param y_deflections: y-component of the deflection angles
    :type y_deflections: np.ndarray
    :param wcs: WCS object for the deflection angle map
    :type wcs: astropy.wcs.WCS
    :param redshift: redshift of the lensing cluster
    :type redshift: float
    :unit: units of the deflection angles (options are "arcseconds" and "pixels"), defaults to "arcseconds"
    :type unit: str, optional
    """
    def __init__(self,x_deflections,y_deflections,wcs,redshift,unit="arcsec"):
        self.wcs = wcs
        self.redshift = redshift
        self.scale = proj_plane_pixel_scales(wcs)*3600
        if unit == "arcsec":
            self.x_deflections, self.y_deflections = x_deflections/self.scale[0], y_deflections/self.scale[1]
        elif unit == "pixels":
            self.x_deflections, self.y_deflections = x_deflections, y_deflections
        else:
            raise ValueError("unit must be either 'arcsec' or 'pixels'")
    
    def magnification(self,grid,source_redshift):
        """
        Computes the magnification of the lensing cluster on the given grid.

        :param grid: grid on which to compute the magnification
        :type grid: Grid
        :param source_redshift: redshift of the source
        :type source_redshift: float

        :return: 2d array of magnification values
        :rtype: numpy.ndarray
        """
        x, y = grid.x, grid.y
        traced_points_x, traced_points_y = self.trace_grid(grid,source_redshift)

        # compute jacobian
        dxdx = np.gradient(traced_points_x, x, axis=1)
        dxdy = np.gradient(traced_points_x, y, axis=0)
        dydx = np.gradient(traced_points_y, x, axis=1)
        dydy = np.gradient(traced_points_y, y, axis=0)

        return abs(1/(dxdx*dydy-dxdy*dydx))

    def convergence(self,grid,source_redshift):
        """
        Computes the convergence of the lensing cluster on the given grid.

        :param grid: grid on which to compute the convergence
        :type grid: Grid
        :param source_redshift: redshift of the source
        :type source_redshift: float

        :return: 2d array of convergence values
        :rtype: numpy.ndarray
        """
        x, y = grid.x, grid.y
        traced_points_x, traced_points_y = self.trace_grid(grid,source_redshift)

        # compute jacobian
        dxdx = np.gradient(traced_points_x, x, axis=1)
        dydy = np.gradient(traced_points_y, y, axis=0)

        return abs(1-0.5*(dxdx+dydy))

    def magnification_line(self,grid,source_redshift,threshold=500):
        """
        Returns a list of points in the image plane with magnification greater than a given threshold.

        :param grid: grid on which to compute the magnification
        :type grid: Grid
        :param source_redshift: redshift of the source
        :type source_redshift: float
        :param threshold: threshold used to filter magnification values, defaults to 500
        :type threshold: float, optional

        :return: list of points with magnification greater than threshold
        """
        image_plane_magnification = self.magnification(grid,source_redshift)
        image_plane_points = grid.as_list_of_points()
        return image_plane_points[image_plane_magnification.flatten() > threshold]

    def caustic(self,image_plane_grid,source_redshift,threshold=500):
        """
        Returns a list of points on the source plane corresponding to the caustic of the lensing cluster.
        The caustic is computed by ray tracing the points on the magnification line back to the source plane.

        :param image_plane_grid: image plane grid on which to compute the magnification
        :type image_plane_grid: Grid
        :param source_redshift: redshift of the source
        :type source_redshift: float
        :param threshold: threshold used to filter magnification values, defaults to 500
        :type threshold: float, optional

        :return: list of points on the caustic
        :rtype: numpy.ndarray
        """
        image_plane_points = self.magnification_line(image_plane_grid,source_redshift,threshold)
        source_plane_points = self.trace_points(image_plane_points,source_redshift)
        return source_plane_points

    def trace_grid(self,grid,source_redshift):
        """
        Ray trace a grid from the image plane back to the source plane at a given redshift.

        :param grid: grid to trace
        :type grid: Grid
        :param source_redshift: redshift of the source
        :type source_redshift: float

        :return: traced grid in meshgrid format
        :rtype: tuple(numpy.ndarray,numpy.ndarray)
        """
        # compute deflection angle scale factor
        scale_factor = deflection_angle_scale_factor(self.redshift,source_redshift)

        # convert coordinate grids into list of points
        image_plane_points_world = grid.as_list_of_points()
        image_plane_points_pix = self.wcs.all_world2pix(image_plane_points_world,0,ra_dec_order=True)

        # perform ray tracing by interpolating deflection angles
        traced_points_x = image_plane_points_pix[:,0] - scale_factor*map_coordinates(self.x_deflections, np.flip(image_plane_points_pix.T,axis=0), order=1)
        traced_points_y = image_plane_points_pix[:,1] - scale_factor*map_coordinates(self.y_deflections, np.flip(image_plane_points_pix.T,axis=0), order=1)

        traced_points_x, traced_points_y = self.wcs.all_pix2world(traced_points_x,traced_points_y,0,ra_dec_order=True)

        n = grid.n
        traced_points_x = np.reshape(traced_points_x, (n,-1))
        traced_points_y = np.reshape(traced_points_y, (n,-1))

        return traced_points_x, traced_points_y

    def trace_points(self,points,source_redshift):
        """
        Ray trace a set of points from the image plane back to the source plane at a given redshift.
        
        :param points: list of points to trace given as an array of shape (n,2)
        :type points: numpy.ndarray

        :return: list of traced points
        :rtype: numpy.ndarray
        """
        # compute deflection angle scale factor
        scale_factor = deflection_angle_scale_factor(self.redshift,source_redshift)

        # convert coordinate grids into list of points
        image_plane_points_pix = self.wcs.all_world2pix(points,0,ra_dec_order=True)

        # perform ray tracing by interpolating deflection angles
        traced_points_x = image_plane_points_pix[:,0] - scale_factor*map_coordinates(self.x_deflections, np.flip(image_plane_points_pix.T,axis=0), order=1)
        traced_points_y = image_plane_points_pix[:,1] - scale_factor*map_coordinates(self.y_deflections, np.flip(image_plane_points_pix.T,axis=0), order=1)

        return np.concatenate((traced_points_x,traced_points_y),axis=1)

def deflection_angle_scale_factor(z1,z2,H0=70,Om0=0.3):
    """
    Compute the deflection angle scale factor :math:`D_{ds}/D_s` for a lens at redshift z1 and a source at redshift z2.

    :param z1: redshift of the lens
    :type z1: float
    :param z2: redshift of the source
    :type z2: float

    :return: deflection angle scale factor :math:`D_{ds}/D_s`
    :rtype: float
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    D_ds = cosmo.angular_diameter_distance_z1z2(z1,z2) # distance from source to lens
    D_s = cosmo.angular_diameter_distance_z1z2(0,z2) # distance to source

    return float(D_ds/D_s)
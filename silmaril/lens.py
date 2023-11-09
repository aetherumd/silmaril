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
        
        reshaped_grid = np.transpose(a=grid,axes=(1,2,0))
        y = reshaped_grid[:,0,1]
        x = reshaped_grid[0,:,0]
        traced_points_x, traced_points_y = self.trace_grid(grid,source_redshift)

        # compute jacobian
        dxdx = np.gradient(traced_points_x, x, axis=1)
        dxdy = np.gradient(traced_points_x, y, axis=0)
        dydx = np.gradient(traced_points_y, x, axis=1)
        dydy = np.gradient(traced_points_y, y, axis=0)

        return abs(1/(dxdx*dydy-dxdy*dydx))

    def convergence(self,grid,source_redshift):

        reshaped_grid = np.transpose(a=grid,axes=(1,2,0))
        y = reshaped_grid[:,0,1]
        x = reshaped_grid[0,:,0]
        traced_points_x, traced_points_y = self.trace_grid(grid,source_redshift)

        # compute jacobian
        dxdx = np.gradient(traced_points_x, x, axis=1)
        dydy = np.gradient(traced_points_y, y, axis=0)

        return 1-0.5*(dxdx+dydy)

    def magnification_line(self,grid,source_redshift,threshold=500):
        pass

    def caustic(self,image_plane_grid,source_redshift,threshold=500):
        pass

    def trace_grid(self,grid,source_redshift):
        # compute deflection angle scale factor
        scale_factor = deflection_angle_scale_factor(self.redshift,source_redshift)

        # convert coordinate grids into list of points
        image_plane_points_world = list_of_points_from_grid(grid)
        image_plane_points_pix = self.wcs.all_world2pix(image_plane_points_world,0,ra_dec_order=True)

        # perform ray tracing by interpolating deflection angles
        traced_points_x = image_plane_points_pix[:,0] - scale_factor*map_coordinates(self.x_deflections, np.flip(image_plane_points_pix.T,axis=0), order=1)
        traced_points_y = image_plane_points_pix[:,1] - scale_factor*map_coordinates(self.y_deflections, np.flip(image_plane_points_pix.T,axis=0), order=1)

        traced_points_x, traced_points_y = self.wcs.all_pix2world(traced_points_x,traced_points_y,0,ra_dec_order=True)

        n = grid[0].shape[0]
        traced_points_x = np.reshape(traced_points_x, (n,-1))
        traced_points_y = np.reshape(traced_points_y, (n,-1))

        return traced_points_x, traced_points_y

    def trace_points(self,points,source_redshift):
        pass


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
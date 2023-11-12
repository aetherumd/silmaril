import numpy as np
from astropy.wcs import WCS

class Grid():
    """
    Grid of points on the sky.

    :param center: Coordinates of the center of the grid
    :type center: astropy.coordinates.SkyCoord
    :param num_pix: number of pixels on each side of the grid
    :type num_pix: int
    :param scale: pixel scale of the grid in arcseconds
    :type scale: float

    :ivar center: Coordinates of the center of the grid
    :ivar n: number of pixels on each side of the grid
    :ivar scale: pixel scale of the grid in arcseconds
    :ivar fov: field of view of the grid in arcseconds
    :ivar x: RA coordinates of the grid
    :ivar y: DEC coordinates of the grid
    :ivar grid: grid of coordinates in meshgrid format
    :ivar wcs: astropy.WCS object of the grid
    """
    def __init__(self,center,num_pix,scale):
        self.center, self.n, self.scale = center, num_pix, scale
        self.fov = num_pix*scale

        # convert fov and pixel scale to degrees
        fov_deg = self.fov / 3600
        scale_deg = scale / 3600

        # get ra and dec of the center in degrees
        ra = center.ra.deg
        dec = center.dec.deg

        # generate x and y coordinates for the center of each pixel
        self.x = np.linspace(ra + fov_deg / 2 - scale_deg / 2, ra - fov_deg / 2 + scale_deg / 2, self.n)
        self.y = np.linspace(dec - fov_deg / 2 + scale_deg / 2 , dec + fov_deg / 2 - scale_deg / 2, self.n)
        self.grid = np.meshgrid(self.x, self.y)

        # create wcs containing position information
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [(self.n+1)/2,(self.n+1)/2]
        self.wcs.wcs.cdelt = [-scale_deg,scale_deg]
        self.wcs.wcs.crval = np.array([ra,dec])
        self.wcs.wcs.ctype = ["RA---TAN","DEC--TAN"]
    
    def as_2d_array(self):
        """
        Returns the grid as a 2d array of (ra,dec) coordinate pairs.

        :return: array of shape (n,n,2)
        :rtype: numpy.ndarray
        """
        return np.transpose(self.grid,(1,2,0))

    def as_list_of_points(self):
        """
        Flattens the grid into a list of (ra,dec) coordinate pairs.

        :return: array of shape (n^2,2)
        :rtype: numpy.ndarray
        """
        return np.reshape(self.as_2d_array(),(-1,2))

class Detector():
    """
    Class representing a detector.

    :param resolution: Resolution of the detector in arcseconds per pixel
    :type resolution: float
    :param fov: Field of view of the detector in arcseconds
    :type fov: float
    :param center: Coordinates of the center of the image
    :type center: astropy.coordinates.SkyCoord
    :param psf: Point spread function of the detector
    :type psf: np.ndarray
    """
    def __init__(self,resolution,fov,center,psf):
        self.resolution = resolution
        self.fov = fov
        self.center = center
        self.psf = psf

class Observation():
    def __init__(detector,lens,galaxy,noise,background_level):
        pass

    def observe():
        pass

    def plot(center,angle):
        pass

    def animate(center,angle):
        pass
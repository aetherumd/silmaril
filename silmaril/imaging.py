import numpy as np
from astropy.wcs import WCS

class Grid():
    def __init__(self,center,fov,scale):
        self.center, self.fov, self.scale = center, fov, scale
        self.n = int(fov / scale)
        self.x = np.linspace(center[0] + fov / 2 - scale / 2, center[0] - fov / 2 + scale / 2, self.n)
        self.y = np.linspace(center[1] - fov / 2 + scale / 2 , center[1] + fov / 2 - scale / 2, self.n)
        self.grid = np.meshgrid(self.x, self.y)

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [(self.n+1)/2,(self.n+1)/2]
        self.wcs.wcs.cdelt = [-self.scale,self.scale]
        self.wcs.wcs.crval = np.array(self.center)
        self.wcs.wcs.ctype = ["RA---TAN","DEC--TAN"]
    
    def as_2d_array(self):
        return np.transpose(self.grid,(1,2,0))

    def as_list_of_points(self):
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
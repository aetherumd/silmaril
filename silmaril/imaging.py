
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
class Node():
  def __init__(self, zams_mass = None, curr_mass = None, track_point = None, age = None, temp_eff = None, lbol = None, log_g = None, r = None, computational_flag = None):
    #Zero Age Main Sequence Mass (MSolar)
    self.zams_mass = zams_mass
    #Current Mass at this track point (Msolar)
    self.curr_mass = curr_mass
    #track point of this data
    self.track_point = track_point
    #Age of the object (Years)
    self.age = age
    #Effective temperature of the object (Kelvin)
    self.temp_eff = temp_eff
    #Bolometric Luminosity of the object (Erg/s)
    self.lbol = lbol
    #Surface gravity of the object (g)
    self.log_g = log_g
    #Solar Radius (RSun)
    self.r = r
    #Computational Flag (bool)
    self.computational_flag = computational_flag
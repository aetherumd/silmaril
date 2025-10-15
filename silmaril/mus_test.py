import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u, constants as const


def get_filter(filter_name: str = "None"):
    #str(files("silmaril.data.mean_throughputs").joinpath(filter_name + "_mean_system_throughput.txt"))
    try:
        filter_data = np.loadtxt(
            "./data/mean_throughputs/" + filter_name + "_mean_system_throughput.txt", #silmaril/silmaril/data/mean_throughputs/F070W_mean_system_throughput.txt
            dtype = float,
            skiprows=1,
        )
        filter_data[:,0] = [x*(10**4) for x in filter_data[:,0]]
        return np.array(filter_data[:, 0]), np.array(filter_data[:,1])
    except FileNotFoundError:
        print(("File silmaril\\data\\mean_throughputs\\"+ filter_name + "_mean_system_throughput.txt does not exist"))
        pass

def get_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def get_flux(redshift, filter_name):
    print(os.listdir(os.getcwd()))
    all_museplheim_files = get_files('data/muspelheim_files')

    #database will hold a tuple of (zams_mass, current_mass, lambda_values, flux_values) for each trackpoint in each txt file
    #change to be hashmap w key = age : value = tuple of (zams_mass, current_mass, lambda_values, flux_values)
    database = []
    unique_mass = []
    unique_age = []
    for file_name in all_museplheim_files:
        print(file_name)
        try:
            with open(file_name, "r") as file:
                lines = file.readlines()  # Read all lines into a list

        # Get the second and third lines if they exist
        except FileNotFoundError:
            print(f"Error: The file '{file_name}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        #Note: because of indexing, all lines are shifted back one
        #so line 1 = index 0
        #Note: for each txt file, lines[0] = empty and lines[-1] is a data point
        first_line = lines[0].strip()
        i = 0
        while i < len(lines):
            if lines[i].strip() == first_line:
                #add the previous trackpoint (if there is one) into the database
                if i != 0:
                    lambda_values = np.array(lamda_array)
                    flux_values = np.array(flux_array)
                    filter_wavelength, filter_output = get_filter(filter_name)
                    integral_filter = np.abs(np.trapz(filter_output * filter_wavelength, filter_wavelength))
                    
                    #shorten lambda_values to be over filter_wavelength
                    lam_min = filter_wavelength.min()
                    lam_max = filter_wavelength.max()
                    mask = (lambda_values >= lam_min) & (lambda_values <= lam_max)
                    lam_sub = lambda_values[mask]
                    flux_sub = flux_values[mask]
                    lam_interp = interp1d(lam_sub, lam_sub, kind='linear', bounds_error=False, fill_value='extrapolate')
                    flux_interp = interp1d(lam_sub, flux_sub, kind='linear', bounds_error=False, fill_value='extrapolate')
                    # Evaluate interpolated values at filter_wavelength
                    lam_on_filter = lam_interp(filter_wavelength)        # essentially == filter_wavelength
                    flux_on_filter = flux_interp(filter_wavelength)     # flux resampled to filter grid
                    integral_output = np.abs(np.trapz((lam_on_filter * flux_on_filter), filter_wavelength))
                    luminosity = integral_output / integral_filter
                    #print("integral output", integral_output, "integral filter", integral_filter, " = luminosity:", luminosity)
                    #output = call_int(lambda_values, flux_values)
                    #database.append((zams_mass, curr_age, output))
                    database.append((zams_mass, curr_age, luminosity, file_name))
                #now set up for the next trackpoint
                lamda_array = []
                flux_array = []
                zams_mass = float(lines[i+1].strip().split()[3])
                unique_mass.append(zams_mass)
                curr_age = float(lines[i+4].strip().split()[2])
                unique_age.append(curr_age)
                i += 10
            else:
                curr_line = lines[i].strip()
                lambda_x = float(curr_line.split()[0])
                flux = float(curr_line.split()[1])
                #need to account for redshift in wavelength: lambda = lambda * (1 + redshift)
                lamda_array.append((lambda_x * (1 + redshift)))
                flux_array.append(flux)
            i += 1
        #add last iteration of track point to database
        lambda_values = np.array(lamda_array)
        flux_values = np.array(flux_array)
        filter_wavelength, filter_output = get_filter(filter_name)
        integral_filter = np.abs(np.trapz(filter_output * filter_wavelength, filter_wavelength))
        #shorten lambda_values to be over filter_wavelength
        lam_min = filter_wavelength.min()
        lam_max = filter_wavelength.max()
        mask = (lambda_values >= lam_min) & (lambda_values <= lam_max)
        lam_sub = lambda_values[mask]
        flux_sub = flux_values[mask]
        # Evaluate interpolated values at filter_wavelength
        lam_interp = interp1d(lam_sub, lam_sub, kind='linear', bounds_error=False, fill_value='extrapolate')
        flux_interp = interp1d(lam_sub, flux_sub, kind='linear', bounds_error=False, fill_value='extrapolate')
        lam_on_filter = lam_interp(filter_wavelength)        # essentially == filter_wavelength
        flux_on_filter = flux_interp(filter_wavelength)     # flux resampled to filter grid
        integral_output = np.abs(np.trapz((lam_on_filter * flux_on_filter), lam_on_filter))
        print(integral_output)
        luminosity = integral_output / integral_filter
        #print("integral output", integral_output, "integral filter", integral_filter, " = luminosity:", luminosity)
        database.append((zams_mass, curr_age, luminosity, file_name))

    masses = np.unique(unique_mass).astype(float)
    ages = np.unique(unique_age).astype(float)
    grid = np.full((len(masses), len(ages)), np.nan)
    print(database[0])
    # Map data points to the grid
    i = 0
    for mass, age, value, name in database:
        mass_matches = np.where(np.isclose(masses, mass, rtol=1e-5))[0]
        age_matches = np.where(np.isclose(ages, age, rtol=1e-5))[0]
        mass_idx = mass_matches[0]
        age_idx = age_matches[0]
        print("i:", i, "mass_idx:,", mass_idx, "age_idx", age_idx, "value:", value)
        grid[mass_idx, age_idx] = value
        i += 1
    print(i)
    grid = np.nan_to_num(grid, nan=0.0) #turning all the Nan's to 0.0's
    return database
    '''
    #create the interpolator    
    #interpolator = RegularGridInterpolator((masses, ages), grid, method='linear', bounds_error=True, fill_value=np.nan)
    interpolator = RegularGridInterpolator((masses, ages), grid, method='linear', bounds_error=True, fill_value=0.0)
    #example query point
    query_point = (10, 1493000.0)
    result = interpolator(query_point)
    print("success")
    print("Interpolated result:", result)'''


def mus_tester(redshift, filter_name):
    """
    Test function that computes AB magnitudes for each data point in a flux database
    at a given redshift and filter, and prints out those at age == 0.

    Parameters
    ----------
    redshift : float
        Redshift value for which to compute AB magnitudes.
    filter_name : str
        The JWST filter name to use for the test (e.g., "F070W").
    """
    flux_database = get_flux(redshift=redshift, filter_name=filter_name)
    ab_magnitude_data = []
    for entry in flux_database:
        luminosity = entry[2]
        stellar_age = entry[1]
        filename = entry[3]
        ab_magnitude = convert_lum_to_magAB(luminosity, redshift, filter_name)
        ab_magnitude_data.append((stellar_age, ab_magnitude, filename))

    # Print results where stellar_age == 0
    for stellar_age, ab_magnitude, filename in ab_magnitude_data:
        if stellar_age == 0:
            print((stellar_age, ab_magnitude, filename))


def convert_lum_to_magAB(luminosity, redshift, filter_name):
    luminosity = luminosity * u.erg / u.s / u.AA
    #ΩM=0.3, ΩΛ=0.7, H0=70 km s-1 Mpc-1 cosmology
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    d_l = cosmo.luminosity_distance(redshift).to(u.cm) # distance luminosity 
    f_lambda = (luminosity / (4 * np.pi * d_l * d_l)).to(u.erg / (u.s * u.cm**2 * u.AA))
    pivot_lambda = get_pivot_lambda(filter_name) * u.AA
    f_nu = ((f_lambda * (pivot_lambda ** 2)) / const.c).to(u.erg / (u.s * u.cm**2 * u.Hz))
    m_AB = -2.5 * np.log10(f_nu.value) - 48.60
    return m_AB

def get_pivot_lambda(filter_name):
    """
    Returns the pivot wavelength (in microns) for a given JWST NIRCam filter.

    Parameters
    ----------
    filter_name : str
        The name of the filter (e.g., 'F070W', 'F200W', ...).

    Returns
    -------
    float
        Pivot wavelength in microns if found, otherwise raises KeyError.
    """
    # Use a constant for the pivot values dictionary to avoid redefining it each call
    # The dictionary could also be moved outside the function for re-use and speed, but is kept here for encapsulation.
    PIVOT_VALUES = {
        "F070W": 0.705,
        "F090W": 0.902,
        "F115W": 1.154,
        "F140M": 1.405,
        "F150W": 1.501,
        "F162M": 1.627,
        "F164N": 1.645,
        "F150W2": 1.672,
        "F182M": 1.845,
        "F187N": 1.874,
        "F200W": 1.988,
        "F210M": 2.096,
        "F212N": 2.121,
        "F250M": 2.503,
        "F277W": 2.776,
        "F300M": 2.996,
        "F322W2": 3.247,
        "F323N": 3.237,
        "F335M": 3.362,
        "F356W": 3.565,
        "F360M": 3.623,
        "F405N": 4.053,
        "F410M": 4.083,
        "F430M": 4.281,
        "F444W": 4.402,
        "F460M": 4.630,
        "F466N": 4.654,
        "F470N": 4.708,
        "F480M": 4.817
    }
    try:
        return PIVOT_VALUES[filter_name]
    except KeyError:
        raise ValueError(f"Filter name '{filter_name}' not recognized. "
                         f"Available filters are: {', '.join(PIVOT_VALUES.keys())}")
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
#import pandas as pd
from importlib.resources import files
import os

def get_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]

def construct_lookup_table (massFiles, filterSelecto, zr):

    numberOfMasses = readNumMasses(massFiles)
    numberOfMasses = len(get_files(".data/muspelheim_files/"))
    numberOfAges = readNumberOfAges()
    numberOfAges

    lookuptable = np.zeros((numberOfMasses, numberOfAges))

    lamdaList = getLamdaList()
    filter = get_filter(filter_name)

    for m in range(numberOfMasses):
        for n in range(numberOfAges):
            lookuptable[m][n] = computeFlux(lamdaList, filter, n, m)

    return lookuptable
"""
def make_table(z, filter_name, tp):  aj

    L, T = get_filter(filter_name)
    F = get_flux()
    age, mass = get_age_and_mass(tp)


    lookup_table = np.empty((len(age)+1, len(F)))
    lookup_table = pd.DataFrame(test_table, index=age, columns=mass)

    for i in age:
        for j in mass:
            lookup_table[i][j] = get_average_flux(L,T,F,z)
"""
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

def compute_luminosity(lowerBound, upperBound, lamd_datapoints, T_datapoints):

    lamd_func, T_func = make_vectors_similar(lamd_datapoints, T_datapoints)

    # make new function
    top_integrand = lambda x: np.dot(lamd_func(x), T_func(x), x)
    bottom_integrand = lambda x: np.dot(x, lamd_func(x))
    
    filter_flux = integrate_function(top_integrand, lowerBound, upperBound)
    flux = integrate_function(bottom_integrand, lowerBound, upperBound)

    return filter_flux / flux

def make_vectors_similar(vectorA, vectorB):
    """
    A_min = min(vectorA[0])
    B_min = min(vectorB[0])
    A_max = max(vectorA[0])
    B_max = max(vectorB[0])
    A_del = vectorA[0][1]-vectorA[0][0]
    B_del = vectorB[0][1]-vectorB[0][0]

    min = max(A_min, B_min)
    max = min(A_max,B_max)
    delta = min(A_del, B_del)

    lamd = list(range(min, max+delta, delta))

    A_interp = interp1d(vectorA[0], vectorA[1])
    B_interp = interp1d(vecotrB[0], vectorB[1])

    newA = A_interp(lamd)
    newB = B_interp(lamd)
    """
        

def integrate_function(func, a, b, num_points=1000):
    """
    Compute the definite integral of a function using the trapezoidal rule.
    
    Parameters:
    -----------
    func : callable
        The function to integrate
    a : float
        Lower bound of integration
    b : float
        Upper bound of integration
    num_points : int, optional
        Number of points to use (default: 1000)
    
    Returns:
    --------
    float
        The approximate value of the definite integral
    """ 
    x = np.linspace(a, b, num_points)
    y = func(x)
    return np.trapz(y, x)

def get_flux(redshift, filter_name):
    
    all_museplheim_files = get_files('./data/muspelheim_files')

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
                    flux_on_filter = flux_interp(filter_wavelength)      # flux resampled to filter grid
                    integral_output = np.abs(np.trapz((lam_on_filter * flux_on_filter * filter_output), lam_on_filter))
                    luminosity = integral_output / integral_filter
                    #output = call_int(lambda_values, flux_values)
                    #database.append((zams_mass, curr_age, output))
                    database.append((zams_mass, curr_age, luminosity))
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
        lam_interp = interp1d(lam_sub, lam_sub, kind='linear', bounds_error=False, fill_value='extrapolate')
        flux_interp = interp1d(lam_sub, flux_sub, kind='linear', bounds_error=False, fill_value='extrapolate')
        # Evaluate interpolated values at filter_wavelength
        lam_on_filter = lam_interp(filter_wavelength)        # essentially == filter_wavelength
        flux_on_filter = flux_interp(filter_wavelength)      # flux resampled to filter grid
        integral_output = np.abs(np.trapz((lam_on_filter * flux_on_filter * filter_output), lam_on_filter))
        luminosity = integral_output / integral_filter
        database.append((zams_mass, curr_age, luminosity))

    masses = np.unique(unique_mass).astype(float)
    ages = np.unique(unique_age).astype(float)
    grid = np.full((len(masses), len(ages)), np.nan)
    print(database[0])
    # Map data points to the grid
    i = 0
    for mass, age, value in database:
        mass_matches = np.where(np.isclose(masses, mass, rtol=1e-5))[0]
        age_matches = np.where(np.isclose(ages, age, rtol=1e-5))[0]
        mass_idx = mass_matches[0]
        age_idx = age_matches[0]
        print("i:", i, "mass_idx:,", mass_idx, "age_idx", age_idx, "value:", value)
        grid[mass_idx, age_idx] = value
        i += 1
    #create the interpolator
    #interpolator = RegularGridInterpolator((masses, ages), grid, method='linear', bounds_error=True, fill_value=np.nan)
    #query_point = (30, 0)
    #result = interpolator(query_point)

    #print("Interpolated result:", result)
import numpy as np
import pandas as pd
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
def make_table(z, filter_name, tp):  

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

    """
    Return two nparrays 
continue this later)   
    """
    filter_data = np.loadtxt(
        str(
            files("silmaril.data.mean_throughputs").joinpath(
                filter_name + "_mean_system_throughput.txt"
            )
        ),
        dtype = float,
        skiprows=1,
    )
    return filter_data[:, 0], filter_data[:,1]

def compute_lumonosity(lowerBound, upperBound, lamd_datapoints, T_datapoints):

    lamd_func, T_func = make_vectors_similar(lamd_datapoints, T_datapoints)

    # make new function
    top_integrand = lambda x: np.dot(lamd_func(x), T_func(x), x)
    bottom_integrand = lambda x: np.dot(x, lamd_func(x))
    
    filter_flux = integrate_function(top_integrand, lowerBound, upperBound)
    flux = integrate_function(bottom_integrand, lowerBound, upperBound)

    return filter_flux / flux

def make_vectors_similar(vectorA, vectorB):
    pass

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
                    #output = call_int(lambda_values, flux_values)
                    #database.append((zams_mass, curr_age, output))
                    database.append((zams_mass, curr_age, lambda_values, flux_values))
                #now set up for the next trackpoint
                lamda_array = []
                flux_array = []
                zams_mass = lines[i+1].strip().split()[3]
                curr_age = lines[i+4].strip().split()[2]
                i += 10
            else:
                curr_line = lines[i].strip()
                lambda_x = curr_line.split()[0]
                flux = curr_line.split()[1]
                #need to account for redshift in wavelength: lambda = lambda * (1 + redshift)
                lamda_array.append((lambda_x * (1 + redshift)))
                flux_array.append(flux)
            i += 1
        #add last iteration of track point to database
        lambda_values = np.array(lamda_array)
        flux_values = np.array(flux_array)
        database.append((zams_mass, curr_age, lambda_values, flux_values))
    lambda_count = 0
    flux_count = 0
    count = 0
    for tp in database:
        lambda_count += len(tp[2])
        flux_count += len(tp[3])
        count += 1
    print(lambda_count) #should be 12538713
    print(flux_count) #should be 12538713
    print(count)
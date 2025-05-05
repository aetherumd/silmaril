import numpy as np
import pandas as pd
import os
from importlib.resources import files

def get_filter(filter_name: str = "None"):
    try:
        filter_data = np.loadtxt(
            str(files("silmaril.data.mean_throughputs").joinpath(filter_name + "_mean_system_throughput.txt")), 
            dtype = float,
            skiprows=1,
        )
        filter_data[:,0] = [x*(10**4) for x in filter_data[:,0]]
        return filter_data[:, 0], filter_data[:,1]
    except FileNotFoundError:
        print(("File silmaril\\data\\mean_throughputs\\"+ filter_name + "_mean_system_throughput.txt does not exist"))
        pass

def get_masses(path:str = "Muspelheim"):
    file_list = get_files(path)
    return [int(f[17:20]) for f in file_list]

def get_ages(path:str = "Muspelheim"):
    file_list = get_files(path)
    ages = []
    temp = []
    del_t = float('inf')
    for f in file_list:
        with open(path+"\\"+f) as file:
            for line in file:
                if line.startswith("Age"):
                    temp.append(float(line[10:-1]))
        ages.append(temp)
        temp = []

    t_max = max([age for age_list in ages for age in age_list])
    t_min = min([age for age_list in ages for age in age_list])

    for i in range(len(ages)):
        for j in range(len(ages[i])-1):
            temp = ages[i][j+1] - ages[i][j]
            if (temp < del_t):
                del_t = temp
    return ages, t_max, t_min, del_t

def get_files(path:str = "Muspelheim"):
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    del file_list[0]
    return file_list

def get_flux(filter_data: np.array, masses, t_max, t_min, del_t, path:str = "Muspelheim",):

    file_list = get_files("Muspelheim")
    masses_len = len(masses)
    ages = list(range(t_min,t_max+del_t,del_t))
    ages_len = len(ages)
    file_lengths = np.zeros(masses)

    if(len(file_list) != 0):
        for index, f in enumerate(file_list):
            with open(("Muspelheim\\" + f), 'r') as file:
                for line in file:
                    file_lengths[index] += 1
    
        for index, f in enumerate(file_list):
            for i in range(12, file_lengths[index]-1997, 20009):
                for j in range(20009, file_lengths[index], 20009):
                    flux_data = np.loadtxt(
                        str(files("Muspelheim").joinpath(f)),
                        dtype = float,
                        skiprows=i,
                        skip_footer=j
                    )
                    break
        flux_lists = np.empty((masses_len, ages_len))
    return flux_lists



def make_table(z, filter_name):
    # 750 ages
    # ___ masses
    L, T = get_filter(filter_name)
    F = get_flux()

    lookup_table = np.empty((len(age), len(F)))
    # lookup_table = pd.DataFrame(test_table, index=age, columns=mass)

    for i in age:
        for j in mass:
            lookup_table[i][j] = integrate_function(L,T,F,z)

    

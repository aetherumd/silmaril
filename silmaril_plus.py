import numpy as np
import pandas as pd
from importlib.resources import files

def get_filter(filter_name: str = "None"):
    try:
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
    except FileNotFoundError:
        print(("File silmaril\\data\\mean_throughputs\\"+ filter_name + "_mean_system_throughput.txt does not exist"))
        pass

def get_flux(mass: float, age: float):
    
    try:
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
    except FileNotFoundError:
        print(("File silmaril\\data\\mean_throughputs\\"+ filter_name + "_mean_system_throughput.txt does not exist"))
        pass



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

    

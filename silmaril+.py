import numpy as np
import pandas as pd
from importlib.resources import files

def get_filter(filter_name: str = "None"):
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

def make_table(z, filter_name, tp):

    L, T = get_filter(filter_name)
    F = get_flux()
    age, mass = get_age_and_mass(tp)


    lookup_table = np.empty((len(age)+1, len(F)))
    lookup_table = pd.DataFrame(test_table, index=age, columns=mass)

    for i in age:
        for j in mass:
            lookup_table[i][j] = get_average_flux(L,T,F,z)

    
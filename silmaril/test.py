from mus_test import *
import matplotlib.pyplot as plt
#get_flux(9, "F070W")
grid, masses, ages = get_flux(5, "F150W")
#plot_grid2(grid=grid, masses=masses, ages=ages)
filled_grid = fill_out_grid(grid=grid)
#plot_grid2(grid, masses, ages)

#print(grid_interpolator(grid, masses, ages, (90, 1.09086e+06)))
#print(grid_interp3(grid, masses, ages, (90, 1.09086e+06)))
#print(grid)
#AB at z=10 for 0 age: 38.092 for 250 solar masses -> we are getting 55.4542
'''for i in range(len(masses)):
    zams_mass = masses[i]
    interp_lums = []
    interp3_lums = []
    for age in ages:
        interp_lums.append(grid_interpolator(grid, masses, ages, (zams_mass, age)))
        #interp3_lums.append(grid_interp3(grid, masses, ages, (zams_mass, age)))
    lums = grid[i]
    for j in range(len(ages)):
        if lums[j] != interp_lums[j] and lums[j] != 0:
            print(i, ages[j], lums[j], interp_lums[j])
            x = 1
    print("index", i)
    plt.scatter(ages, interp_lums, label='interp1', alpha=0.4, c='blue')
    plt.scatter(ages, lums, label='Muspelheim', alpha=0.4, c='orange')
    #plt.scatter(ages, interp3_lums, label='interp3')
    plt.title("Age vs Lum for " + str(zams_mass))
    plt.xlabel('Ages (Log)')
    plt.ylabel('Luminosity (Log)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
'''
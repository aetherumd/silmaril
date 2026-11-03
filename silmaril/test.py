from mus_test import *
import matplotlib.pyplot as plt
from sample_IMF import *
ages, lum = parse_starburst99()
print(len(ages), len(lum))

grid, masses, ages = get_flux(5, "F150W")
print(len(masses), len(ages))
filled_grid = fill_out_grid(grid=grid)


#AB at z=10 for 0 age: 38.092 for 250 solar masses -> we are getting 55.4542

N=100000
res=sample_massive_stars(N,m_particle=10.0)
print(res)
print("min:", min(res), "max:", max(res))
lres=np.log10(np.array(res))
y,x_e=np.histogram(lres,bins=20,range=(np.log10(4),np.log10(200)))
x_c=(x_e[1:]+x_e[:-1])/2.0
plt.plot(x_c,y,'og')

plt.xscale('linear')
plt.yscale('log')
plt.title('IMF function')
plt.xlabel('Mass')
plt.ylabel('Count')
plt.show()
interp_lums = []

for star in res:
    interp_lums.append(grid_interpolator(grid, masses, ages, (star, 1e6)))

plt.figure(figsize=(8,6))
plt.scatter(res, interp_lums, alpha=0.5, label='Interpolated')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Masses')
plt.ylabel('Luminosity')
plt.title('Interpolated Luminosity at Age 10^6')
plt.legend()
plt.show()
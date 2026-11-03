from mus_test import *
import matplotlib.pyplot as plt
from sample_IMF import *

grid, masses, ages = get_flux(9, "F150W")
min_age = min(ages)
max_age = max(ages)
filled_grid = fill_out_grid(grid=grid)

timestamps, lum = parse_starburst99()
N=1000000
res=sample_massive_stars(N,m_particle=10.0)
new_res = [x for x in res if x > 100]
print("res size", len(res))
print("new res size", len(new_res))
plt.figure()

plt.hist(new_res, bins=10)   # adjust bins as needed
plt.xlabel("Stellar Mass (M☉)")
plt.ylabel("Number of Stars")
plt.title("IMF Sampled Stellar Mass Distribution")

plt.show()

#mask = (timestamps < 27545400.0) & (timestamps > 108728.0)
#print(len(timestamps[mask]))
#print("min:", min(timestamps), "max:", max(timestamps))
#print("mus min: 108728.0 max: 27545400.0")
valid_mask = (timestamps >= min_age) & (timestamps <= max_age)
valid_times = timestamps[valid_mask]

star_grid, time_grid = np.meshgrid(res, valid_times)
query_points = np.column_stack([star_grid.ravel(), time_grid.ravel()])

results = grid_interpolator(grid, masses, ages, query_points).reshape(len(valid_times), len(res))

sums = results.sum(axis=1)

total_lums = np.full(len(timestamps), np.nan)
total_lums[valid_mask] = np.log10(sums)

print(len(timestamps))
print(len(lum))
print(len(total_lums))
plt.plot(timestamps, total_lums, label='interp')
plt.plot(timestamps, lum, label='starburst')
plt.xscale('log')
plt.xlim(1e5, 1e9)
plt.xlabel('Time (yr)')
plt.axvline(x=max_age, color='red')
plt.axvline(x=min_age, color='red')
plt.ylabel('log(Luminosity)')
plt.legend()
plt.show()
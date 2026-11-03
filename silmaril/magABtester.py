import os
from mus_test import *

filepath = os.path.join(os.path.dirname(__file__), 'data/JWST_NIRCam_ABmag_SED_BoOST22_SMC_f575-100.smc_grid_interpolation.txt')
results = []
with open(filepath, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Skip empty lines and header lines
        if not line or line.startswith('Redshift'):
            continue
        
        parts = line.split()
        if len(parts) < 15:
            continue
        
        try:
            redshift = float(parts[0])
            age_yr   = float(parts[1])
            f150w    = float(parts[12])   # F150W is the 8th column (index 7)
            
            results.append((redshift, age_yr,f150w))
        except ValueError:
            continue
#print(len(results))
#print(results[0])
diff_results = []
for z in range(19,21):
    ages, mags = mus_tester(z, "F356W")
    print("startign comparison")
    i = 0
    diffs = []
    max_diff = 0
    for entry in results:
        if entry[0] != z:
            continue
        age = entry[1]
        if age <= 100000:
            print(age)
            continue
        mag_AB = -1
        for j in range(len(ages)):
            if ages[j] == age:
                mag_AB = mags[j]
        if mag_AB == -1:
            print(entry, i)
            break
        if entry[2] != mag_AB:
            diff = entry[2] - mag_AB
            max_diff = max(max_diff, abs(diff))
            #print("age:", age, "calculated:", mag_AB, "musp:", entry[2], "diff: ", diff)
            diffs.append(diff)
        i += 1
    #print(i, "comparisons")
    #print("Max difference for", z, "is:", abs(max_diff))
    diff_results.append((z,np.mean(diffs), max_diff))
for this_diff in diff_results:
    print("Avg Difference for redshift", this_diff[0], "is:", this_diff[1])
    print("Max Difference for redshift", this_diff[0], "is:", this_diff[2])
        
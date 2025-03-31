FILE_PATH = '/content/drive/MyDrive/Aether/Research Sub-Teams/Lensing'
from google.colab import drive
drive.mount('/content/drive')
file_name = FILE_PATH + "/SED_BoOST22_SMC_f575-100.smc_grid_interpolation.txt"

import matplotlib.pyplot as plt

try:
  with open(file_name, "r") as file:
    lines = file.readlines()  # Read all lines into a list

    # Get the second and third lines if they exist
except FileNotFoundError:
  print(f"Error: The file '{file_name}' was not found.")
except Exception as e:
  print(f"An error occurred: {e}")

'''first_line = lines[12].strip() if len(lines) > 1 else "No second line available"
lambda_x = first_line.split()[0]
flux = first_line.split()[1]
print(f"lambda_x: {lambda_x}, Flux: {flux}")'''
first_line = lines[0].strip()
#print(first_line)
#make a list of dicts for each point
#list[trackpoint] = dict of all points at that track point
#Note: because of indexing, all lines are shifted back one
#so line 1 = index 0
list_of_dicts = []
i = 11
#initialize an empty dictionary
curr_dict = {}
while i < len(lines):
  if lines[i].strip() == first_line:
    list_of_dicts.append(curr_dict)
    curr_dict = {}
    i += 10
  else:
    curr_line = lines[i].strip()
    lambda_x = curr_line.split()[0]
    flux = curr_line.split()[1]
    #now add to curr_dict
    curr_dict[lambda_x] = flux
  i += 1
list_of_dicts.append(curr_dict)
#print(list_of_dicts)
print(len(list_of_dicts))
'''for track_point in list_of_dicts:
  print(len(track_point))
  print(track_point) #should be a dict of all points in that track point'''
i = 0
for track_point in list_of_dicts:
  wavelength_values = list(track_point.keys())
  flux_values = list(track_point.values())
  plt.plot(wavelength_values, flux_values, marker='o', linestyle='-')
  # Add labels and title
  plt.xlabel('Wavelength (AA)')
  plt.ylabel('Flux (erg/s/AA)')
  plt.title("{i} Plot of Wavelength vs Flux")
  plt.xscale('log')
  plt.yscale('log')
  #plt.grid(True)
  plt.show()
  print("\n")
  i += 1
#work to make it Node based I think
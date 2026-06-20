import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel, uniform,randint
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

data = np.load("6-measure-heatmap.npz")
dis, denc, EOF2, Neg, Ree, Bures, gdis= data["dis"], data["denc"], data["EOF2"], data["Neg"],data["Ree"],data["Bures"], data["gdis"]
from scipy.stats import binned_statistic_2d

# Your data (lists or arrays)
Ree= np.nan_to_num(Ree, nan=0.0)

A = np.real(dis)
B = np.real(denc)
C = np.real(Bures)

# Number of mesh cells in each direction
nx, ny = 80, 80   # adjust resolution here

# Compute 2D bins with max/mean statistics
stat, x_edges, y_edges, binnumber = binned_statistic_2d(
    A, B, C,
    statistic='max',
    bins=[nx, ny]
)

plt.figure(figsize=(6,5))

        #Transpose because of how numpy stores axes
        
# plt.pcolormesh(x_edges, y_edges, stat.T, shading='auto',cmap='hot')

# plt.xlabel('REE',fontsize=20)
# plt.ylabel('Quantum Advantage',fontsize=20)

# plt.title('Min discord per mesh cell')
# plt.colorbar(label='Min $\delta_{in}$')

# plt.show()

####################################################################
######################## Plot one binn #############################
####################################################################

A_fixed = 0.2   #example value

# find which bin this A belongs to
i = np.digitize(A_fixed, x_edges) - 1
maxC_list = stat[i, :]   # all B bins at this A

y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
B_slice = y_centers      # matching B values


mask = ~np.isnan(maxC_list)
B_slice = B_slice[mask]
maxC_list = maxC_list[mask]


plot(B_slice, maxC_list,'o')


A_fixed = 0.3   #example value

i = np.digitize(A_fixed, x_edges) - 1
maxC_list = stat[i, :]   # all B bins at this A

y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
B_slice = y_centers      # matching B values


mask = ~np.isnan(maxC_list)
B_slice = B_slice[mask]
maxC_list = maxC_list[mask]


plot(B_slice, maxC_list,'o')

A_fixed = 0.4   #example value

i = np.digitize(A_fixed, x_edges) - 1
maxC_list = stat[i, :]   # all B bins at this A

y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
B_slice = y_centers      # matching B values


mask = ~np.isnan(maxC_list)
B_slice = B_slice[mask]
maxC_list = maxC_list[mask]


plot(B_slice, maxC_list,'o')
xlabel('Advantage',fontsize=20) 
ylabel('max E',fontsize=20)
plt.show()
            
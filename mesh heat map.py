import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel, uniform,randint
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

start_time = time.time()

#-----------------------basis--------
bra0 = np.zeros([1,2],dtype = complex)
bra0[0][0]=1

ket0 = np.zeros([2,1],dtype = complex)
ket0[0][0]= 1

bra1 = np.zeros([1,2],dtype = complex)
bra1[0][1]= 1

ket1 = np.zeros([2,1],dtype = complex)
ket1[1][0]= 1

ket00 = np.kron(ket0, ket0)
bra00 = np.kron(bra0, bra0)
ket11 = np.kron(ket1, ket1)
bra11 = np.kron(bra1, bra1)

#----------------------------------------
sigmay = np.zeros([2,2] , dtype = complex)
sigmay[0][1] = 0-1j
sigmay[1][0] = 0+1j
#----------------identity matrices--------

ii = np.identity(4)
i1 = np.identity(2)

#---------------------------------------------------------

eta = 0.5  #reflectivity
p0 = 0.5 #probability of presence of the object
p1 = 1 - p0

l=180 #range for generating states
dis =[]
dis2=[]
denc=[]
c1ss = []
c2ss = []
c3ss=[]
EOF2=[]
cs=[]
dencw=[]# Werner states
disw=[]
Neg=[]
Ree=[]
Bures=[]
gdis=[]
for i11 in range(-l,l+1):
    for i2 in range(-l,l+1):
        for i3 in range(-l,l+1):


            

            c1  = i11/l
            c2 = i2/l
            c3 = i3/l
            if (1-c1-c2-c3)>=0 and (1-c1+c2+c3)>=0 and (1+c1-c2+c3)>=0 and (1+c1+c2-c3)>=0:
                bb= np.zeros([4,4])
                bb[0][0] = 1 + c3
                bb[0][3] = c1 - c2
                bb[3][0] = c1 - c2
                bb[1][1] = 1 - c3
                bb[1][2] = c1 + c2
                bb[2][1] = c1 + c2
                bb[2][2] = 1 - c3
                bb[3][3] = 1 + c3
                
                
                #states.append( 1/4 * bb)
                #cs.append([c1,c2,c3])
                bb =0.25*bb

                #eigenvalues
                lam1=0.25*(1-c1-c2-c3) 
                lam2=0.25*(1-c1+c2+c3)
                lam3=0.25*(1+c1-c2+c3)
                lam4=0.25*(1+c1+c2-c3)   

# Entanglement of formation------------------------------

                rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bb)) , np.kron(sigmay, sigmay))
                
                e1 , e2 = np.linalg.eig(np.dot(bb,rhotild))
                
                e1.sort()
                
                C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
                
                if ((1 - np.sqrt( 1 - C**2 ))/2)>10**-10:
                    ew=((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                                -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
                    EOF2.append(ew)
                #else:
                    #EOF2.append(0)
                    
# Negativity ---------------------------------------
                    Lam = max(lam1,lam2,lam3,lam4)

                    Neg.append(max(0.0, Lam - 0.5))
# Relative entropy of entanglement------------------

                    #Ree.append(1+Lam*math.log2(Lam) + (1-Lam)*math.log2(1-Lam+10**-14))
                    if Lam > 0.5:
                        Ree.append(
                            1
                            + Lam*np.log2(Lam)
                            + (1-Lam)*np.log2(1-Lam)
                        )
                    else:
                        Ree.append(0.0)
# Bures measure------------------------------------

                    if Lam > 0.5:
                        Fmax = 0.5 + np.sqrt(Lam*(1-Lam))
                        Bures.append(np.sqrt( 2 * (1 - np.sqrt(Fmax)) ))
                    else:
                        Bures.append(0.0)

# Geometric discord---------------------------------
                    Cmax2 = max(abs(c1)**2,abs(c2)**2,abs(c3)**2)

                    gdis.append(0.25*(c1**2+c2**2+c3**2 - Cmax2))

# Initial discord-------------------------------------
                
                 
                    ccc  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                            (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                            + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                            (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
                    
                    
                    sm = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
                          + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                          + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
                          + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))
    
        
                   
                    qqq = 2 + sm - ccc
                    
                    c1ss.append(c1)
                    c2ss.append(c2)
                    c3ss.append(c3)
                    
                    
    #discord after noise----------------------------------------
                    c1=eta*c1
                    c2=eta*c2
                    c3=eta*c3
    
                    ccc1  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                            (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                            + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                            (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
                    
                    
                    sm1 = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
                          + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                          + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
                          + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))
                    
                    qqq1 = 2 + sm1 - ccc1
                    
    ##discord of average state----------------------------------------
                    
                    c1=p0*c1
                    c2=p0*c2
                    c3=p0*c3
    
    
                    ccc2  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                            (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                            + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                            (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
                    
                    
                    sm2 = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
                          + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                          + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
                          + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))
                    
                    qqq2 = 2 + sm2 - ccc2
                    
                    
                    dis.append(qqq)
                    #EOF2.append(ew)
                    denc.append((p0*qqq1 - qqq2))



np.savez("6-measure-heatmap.npz", dis=dis, denc=denc, EOF2=EOF2, Neg=Neg, Ree=Ree, Bures=Bures, gdis=gdis)

from scipy.stats import binned_statistic_2d

# Your data (lists or arrays)
A = np.real(dis)
B = np.real(denc)
C = np.real(EOF2)

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
        
plt.pcolormesh(x_edges, y_edges, stat.T, shading='auto',cmap='hot')

plt.xlabel('Entanglement of formation',fontsize=20)
plt.ylabel('Quantum Advantage',fontsize=20)

plt.title('Min $\delta_{in}$ per mesh cell')
plt.colorbar(label='Min $\delta_{in}$')

plt.show()

####################################################################
######################## Plot one binn #############################
####################################################################

#A_fixed = 0.2   #example value

#find which bin this A belongs to
#i = np.digitize(A_fixed, x_edges) - 1
#maxC_list = stat[i, :]   # all B bins at this A
#print(maxC_list)

#y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
#B_slice = y_centers      # matching B values


#mask = ~np.isnan(maxC_list)
#B_slice = B_slice[mask]
#maxC_list = maxC_list[mask]


#plot(B_slice, maxC_list,'o')
#xlabel('Advantage',fontsize=20) 
#ylabel('Entanglement of formation',fontsize=20)

#A_fixed = 0.3   #example value

#i = np.digitize(A_fixed, x_edges) - 1
#maxC_list = stat[i, :]   # all B bins at this A

#y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
#B_slice = y_centers      # matching B values


#mask = ~np.isnan(maxC_list)
#B_slice = B_slice[mask]
#maxC_list = maxC_list[mask]


#plot(B_slice, maxC_list,'o')
#xlabel('Advantage',fontsize=20) 
#ylabel('Entanglement of formation',fontsize=20)

#A_fixed = 0.4   #example value

#i = np.digitize(A_fixed, x_edges) - 1
#maxC_list = stat[i, :]   # all B bins at this A

#y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
#B_slice = y_centers      # matching B values


#mask = ~np.isnan(maxC_list)
#B_slice = B_slice[mask]
#maxC_list = maxC_list[mask]


#plot(B_slice, maxC_list,'o')
#xlabel('Advantage',fontsize=20) 
ylabel('Entanglement of formation',fontsize=20)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
                


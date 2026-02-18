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

l=100 #range for generating states
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
#------------------------------------


# Entanglement of formation
                rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bb)) , np.kron(sigmay, sigmay))
                
                e1 , e2 = np.linalg.eig(np.dot(bb,rhotild))
                
                e1.sort()
                
                C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
                
                if ((1 - np.sqrt( 1 - C**2 ))/2)>10**-15:
                    EOF2.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                                -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
                else:
                    EOF2.append(0)
                    



# Initial discord-------------------------------------
                
                #initial discord
                 
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
                denc.append((p0*qqq1 - qqq2))
                
# alpha states--------------------------------
eofw=[]    
for p in range(1000):
    alphaa = p/1000
    c1 = alphaa +10**(-10)
    c2 = -alphaa+10**(-10)
    c3 = 2*alphaa-1 +10**(-10)
    bb= np.zeros([4,4])
    bb[0][0] = 1 + c3
    bb[0][3] = c1 - c2
    bb[3][0] = c1 - c2
    bb[1][1] = 1 - c3
    bb[1][2] = c1 + c2
    bb[2][1] = c1 + c2
    bb[2][2] = 1 - c3
    bb[3][3] = 1 + c3
    bb=0.25*bb
# Entanglement of formation
    rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bb)) , np.kron(sigmay, sigmay))
    
    e1 , e2 = np.linalg.eig(np.dot(bb,rhotild))
    
    e1.sort()
    
    C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
    
    if ((1 - np.sqrt( 1 - C**2 ))/2)>10**-15:
        eofw.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                    -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
    else:
        eofw.append(0)
                        


    ccc  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
            (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
            + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
            (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
    
    
    sm = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
          + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-10) 
          + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
          + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))


   
    qqq = 2 + sm - ccc
    
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
    dencw.append((p0*qqq1 - qqq2))
    disw.append(qqq)
    
# beta-states-----------------------------------------
eofb=[] 
disb=[]
dencb=[]   
for p in range(1000):
    beta = p/1000
    c1 = 1
    c2 = 1-2*beta-10**(-10)
    c3 = 2*beta-1 +10**(-10)
    bb= np.zeros([4,4])
    bb[0][0] = 1 + c3
    bb[0][3] = c1 - c2
    bb[3][0] = c1 - c2
    bb[1][1] = 1 - c3
    bb[1][2] = c1 + c2
    bb[2][1] = c1 + c2
    bb[2][2] = 1 - c3
    bb[3][3] = 1 + c3
    bb=0.25*bb
# Entanglement of formation
    rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bb)) , np.kron(sigmay, sigmay))
    
    e1 , e2 = np.linalg.eig(np.dot(bb,rhotild))
    
    e1.sort()
    
    C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
    
    if ((1 - np.sqrt( 1 - C**2 ))/2)>10**-15:
        eofb.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                    -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
    else:
        eofb.append(0)
                        


    ccc  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
            (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
            + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
            (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
    
    
    sm = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
          + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-10) 
          + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
          + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))


   
    qqq = 2 + sm - ccc
    
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
    dencb.append((p0*qqq1 - qqq2))
    disb.append(qqq)
    
eofwe=[] 
diswe=[]
dencwe=[]   
for p in range(1000):
    
    c1 = -p/1000 + 10**(-10)
    c2 = -p/1000 + 10**(-10)
    c3 = -p/1000 + 10**(-10)
    bb= np.zeros([4,4])
    bb[0][0] = 1 + c3
    bb[0][3] = c1 - c2
    bb[3][0] = c1 - c2
    bb[1][1] = 1 - c3
    bb[1][2] = c1 + c2
    bb[2][1] = c1 + c2
    bb[2][2] = 1 - c3
    bb[3][3] = 1 + c3
    bb=0.25*bb
# Entanglement of formation
    rhotild = np.dot(np.dot(np.kron(sigmay, sigmay) ,np.conjugate(bb)) , np.kron(sigmay, sigmay))
    
    e1 , e2 = np.linalg.eig(np.dot(bb,rhotild))
    
    e1.sort()
    
    C = max(0 , np.sqrt(e1[3]) -np.sqrt( e1[2]) - np.sqrt(e1[1]) - np.sqrt(e1[0]))
    
    if ((1 - np.sqrt( 1 - C**2 ))/2)>10**-15:
        eofwe.append((-((1 + np.sqrt( 1 - C**2 ))/2) * math.log2((1 + np.sqrt( 1 - C**2 ))/2)) 
                    -((1 - np.sqrt( 1 - C**2 ))/2) * math.log2((1 - np.sqrt( 1 - C**2 ))/2))
    else:
        eofwe.append(0)
                        


    ccc  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
            (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
            + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
            (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
    
    
    sm = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
          + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-10) 
          + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
          + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))


   
    qqq = 2 + sm - ccc
    
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
    dencwe.append((p0*qqq1 - qqq2))
    diswe.append(qqq)
    
# approximation near 0
apdis=[]
apdenc=[]
for i in range(100):
    apdis.append(i/100)
    apdenc.append((1/16)*(i/100))

plot(EOF2 , dis ,'o',alpha=0.6,markersize=2)
#plot(EOF2,denc,'o', alpha=0.6,linewidth=3.5)
plot(eofw, disw)
plot(eofb, disb)
#plot(eofw, dencw)
#plot(eofb, dencb)
plot(eofwe, diswe)
ylabel('Quantum discord',fontsize=30)
xlabel('Entanglement of formation',fontsize=30)
xlim(0, )
ylim(0, )
plt.legend(['MMM-states',r'$\alpha$-states',r'$\beta$-states','Werner-states','approximation'])
plt.rc('font', size=20) 

plt.show()
#discords = np.zeros(len(dis))           
#c1s = np.zeros(len(dis))           
#c2s = np.zeros(len(dis))  
#c3s = np.zeros(len(dis))           
         
#for i in range(len(discords)):
 #   c1s[i] = c1ss[i] #- c1ss2[i] 
  #  c2s[i] = c2ss[i] #- c2ss2[i] 
   # c3s[i] = c3ss[i]
    #discords[i] = dis[i] #- discordss1[i]

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#img = ax.scatter(c1s , c2s , c3s , c=discords , cmap=plt.hot() , s=50 , vmin = 0 , vmax= 0.06 )
#fig.colorbar(img)

#ax.set(xlabel=r'$c_1$', ylabel=r'$c_2$', zlabel=r'$c_3$')

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)
                


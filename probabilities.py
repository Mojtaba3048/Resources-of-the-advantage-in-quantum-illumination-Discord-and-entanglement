import numpy as np
from pylab import plot, xlim, ylim, xlabel, ylabel, uniform,randint
import math
import time
import matplotlib.pyplot as plt

et = []
ps = []
ps1=[]
ps2=[]
rel_perf=[]
for i in range(101):
    
    eta = i/100
    et.append(eta)
    
    
    #p = (2+7*eta + 3*eta**2)/(2+5*eta + 3*eta**2)
    p1=(1+3*eta)/(2+3*eta)
    p2=(1+eta)/(2+eta)
    #ps1.append(math.log2(p)/math.log2(p2))
    m1= p1
    m2=p2
    rel_perf
    #ps1.append(p1)
    rel_perf.append(m1)
    ps2.append(m2)

plot(et, rel_perf , color='green', label='joint')
plot(et, ps2 , color='blue' , label='local')
plt.legend()
xlabel(r'$\eta$' ,fontsize=15)
ylabel(r'$\log p(there|yes)$' ,fontsize=15)





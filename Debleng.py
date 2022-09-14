import pandas as pd
import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
#parameters
m0=9.109*1e-31
q=1.6*1e-19
K=1.38*1e-23
Ke=K/q
T=300.00
hbar=1.05*1e-34
vol=308.5974
conv=0.529177 #au to Ang
vol*=conv**3*1e-30 #volume in m3
b=4.0*1e10 #ioffe
d0=8.85*1e-12
dr=16.2
de=dr*d0
z=4
Ev=11.5377
Eg=3.39
Ec=Ev+Eg
Egexp=3.4
shf=Egexp-Eg
Ef=0.5*(Ec+Ev)
x=0.7 #energy window inside each band [Ev-x: Ec+x]
m=0
q0m=0
nimp=-1e24 #impurity concentration 1/m^3
print(f'carrier concentration is {nimp}')
imu=0
imin=math.log(abs(nimp),10)+1 #large
muimp=0.0 #chemical potential of impurity concentration
r=np.loadtxt("dos400", dtype='str').shape[0]
c=np.loadtxt("dos400", dtype='str').shape[1]
g=np.zeros((r,c))
g=np.loadtxt("dos400")
deltE=g[1,0]-g[0,0]
#chemical potential step equals energy step
mupt= math.floor((Egexp+2*x)/deltE) #number of chemical potential points
print(f'number of chemical potential point is {mupt}')
mu=np.zeros(mupt)
ne=np.zeros((mupt,2))
nh=np.zeros((mupt,2))
nt=np.zeros((mupt,2))
sig=np.zeros(3)
dos=np.zeros((g.shape[0],2))
#strongly screened
ni=np.zeros((r,2))
mob=np.zeros(r)
dndmu=0.0
#q0=np.zeros((r+math.floor(shf/deltE),2))
p=np.zeros((r+math.floor(shf/deltE),5)) #scattering rate
fd=np.zeros((mupt,dos.shape[0]))
print(f' rows {dos.shape[0]}')
print(f' columns {dos.shape[1]}')
dos[0,0]=g[0,0]
for i in range(0,dos.shape[0]):
    dos[i,0]=dos[0,0]+i*deltE
    if (dos[i,0] <= Ec):
        dos[i,1]=g[i,1]
    elif(dos[i,0]> Ec) and (dos[i,0] <=Ec+shf):
            dos[i,1]=0.00
            m+=1
    else:
        dos[i,1]=g[i-m,1]


##DOS plot
#plt.xlabel('$\mu$-$E_v$ (eV)',fontname="serif", fontsize=14,color='black',weight='normal')
#plt.ylabel('DOS ($1/eV$)', fontname="serif", fontsize=14,color='black',weight='bold')
#plt.plot(dos[:,0]-Ev,dos[:,1],color='blue',label='$SH$')
#plt.plot(g[:,0]-Ev,g[:,1],color='red',label='$DFT$')
#plt.xlim([-2.0, 4.0])
#plt.ylim([0.0, 1.0])
#plt.legend()
#plt.show()
np.savetxt('movdos.txt',dos, delimiter=' ')
#Chemical potential and fermi-dirac
for i in range(0,mupt):
    mu[i]=Ev-x+deltE*i
    for j in range(0,g.shape[0]):
        if dos[j,0] <= mu[i]+Ke*T*math.log(1e-4) :
            fd[i,j]=1
        elif dos[j,0] >= mu[i]+Ke*T*math.log(1e4) :
            fd[i,j]=0.00
        else:
            fd[i,j]=1/(1+math.exp((dos[j,0]-mu[i])/(Ke*T)))
muf=0
fmin=1e40
for i in range(0,mupt):
    nt[i,0]=mu[i]
    ne[i,0]=mu[i]
    nh[i,0]=mu[i]
    for j in range(0,dos.shape[0]-1):
        if dos[j,0]>=Ec+shf:
            ne[i,1]=ne[i,1]+0.5*(dos[j+1,0]-dos[j,0])*(dos[j,1]*fd[i,j]+dos[j+1,1]*fd[i,j+1])
        if dos[j,0]<=Ev:
            nh[i,1]=nh[i,1]+0.5*(dos[j+1,0]-dos[j,0])*(dos[j,1]*(1-fd[i,j])+dos[j+1,1]*(1-fd[i,j+1]))
        nt[i,1]=ne[i,1]+nh[i,1]
    if abs(ne[i,1]-nh[i,1])<=fmin:
        fmin=abs(ne[i,1]-nh[i,1])
        muf=mu[i]
print(f'intrinsic fermi level is {muf}')
ne[:,1]/=vol
nh[:,1]/=vol
nt[:,1]/=vol
np.savetxt('nmu.txt',nt, delimiter=' ')
np.savetxt('ne.txt',ne, delimiter=' ')
np.savetxt('nh.txt',nh, delimiter=' ')

#Corresponding mu to n_impurities
if nimp > 0:
    for i in range (0,mupt):
        if abs(math.log(abs(nimp-nh[i,1]),10))<imin:
            imin=abs(math.log(abs(nimp-nh[i,1]),10))
            muimp=ne[i,0]
else:
    nimp=abs(nimp)
    for i in range (0,mupt):
        if abs(math.log(abs(nimp-ne[i,1]),10))<imin:
            imin=abs(math.log(abs(nimp-ne[i,1]),10))
            muimp=ne[i,0]
            imu=i

print(f'ionized impurity chemical potential is {muimp}')
#screening length
for i in range (0,nt.shape[0]-1):
    if abs(muimp-nt[i,0])<deltE:
        dndmu=(nt[i+1,1]-nt[i-1,1])/(q*(nt[i+1,0]-nt[i-1,0]))

print(f'dndmu is {dndmu}')

LD4=math.sqrt(abs(de/(q**2*dndmu)))
print(f'LD poison is {LD4}')
LDsp4=math.sqrt(abs(de*K*T/(q**2*nimp)))
print(f'LD spherical is {LDsp4}')

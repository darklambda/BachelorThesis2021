# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import *
matplotlib.rcParams.update({'font.size': 18})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
from matplotlib.colors import LinearSegmentedColormap

path = 'Outputs'

def plotV(LX,LY,w,t,b,bb,w_min,w_max,x,y, median, minimum, maximum):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()#projection='3d')
    xlim = (np.min(x),np.max(x))
    ylim = (np.min(y),np.max(y))
    zlim = (w_min*0.999,w_min*1.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_zlim(zlim)
    land = np.where(bb==0, w_max, np.nan)
    water = np.where(bb==2, w, np.nan)
    landPlot = ax.contourf(x, y, land)
    #waterPlot = ax.plot_surface(x, y, water)
    waterPlot = ax.contourf(x, y, water, levels=np.linspace(minimum, maximum, 40), cmap='bwr')
    cbar = fig.colorbar(waterPlot)
    cbar.ax.set_ylabel('Water Level')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('t = '+str(t))
    plt.savefig('plot_'+str(t).rjust(5,'0')+'.png')
    plt.close()

def Read_Bat(name): # name = input file
    a = open(name+'.txt')
    li = a.readline().strip().split()
    dx,x0,y0 = map(float,li[2:])
    LX,LY = map(int,li[:2])
    a.close()
    b,w,bb = np.loadtxt(name+'.txt', delimiter=' ', skiprows=1, usecols=(0, 1, 2), unpack=True)
    b  = b.reshape((LY,LX))    
    x = x0+dx*np.arange(LX)    
    y = y0+dx*np.arange(LY)
    x,y = np.meshgrid(x,y)
    bb = bb.reshape((LY,LX))
    w = w.reshape((LY,LX))    
    wmi,wma = np.min(w),np.max(w)
    tmp = np.where(bb==2, w, np.nan)
    median = np.nanmedian(tmp)
    if wmi == wma:
        wma += 1
        wmi -= 1
    return x,y,b,w,bb,wmi,wma,LX,LY, median

def Read_Output(LX,LY,name):
    w =np.fromfile(name+".dat")
    w = w.reshape((LY,LX))
    return w
    
def updateBounds(bb, w, minimum, maximum):
	tmp = np.where(bb==2, w, np.nan)
	minTmp = np.nanmin(tmp)
	maxTmp = np.nanmax(tmp)
	if minTmp < minimum:
		minimum = minTmp
	if maxTmp > maximum:
		maximum = maxTmp
	return minimum, maximum

TMAX = 20000
dt = 200
print("Input")
inp = str(input("File Name: ")) #Nombre del archivo hasta antes de extensión
print("Reading {}.txt".format(inp))
name = inp
print("Reading File - Init")
x,y,b,w,bb,wmi,wma,LX,LY, median = Read_Bat(name)
print("Reading File - Done")
name = 'outputs_' + inp + '/output_'+str(0).rjust(5,'0')
w = Read_Output(LX,LY,name)
wmi = np.min(w)
wma = np.max(w)
landlvl = np.amax(w)

print('Median is equal to ', median)

minimum = np.inf
maximum = 0

print('Updating bounds of plotting...')
for t in range(0,TMAX+1,dt):
    name = 'outputs_' + inp + '/output_'+str(t).rjust(5,'0')
    w = Read_Output(LX,LY,name)
    minimum, maximum = updateBounds(bb, w, minimum, maximum)
    
print('Minimum and maximum water levels are ', minimum, ' and ', maximum)

for t in range(0,TMAX+1,dt):
    name = 'outputs_' + inp + '/output_'+str(t).rjust(5,'0')
    w = Read_Output(LX,LY,name)
    print("Graficando t =", t)
    plotV(LX,LY,w,t,b,bb,wmi,wma,x,y, median, minimum, maximum)
    
    

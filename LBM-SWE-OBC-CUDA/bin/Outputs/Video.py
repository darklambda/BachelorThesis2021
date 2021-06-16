#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import *
matplotlib.rcParams.update({'font.size': 28})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
from matplotlib.colors import LinearSegmentedColormap

fac = 1
path = 'Outputs'

def plotV(path, LX,LY,w,t,b,bb,w_min,w_max):
    colors = [(1/255.0, 100/255.0, 68/255.0), (230/255.0, 210/255.0, 120/255.0), (156/255.0, 5/255.0, 5/255.0),(155/255.0,155/255.0,155/255.0)]
    n_bin = 100 
    cmap_name = 'tetocolors'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    
    f, ax = plt.subplots(figsize=(7,12))
    ax.grid(True)    
    ax.set_xticks([])
    ax.set_yticks([])
    levels = np.linspace(-0.3, 0.3, 100)
    cs = ax.contour(range(LX), range(LY), w, levels=levels,  cmap="bwr", extend="both")
    #cs.cmap.set_under('b')
    #cs.cmap.set_over('r')
    levels2 = np.linspace(np.min(b[np.where(bb==0)])-1, np.max(b[np.where(bb==0)])+1, 2)
    cs2 = ax.contour(range(LX), range(LY), b, levels=levels2,  cmap=cm)
    #w[np.where(bb==0)] = np.nan
    #b[np.where(bb!=0)] = np.nan
    divider2 = make_axes_locatable(ax)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(cs, cax= cax2, ticks=[-0.3,-0.15,0.0,0.15,0.3])
    cbar2.set_label('Water level $[m]$', rotation=90, fontsize=32)
    ticklabs = cbar2.ax.get_yticklabels()
    cbar2.ax.set_yticklabels(ticklabs, fontsize=30)
    times= [str(int(int(0.5*t)/3600)).rjust(2,'0'),str(int(int(0.5*t)%3600/60)).rjust(2,'0'),str(int(0.5*t)%60).rjust(2,'0')]
    ax.set_title('$t = $ '+':'.join(times))
    plt.tight_layout()
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
    if wmi == wma:
        wma += 1
        wmi -= 1
    return x,y,b,w,bb,wmi,wma,LX,LY

def Read_Output(LX,LY,name):
    w =np.fromfile(name+".dat")
    w = w.reshape((LY,LX))
    return w

TMAX = 1000
dt = 200
name = '../../Inputs/Test_40000'
x,y,b,w,bb,wmi,wma,LX,LY = Read_Bat(name)
w0 = w[0,0]
wmin = []
wmax = []
for t in range(0,TMAX+1,dt):
    name = 'outputs_Test_40000/output_'+str(t).rjust(5,'0')
    w = Read_Output(LX,LY,name)
    w-=w0
    wmin.append(np.min(w))
    wmax.append(np.max(w))
wmi,wma = min(wmin), max(wmax)
for t in range(0,TMAX+1,dt):
    name = 'outputs_Test_40000/output_'+str(t).rjust(5,'0')
    w = Read_Output(LX,LY,name)
    w-=w0
    print("Graficando t =", t)
    plotV(path,LX,LY,w,t,b,bb,wmi,wma)    
    

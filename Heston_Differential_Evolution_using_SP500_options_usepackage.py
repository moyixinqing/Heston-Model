# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:42:51 2018

@author: renxi
"""

# Parameter estimation using loss functions and Fractional FFT
# Uses S&P500 options on April 13, 2012
import HestonDE
from scipy.optimize import differential_evolution
import numpy as np
from scipy.stats import norm
from math import log,sqrt,exp
# Black Scholes call
def BSC(s,K,rf,q,v,T):
    BSC = (s*exp(-q*T)*norm.cdf((log(s/K) + (rf-q+v**2/2)*T)/v/sqrt(T)) - K*exp(-rf*T)*norm.cdf((log(s/K) + (rf-q+v**2/2)*T)/v/sqrt(T) - v*sqrt(T)));
    return BSC    

## Load the SP500 data
MktIV = np.array([[27.8000,  26.3800,  25.3200,   25.1800],
           [24.7700,   24.0200,   23.6400,   23.6900],
           [21.8600,   21.5800,   22.0300,   22.3900],
           [18.7800,   19.3000,   20.4700,   20.9800],
           [15.7200,   17.1200,   18.9400,   19.7000],
           [13.3400,   15.1700,   17.4800,   18.4900],
           [13.2300,   13.7300,   16.1800,   17.3600]])/100;
T = np.array([45, 98, 261, 348])/365;
K = np.array([120,   125,   130,   135,   140,   145,   150]);
[NK,NT] = MktIV.shape
rf = 0.0010;
q  = 0.0068;
S = 137.14;
PutCall = np.tile('C',(NK,NT));


## Use ITM Calls and Puts in the parameter estimation
MktPrice=np.zeros((NK,NT))
for k in range(NK):
    for t in range(NT):
        MktPrice[k,t] = BSC(S,K[k],rf,q,MktIV[k,t],T[t]);

## Parameter Estimation settings
# Initial values for kappa,theta,sigma,v0,rho
start = [9.0, 0.05, 0.3, 0.05, -0.8];

# Specify the objective function and method to obtain the option price
ObjFun = 1;
Method = 1;

# Gauss Laguerre Weights and abscissas
[x, w] = HestonDE.GenerateGaussLaguerre(32);
trap = 1;

# Settings for the Bisection method to find the Model IV
a = .001;
b = 3;
Tol = 1e-7;
MaxIter = 1000;

# Estimation bounds
e = 1e-5;
kappaL = e;       kappaU = 20;
thetaL = e;       thetaU = 2;
sigmaL = e;       sigmaU = 2;
v0L    = e;       v0U    = 2;
rhoL   = -.999;   rhoU   = .999;
Hi = np.array([kappaU, thetaU, sigmaU, v0U, rhoU]);
Lo = np.array([kappaL, thetaL, sigmaL, v0L, rhoL]);

# =============================================================================
# # Gauss-Laguerre integration abscissas and weights
# [x, w] = GenerateGaussLaguerre(32);
# 
# =============================================================================
## Parameter Estimates using the FRFT
N = 2**7;
uplimit = 25;
eta = 0.25;
alpha = 1.75;
rule = 'T';
K1 = K[0]-1;
# =============================================================================
# tic
# FTparam = fmincon(@(p) HestonObjFunFRFT(p,S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule),start,[],[],[],[],Lo,Hi);
# t1 = toc;
# 
# =============================================================================
# In[]
## Differential Evolution algorithm
# Number of members in the population and number of generations
NP = 75;
NG = 300;

# Scale factor and crossover ratiob
F  = 0.8;
CR = 0.5;

# Find the Differential Evolution parameters
tic()
#DEparam = HestonDE.HestonDE(NG,NP,CR,F,Hi,Lo,S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule);
# alternative scipy
bounds=[(kappaL,kappaU), (thetaL,thetaU), (sigmaL,sigmaU), (v0L,v0U), (rhoL,rhoU)]
def obj(Pnew):
    return HestonDE.HestonObjFunFRFT(Pnew,S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule)
DEresult = differential_evolution(obj, bounds)
DEparam= DEresult.x

t2 = toc();

# In[]
## Generate prices and parameters using both sets of parameters
[x,w] = HestonDE.GenerateGaussLaguerre(32);
a = 0.001;
b = 5;
Tol = 1e-5;
MaxIter = 5000;
Sum1 = 0;
Sum2 = 0;

PriceDE=np.zeros((NK,NT))
IVDE=np.zeros((NK,NT))
error2=np.zeros((NK,NT))
for k in range(NK):
    for t in range(NT):
        PriceDE[k,t] = HestonDE.HestonPriceGaussLaguerre(PutCall[k,t],S,K[k],T[t],rf,q,DEparam,trap,x,w);
        IVDE[k,t] = HestonDE.BisecBSIV(PutCall[k,t],S,K[k],rf,q,T[t],a,b,PriceDE[k,t],Tol,MaxIter);
        if IVDE[k,t] != -1:
            error2[k,t] = (IVDE[k,t] - MktIV[k,t])**2;
        else:
            IVDE[k,t] = NaN;
            Sum2 = Sum2+1;

ErrorDE =  sum(sum(error2)) / (NK*NT - Sum2);

## Display the results

#fmtstring = [repmat('#10.4f',1,6) '#12.2d'];
print('--------------------------------------------------------------------------------\n')
print('Method      kappa     theta     sigma      v0        rho      time      IVMSE\n')
print('--------------------------------------------------------------------------------\n')

print('DE      {} {} {}\n'.format(DEparam,t2,ErrorDE))
print('--------------------------------------------------------------------------------\n')
  
# =============================================================================
# ## Plot the results
# for t=1:NT
# 	subplot(2,2,t)
# 	plot(K,MktIV(:,t),'kx:',K,IVFT(:,t),'kx-',K,IVDE(:,t),'ko-')
# 	legend('Market','Objective Function', 'Differential Evolution')
# end
# 
# =============================================================================















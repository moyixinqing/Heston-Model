# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:50:19 2018

@author: renxi
"""


# In[]  BisecBSIV
def BisecBSIV(PutCall,S,K,rf,q,T,a,b,MktPrice,Tol,MaxIter):
        # =============================================================================
        # The earlier code invokes theMatlab function BisecBSIV.m that uses the bisection
        # algorithm to find the implied volatilities once the prices are generated. The first part
        # of the function defines a function handle for calculating the price of calls under the
        # Black-Scholes model. The main part of the function calculates the Black-Scholes call
        # price at the midpoint of the two endpoints, and compares the difference between this
        # call price and themarket price. If the difference is greater than the user-specified tolerance,
        # the function updates one of the endpoints with the midpoint and passes through
        # the iteration again. To conserve space, parts of the function have been omitted.
        # =============================================================================
        # Bisection algorithm
        # PutCall = "P"ut or "C"all
        # S = Spot price
        # K = Strike price
        # rf = Risk free rate
        # T = Maturity
        # a = Starting point lower value of vol
        # b = Starting point upper value of vol
        # MktPrice = Market price
        # Tol = Tolerance
        # MaxIter = maximum number of iterations
    
        from scipy.stats import norm
        from math import exp,sqrt,log
        # Function for the Black Scholes call and put
        def BSC(s,K,rf,q,v,T):
            BSC=(s*exp(-q*T)*norm.cdf((log(s/K) + (rf-q+v**2/2)*T)/v/sqrt(T)) - K*exp(-rf*T)*norm.cdf((log(s/K) + (rf-q+v**2/2)*T)/v/sqrt(T) - v*sqrt(T)));
            return BSC
        def BSP(s,K,rf,q,v,T): 
            BSP= (K*exp(-rf*T)*norm.cdf(-(log(s/K) + (rf-q+v**2/2)*T)/v/sqrt(T) + v*sqrt(T)) - s*exp(-q*T)*norm.cdf(-(log(s/K) + (rf-q+v**2/2)*T)/v/sqrt(T)));
            return BSP
        if 'C' in PutCall:
            lowCdif  = MktPrice - BSC(S,K,rf,q,a,T);
            highCdif = MktPrice - BSC(S,K,rf,q,b,T);
        else:
            lowCdif  = MktPrice - BSP(S,K,rf,q,a,T);
            highCdif = MktPrice - BSP(S,K,rf,q,b,T);
        
        if lowCdif*highCdif > 0:
            y = -1;
        else:
            for x in range(MaxIter):
                midP = (a+b)/2;
                if 'C' in PutCall:
                    midCdif = MktPrice - BSC(S,K,rf,q,midP,T);
                else:
                    midCdif = MktPrice - BSP(S,K,rf,q,midP,T);
                
                if abs(midCdif)<Tol:
                    break
                else:
                    if midCdif>0:
                        a = midP;
                    else:
                        b = midP;
            y = midP;
        return y

# In[]  GenerateGaussLaguerre
def  GenerateGaussLaguerre(n):
    # =============================================================================
    #     # Generate abscissas (x) and weights (w) for Gauss Laguerre integration
    #     # The Laguerre polynomial
    #     L=np.zeros(n)
    #     for k in range(n):
    #         L[k]=(((-1)**k)/math.factorial(k)*nchoosek(n,k))
    #     
    #     L.ndim
    #     # Need to flip the vector to get the roots in the correct order
    #     L = np.flip(L);
    # 
    #     # Find the roots.  
    #     x = np.flipud(np.roots(L));
    #     
    # =============================================================================
    import math
    from math import exp
    import numpy as np
    
    def nchoosek(n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)
    [x,w]=np.polynomial.laguerre.laggauss(n) 
    
    # Find the weights
    w = np.zeros(n)
    dL = np.zeros((n,len(x)));    
    for j in range(len(x)):
        # The coefficients of the derivative of the Laguerre polynomial
        for k in range(n):
            dL[k,j] = (-1)**(k+1)/math.factorial(k)*nchoosek(n,k+1)*x[j]**(k);
        # The weight w[j]
        w[j] = 1/x[j]/sum(dL[:,j])**2;
        # The total weight w[j]exp(x(j))
        w[j] = w[j]*exp(x[j]);
  
    return [x, w]
# In[]  HestonProb   
def HestonProb(phi,kappa,theta,lambd,rho,sigma,tau,K,S,r,q,v,Pnum,Trap):
    # Returns the integrand for the risk neutral probabilities P1 and P2.
    # phi = integration variable
    # Pnum = 1 or 2 (for the probabilities)
    # Heston parameters:
    #    kappa  = volatility mean reversion speed parameter
    #    theta  = volatility mean reversion level parameter
    #    lambda = risk parameter
    #    rho    = correlation between two Brownian motions
    #    sigma  = volatility of variance
    #    v      = initial variance
    # Option features.
    #    PutCall = 'C'all or 'P'ut
    #    K = strike price
    #    S = spot price
    #    r = risk free rate
    #    q = dividend yield
    #    Trap = 1 "Little Trap" formulation 
    #           0  Original Heston formulation
    
    from math import pi
    from cmath import exp, sqrt,log
    import cmath
    import numpy as np
    
# =============================================================================
#     phi = 0.0444894
#     tau =1.5
#     v=0.05
#     S=S0
# =============================================================================
    
    i=1j
    # Log of the stock price.
    x = log(S);
    
    # Parameter "a" is the same for P1 and P2.
    a = kappa*theta;
    
    # Parameters "u" and "b" are different for P1 and P2.
    if Pnum==1:
        u = 0.5;
        b = kappa + lambd - rho*sigma;
    else:
        u = -0.5;
        b = kappa + lambd;
    d = sqrt((rho*sigma*i*phi - b)**2 - (sigma**2)*(2*u*i*phi - phi**2));
    g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d);
    if Trap==1:
        # "Little Heston Trap" formulation
        c = 1/g;
        D = (b - rho*sigma*i*phi - d)/(sigma**2)*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)));
        G = (1 - c*np.exp(-d*tau))/(1-c);
        C = (r-q)*i*phi*tau + a/(sigma**2)*((b - rho*sigma*i*phi - d)*tau - 2*np.log(G));
    elif Trap==0:
        # Original Heston formulation.
        G = (1 - g*np.exp(d*tau))/(1-g);
        C = (r-q)*i*phi*tau + a/sigma**2*((b - rho*sigma*i*phi + d)*tau - 2*np.log(G));
        D = (b - rho*sigma*i*phi + d)/sigma**2*((1-np.exp(d*tau))/(1-g*np.exp(d*tau)));
    # The characteristic function.
    f =np.exp(C + D*v + i*phi*x);
    # Return the real part of the integrand.
    y = (np.exp(-i*phi*np.log(K))*f/i/phi)
    y=y.real
    return y
# In[]  HestonProb        
def HestonCF(phi,kappa,theta,lambd,rho,sigma,tau,S,r,q,v0,Trap):
    # Returns the Heston characteristic function,f2 (the second CF)
    # Uses original Heston formulation 1,or formulation 2 from "The Little Heston Trap"
    # phi = integration variable
    # Heston parameters:
    #    kappa  = volatility mean reversion speed parameter
    #    theta  = volatility mean reversion level parameter
    #    lambd = risk parameter
    #    rho    = correlation between two Brownian motions
    #    sigma  = volatility of variance
    #    v0     = initial variance
    # Option features.
    #    tau = maturity
    #    S = spot price
    #    r = risk free rate
    #    q = dividend yield
    # Trap = 0 --> Original Heston Formulation
    # Trap = 1 --> Little Trap formulation

    from math import pi
    from cmath import exp, sqrt,log
    import cmath
    import numpy as np
    #phi = 0.0444894
    #tau =1.5
    #v=0.05
    
    i=1j
    # Log of the stock price.
    x = log(S);
    
    # Parameter a
    a = kappa*theta;
    
    # Parameters u,b,d,and b
    u = -0.5;
    b = kappa + lambd;
    d = np.sqrt((rho*sigma*i*phi - b)**2 - (sigma**2)*(2*u*i*phi - phi**2));
    g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d);
    if Trap==1:
        # "Little Heston Trap" formulation
        c = 1/g;
        G = (1 - c*np.exp(-d*tau))/(1-c);
        C = (r-q)*i*phi*tau + a/sigma**2*((b - rho*sigma*i*phi - d)*tau - 2*np.log(G));
        D = (b - rho*sigma*i*phi - d)/sigma**2*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)));
    elif Trap==0:
        # Original Heston formulation.
        G = (1 - g*np.exp(d*tau))/(1-g);
        C = (r-q)*i*phi*tau + a/sigma**2*((b - rho*sigma*i*phi + d)*tau - 2*np.log(G));
        D = (b - rho*sigma*i*phi + d)/sigma**2*((1-np.exp(d*tau))/(1-g*np.exp(d*tau)));
    
    # The characteristic function.
    y = np.exp(C + D*v0 + i*phi*x);
    return y

# In[] FRFT
def FRFT(x,beta):
    # Fractional fast Fourier transform
    # Uses the Matlab functions fft() and ifft() for the FFT and inverse FFT
    # Input:
    #   x = a row vector of values
    #   beta = increment parameter
    
    import numpy as np
    from cmath import exp, pi
    from numpy import fft

    i = 1j
    N = len(x);
    x=x.conj().transpose()
    # Construct the y and z vectors
    #y = [np.exp(-i*pi*np.arange(N)**2*beta)*x, np.zeros(N)];
    #z = [np.exp( i*pi*np.arange(N)**2*beta), np.exp(i*pi*np.arange(N,0,-1)**2*beta)];

    #y = np.pad(np.exp(-i*pi*np.arange(N)**2*beta)*x, (0,N),'constant');
    y = np.append(np.exp(-i*pi*np.arange(N)**2*beta)*x, np.zeros(N));
   
    z = np.append(np.exp( i*pi*np.arange(N)**2*beta), np.exp(i*pi*np.arange(N,0,-1)**2*beta));

    # FFT on y and z
    Dy = np.fft.fft(y);
    Dz = np.fft.fft(z);
    
    # h vectors
    h = Dy*Dz;
    ih = np.fft.ifft(h);
    
    # e vector
    #e = np.pad(np.exp(-i*pi*np.arange(N)**2*beta), (0,N),'constant');
    e = np.append(np.exp(-i*pi*np.arange(N)**2*beta), np.zeros(N));
    
    #e = [np.exp(-i*pi*np.arange(N)**2*beta), np.zeros(N)];
    
    # The FRFT vector
    xhat = e*ih;
    xhat = xhat[0:N];
    
    return xhat

# In[]HestonCallFRFT
def HestonCallFRFT(N,uplimit,S0,r,q,tau,kappa,theta,lambd,rho,sigma,v0,Trap,alpha,rule,eta,lambdinc):
    # Fast Fourier Transform of the Heston Model
    # INPUTS
    #   N  = number of discretization points
    #   uplimit = Upper limit of integration
    #   S0 = spot price
    #   r = risk free rate
    #   q = dividend yield
    #   tau = maturity
    #   sigma = volatility
    #   alpha = dampening factor
    #   rule = integration rule
    #     rule = 1 --> Trapezoidal
    #     rule = 2 --> Simpson's Rule
    #   eta = integration range increment
    #   lambdinc = strike range increment
    # --------------------------------------
    # OUTPUTS
    #   CallFFT = Heston call prices using FFT
    #   CallBS  = Heston call prices using closed form
    #         K = Strike prices
    #       eta = increment for integration range
    # lambdinc = increment for log-strike range
    # --------------------------------------

    from math import log, pi
    import numpy as np
    from cmath import exp
    i =1j

    # log spot price
    s0 = log(S0);
    
    # Initialize and specify the weights
    w = np.zeros(N);
    if 'T' in rule:             # Trapezoidal rule
        w[0] = 1/2;
        w[N-1] = 1/2;
        w[1:N-1] = 1;
    elif 'S' in rule:        # Simpson's rule
        w[0] = 1/3;
        w[N-1] = 1/3;
        for k in range (1,N-1):
            if (k-1)%2==0:
                w[k] = 4/3;
            else:
                w[k] = 2/3;
    
    # Specify the b parameter
    b = N*lambdinc/2;
    
    # Create the grid for the integration
    v = eta*np.arange(N);
    
    # Create the grid for the log-strikes
    k = -b+lambdinc*np.arange(N) + s0;
    
    # Create the strikes and identify ATM 
    K = np.exp(k);
    
    # Initialize the price vector;
    CallFRFT = np.zeros(N);
    
    # Parameter for the FRFT
    beta = lambdinc*eta/2/pi;
    
    # Implement the FRFT - fast algorithm
    U = np.arange(N);
    J = np.arange(N);
    psi = HestonCF(v-(alpha+1)*i,kappa,theta,lambd,rho,sigma,tau,S0,r,q,v0,Trap);
    psi = np.conj(psi);
    phi = exp(-r*tau)*psi/np.conj(alpha**2 + alpha - v**2 + i*v*(2*alpha+1));
    x = np.conj(np.exp(i*(b-s0)*v))*phi*w;
    y = (FRFT(x,beta).real);
    CallFRFT = eta*np.exp(-alpha*k)*y/pi;

    return[CallFRFT, K, lambdinc, eta]

# In[] HestonObjFunFRFT
def HestonObjFunFRFT(param,S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule):
    # S  = Spot price
    # rf = risk free rate
    # q  = dividend yield
    # MktPrice = matrix of market prices.
    # K = vector of strikes.
    # T = vector of maturities.
    # PutCall = matrix of put/call indicator 'C' or 'P'
    # MktIV = matrix of market implied volatilities
    # x = abscissas for Gauss Laguerre integration
    # w = weights for Gauss Laguerre integration
    # ObjFun = Type of Objective Function
    #    1 = MSE
    #    2 = Relative MSE
    #    3 = Relative IVMSE
    #    4 = Relative IVMSE Christoffersen, Heston, Jacobs proxy
    # a = Bisection algorithm, small initial estimate
    # b = Bisection algorithm, large initial estimate
    # Tol = Bisection algorithm, tolerance
    # MaxIter = Bisection algorithm, maximum iterations
    # trap = 1:"Little Trap" c.f., 0:original Heston c.f.
    # FFT settings
    #   N  = number of discretization points
    #   uplimit = Upper limit of integration
    #   eta = integration grid size
    #   rule = integration rule
    #     rule = 'T' --> Trapezoidal
    #     rule = 'S' --> Simpson's Rule
    
    from math import log,sqrt,exp
    from scipy.stats import norm
    import numpy as np
    

    kappa  = param[0]; 
    theta  = param[1]; 
    sigma  = param[2]; 
    v0     = param[3]; 
    rho    = param[4];
    lambd = 0;
    
    [NK,NT] = MktPrice.shape;
    lambdinc = 2/N*log(S/K1);
    
    ModelPrice=np.zeros((NK,NT))
    error=np.zeros((NK,NT))
    for t in range(NT):
        [CallFRFT, KK, lambdinc, eta] = HestonCallFRFT(N,uplimit,S,rf,q,T[t],kappa,theta,lambd,rho,sigma,v0,trap,alpha,rule,eta,lambdinc);
        #    CallPrice = interp1(KK,CallFRFT,K,'linear')';

        #from scipy import interpolate
        #CallPrice = interpolate.interp1d(KK,CallFRFT, kind='linear')
        #CallPrice=CallPrice(K)
        CallPrice = np.interp(K,KK,CallFRFT)
        #CallPrice = LinearInterpolate(KK,CallFRFT,K);
        for k in range(NK):
            if 'C' in PutCall[k,t]:
                ModelPrice[k,t] = CallPrice[k];
            else:
                ModelPrice[k,t] = CallPrice[k] - S*exp(-q*T[t]) + exp(-rf*T[t])*K[k];
        #Select the objective function
        if ObjFun == 1:
            # MSE
            error[:,t] = (MktPrice[:,t] - ModelPrice[:,t])**2;
        elif ObjFun == 2:
                # RMSE
            error[:,t] = (MktPrice[:,t] - ModelPrice[:,t])**2 / MktPrice[:,t];
        elif ObjFun == 3:
        
            for k in range(NK):
                # IVMSE
                ModelIV = BisecBSIV(PutCall[k,t],S,K[k],rf,q,T[t],a,b,ModelPrice[k,t],Tol,MaxIter);
                error[k,t] = (ModelIV - MktIV[k,t])**2 / MktIV[k,t];
        elif ObjFun == 4:
        
            for k in range(NK):
                # IVRMSE Christoffersen, Heston, Jacobs proxy
                d = (log(S/K[k]) + (rf-q+MktIV[k,t]**2/2)*T[t])/MktIV[k,t]/sqrt(T[t]);
                Vega[k,t] = S*norm.pdf(d)*sqrt(T[t]);
                error[k,t] = (ModelPrice[k,t] - MktPrice[k,t])**2/Vega[k,t]**2;
    y = sum(sum(error)) / (NT*NK);
    return y

# In[] HestonDE
def HestonDE(NG,NP,CR,F,Hi,Lo,S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule):

    # Differential Algorithm for Heston parameter estimation
    # NG = Number of generations (iterations)
    # NP = Number of population members
    # CR = Crossover ratio (=0.5)
    # F  = Threshold (=0.8)
    # Hi = Vector of upper bounds for the parameters
    # Lo = Vector of lower bounds for the parameters
    # S  = Spot Price
    # K1 = First strike point for the FRFT
    # rf = Risk free rate
    # q  = Dividend Yield
    # MktPrice = Market quotes for prices
    # K = Vector of Strikes
    # T = Vector of Maturities
    # PutCall = Matrix of 'P'ut or 'C'all
    # MktIV = Market quotes for implied volatilities
    # ObjFun = Type of Objective Function
    #    1 = MSE
    #    2 = RMSE 
    #    3 = IVMSE (implied volatility mean error sum of squares)
    #    4 = Christoffersen, Jacobs, Heston (2009)
    # Bisection method settings
    #    a = Lower limit
    #    b = upper limit
    #    Tol = Tolerance
    #    MaxIter = Max number of iterations
    # trap = 1 is "Little Trap" c.f., 0 is Heston c.f.
    # FRFT Settings
    #    N = Number of points
    #    uplimit = Upper integration point
    #    eta = integration grid size
    #    alpha = dampening factor
    #    rule = 'T'rapezoidal or 'S'impsons
    
    import numpy as np
    
    kappaU = Hi[0];  kappaL = Lo[0];
    thetaU = Hi[1];  thetaL = Lo[1];
    sigmaU = Hi[2];  sigmaL = Lo[2];
    v0U    = Hi[3];  v0L    = Lo[3];
    rhoU   = Hi[4];  rhoL   = Lo[4];
    
    # Step1.  Generate the population matrix of random parameters
    P = np.array([kappaL + (kappaU-kappaL)*np.random.random((NP,)),
          thetaL + (thetaU-thetaL)*np.random.random((NP,)), 
          sigmaL + (sigmaU-sigmaL)*np.random.random((NP,)), 
             v0L + (   v0U-   v0L)*np.random.random((NP,)),
            rhoL + (  rhoU-  rhoL)*np.random.random((NP,))]);
    
    # Generate the random numbers outside the loop
    U=[np.random.rand(5,NP) for x in range(NG)]
    
    # Loop through the generations
    for k in range(NG):
    # Loop through the population
        for i in range(NP):
            # Select the i-th member of the population
            P0 = P[:,i]
            # Set the condition so that the "while" statement is executed at least once
            Pnew = -np.ones(5);
            Condition = sum(np.logical_and(Lo < Pnew,Pnew < Hi));
            while Condition < 5:
                # Select random indices for three other distinct members
                I = np.random.permutation(NP);
                I = I[(I!=i)];
                r = I[0:3];
                # The three distinct members of the population
                Pr1 = P[:,r[0]];
                Pr2 = P[:,r[1]];
                Pr3 = P[:,r[2]];
                R = np.random.permutation(5);
                Pnew = np.zeros(5);
                # Steps 2 and 3.  Mutation and recombination
                for j in range(5):
                    Ri = R[0];
                    u = U[k][j,i];
                    if u<=CR or j==Ri:
                        Pnew[j] = Pr1[j] + F*(Pr2[j] - Pr3[j]);
                    else:
                        Pnew[j] = P0[j];
                Condition = sum(np.logical_and(Lo < Pnew,Pnew < Hi));
                
            # Step 4.  Selection
            # Calculate the objective function for the i-th member and for the candidate
            f0   = HestonObjFunFRFT(P0,  S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule);
            fnew = HestonObjFunFRFT(Pnew,S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule);
              

         # Verify whether the candidate should replace the i-th member
            # in the population and replace if conditions are satisfied
            
            if fnew < f0:
                P[:,i] = Pnew;
        
        print('Completed loop for generation {}'. format(k))
    
    # Calculate the objective function for each member in the updated population
    f=np.zeros(NP)
    for i in range(NP):
        f[i] = HestonObjFunFRFT(P[:,i],S,K1,rf,q,MktPrice,K,T,PutCall,MktIV,ObjFun,a,b,Tol,MaxIter,trap,N,uplimit,eta,alpha,rule);
        
    # Find the member with the lowest objective function
    J = np.argmin(f);
    
    # Return the selected member/parameter
    y = P[:,J];
    
    return y

# In[] HestonPriceGaussLaguerre
def HestonPriceGaussLaguerre(PutCall,S,K,T,r,q,param,trap,x,w):

    # Heston (1993) call or put price by Gauss-Laguerre Quadrature
    # Uses the original Heston formulation of the characteristic function,
    # or the "Little Heston Trap" formulation of Albrecher et al.
    # INPUTS -------------------------------------------------------
    #   PutCall = 'C' Call or 'P' Put
    #   S = Spot price.
    #   K = Strike
    #   T = Time to maturity.
    #   r = Risk free rate.
    #   kappa  = Heston parameter: mean reversion speed.
    #   theta  = Heston parameter: mean reversion level.
    #   sigma  = Heston parameter: volatility of vol
    #   lambda = Heston parameter: risk.
    #   v0     = Heston parameter: initial variance.
    #   rho    = Heston parameter: correlation
    #   trap:  1 = "Little Trap" formulation
    #          0 = Original Heston formulation
    #   x = Gauss Laguerre abscissas
    #   w = Gauss Laguerre weights
    # OUTPUT -------------------------------------------------------
    #   The Heston call or put price
    from math import exp,pi
    import numpy as np

    kappa  = param[0]; 
    theta  = param[1]; 
    sigma  = param[2]; 
    v0     = param[3]; 
    rho    = param[4];
    lambd = 0;

# =============================================================================
# T=tau
# S=S0
# trap=1
#  
# =============================================================================
    # Numerical integration
    int1=np.zeros(len(x))
    int2=np.zeros(len(x))
    for k in range(len(x)):
        int1[k] = w[k]*HestonProb(x[k],kappa,theta,lambd,rho,sigma,T,K,S,r,q,v0,1,trap);
        int2[k] = w[k]*HestonProb(x[k],kappa,theta,lambd,rho,sigma,T,K,S,r,q,v0,2,trap);
    # Define P1 and P2
    P1 = 1/2 + 1/pi*sum(int1);
    P2 = 1/2 + 1/pi*sum(int2);
    
    # The call price
    HestonC = S*exp(-q*T)*P1 - K*exp(-r*T)*P2;
    
    # The put price by put-call parity
    HestonP = HestonC - S*exp(-q*T) + K*exp(-r*T);
    
    # Output the option price
    if 'C' in PutCall:
        y = HestonC;
    else:
        y = HestonP;
    return y

# In[]
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
  
    

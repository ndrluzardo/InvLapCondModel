import numpy as np
import math

# def InvLapModel(k,ts_ini,ts_end,ts_num,t):
#     ts=np.logspace(math.log10(ts_ini),math.log10(ts_end),num=ts_num)
#     Ck=(1/math.factorial(k))*k**(k+1)
#     Fs=np.exp(-k*np.outer(t,1/ts)) # temporal context cells: Fs(time,taustar)
#     # Fs_to_k=((-t)**k)*Fs # derivatives
#     ftilde=Ck*np.outer((t**k),1/ts**(k+1))*Fs # time cells: ftilde(time,taustar)
#     return Fs,ftilde,ts

# def Translation(t,delta,k,ts):
#     Ck=(1/math.factorial(k))*k**(k+1) 
#     Fs=np.exp(-k*np.outer(t+delta,1/ts)) # temporal context cells: Fs(delta,taustar)
#     ftilde_trans=Ck*np.outer(((t+delta)**k),1/ts**(k+1))*Fs # time cells: ftilde(delta,taustar)
#     return ftilde_trans

def c_division(n, d):
    return n / d if d else 1

def DerivMatrix(s):
    # Calculates the numerical derivatives for F(s)
    # Details of this calculation are explained in section 3.1.1 of Shankar, K. H. and
    # Howard, M. W. (2013) ‘Optimally Fuzzy Temporal Memory’, Journal of
    # Machine Learning Research, 14, pp. 3785–3812.
    N = s.size   # number of cells in the intermediate layer (leaky integrators): F(s)
                    # k edges will be ignored because the spatial derivative will be
                    # incomplete without them. 
    # Create DerivMatrix that will be used to compute a numeric derivative of F to
    # compute ftilde
    DerivMatrix = np.zeros((N,N))
    for i in range(1,N-1):
        DerivMatrix[i,i-1] = -(s[i+1]-s[i])/(s[i]-s[i-1])/(s[i+1] - s[i-1])
        DerivMatrix[i,i] = ((s[i+1]-s[i])/(s[i]- s[i-1])-(s[i]-s[i-1])/(s[i+1]-s[i]))/(s[i+1] - s[i-1])
        DerivMatrix[i,i+1] = (s[i]-s[i-1])/(s[i+1]-s[i])/(s[i+1] - s[i-1])
    return DerivMatrix

def TempContCell(h,y_n,Input,s):
    k1=h*(-s*y_n+Input)
    k2=h*(-s*(y_n+k1/2)+Input)
    k3=h*(-s*(y_n+k2/2)+Input)
    k4=h*(-s*(y_n+k3)+Input)

    y_nplus1=y_n+(1/6)*(k1+2*k2+2*k3+k4)
    return y_nplus1

def TimeCell(ConstC,TempContCells,Dtothek,k):
    """
    returns a vector ftil(taustar) without the padded taustars
    """
    Fderivatives=np.dot(Dtothek,TempContCells)
    ftil=np.multiply(ConstC,Fderivatives)[k:-k]
    return ftil

def Trans(TempCells,s,delta,ConstC,Dtothek,k):
    """
    returns TransTimeCells(taustar,delta) without the padded taustars/deltas
    """
    TransTempCells=TempCells[:,np.newaxis]*np.exp(np.outer(-s,delta)) # this works fine, output is F(s,delta)
    Fderivatives=np.dot(Dtothek,TransTempCells)
    TransTimeCells=np.multiply(ConstC[:,np.newaxis],Fderivatives)[k:-k,k:-k]
    #TransTimeCells=TimeCell(ConstC[:,np.newaxis],TransTempCells,Dtothek,k)[:,k:-k] # this also works, needed the np.newaxis for broadcasting to work
    return TransTimeCells

#shiftedftilde = (Ck[:,np.newaxis]*dkdsk.dot(np.exp(-s*delta)[:,np.newaxis]*F))[k:-k,:]
#        shiftedftilde[shiftedftilde<0] = 1e-20




#ftilde = (Ck[:,np.newaxis]*dkdsk.dot(F))[k:-k,:]


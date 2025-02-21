import numpy as np
import itertools

from .indexing import linear_combinations


def pdq_avg_integral(A,n,idx=0):
    return np.sum([nk[idx]*(np.abs(Ak)**2)/2 for Ak,nk in zip(A,n)])
    

def pdq_integral(A,n,Theta,idx=0):
    if not isinstance(Theta,list):
        Theta = [Theta]
    assert len(Theta) == len(n[0])-2, 'Theta should include all non-integrated angles, i.e. [Theta_y,Theta_zeta] for x-plane integration'
    
    # Computing the non-integrated phase (works for any number of dimensions)
    n_tmp   = [nk[:idx] + nk[idx+1:] for nk in n]
    phi     = [np.angle(Ak) + sum([nk[i]*Theta[i] for i in range(len(Theta))]) for Ak,nk in zip(A,n_tmp)]
    
    # Computing the full integral
    return 1/2 * sum(np.abs(Ak)*np.abs(Aj)*nj[idx]*np.cos(phik-phij)    for nk,Ak,phik in zip(n,A,phi) 
                                                                        for nj,Aj,phij in zip(n,A,phi) 
                                                                        if nj[idx]==nk[idx])


def sin_integral(Q0,nk,nj,phik,phij):
    # https://www.wolframalpha.com/input?i=integrate+sin%28nx%2Bb%29*sin%28mx%2Bc%29+from+0+to+a

    a = 2*np.pi*Q0
    n = nk
    m = nj
    b = phik
    c = phij

    if (n != m) and (n != -m) and (n*m!=0):
        return 1/2 * ( np.sin(b-c) - np.sin(a*(n-m) + b - c))/(m-n) + 1/2*(np.sin(b+c) - np.sin(a*(m+n) + b + c))/(m+n)
    
    elif (n == m) and (n*m!=0):
        return (a*n*np.cos(b-c) - np.sin(a*n)*np.cos(a*n + b + c))/(2*n)
    
    elif (n == -m) and (n*m!=0):
        return (np.sin(a*n)*np.cos(a*n + b - c) - a*n*np.cos(b+c))/(2*n)

    elif (n == 0) and (m != 0):
        return np.sin(b)*(np.cos(c) - np.cos(a*m + c))/m
    
    elif (n != 0) and (m == 0):
        return np.sin(c)*(np.cos(b) - np.cos(a*n + b))/n
    
    elif (n == 0) and (m == 0):
        return a*np.sin(b)*np.sin(c)
    
    else:
        print(n,m)
        return 0



def pdq_partial_integral(Q0,A,n,Theta,idx=0):
    if not isinstance(Theta,list):
        Theta = [Theta]
    assert len(Theta) == len(n[0])-2, 'Theta should include all non-integrated angles, i.e. [Theta_y,Theta_zeta] for x-plane integration'
    
    # Computing the non-integrated phase (works for any number of dimensions)
    n_tmp   = [nk[:idx] + nk[idx+1:] for nk in n]
    phi     = [np.angle(Ak) + sum([nk[i]*Theta[i] for i in range(len(Theta))]) for Ak,nk in zip(A,n_tmp)]
    
    # Computing the full partial integral (going from N to N+1) https://www.wolframalpha.com/input?i=integrate+sin%28nx%2Bb%29*sin%28mx%2Bc%29+from+0+to+a    
    return 1/2/np.pi * sum(np.abs(Ak)*np.abs(Aj)*nj[idx]*sin_integral(Q0,nk[idx],nj[idx],phik,phij)     for nk,Ak,phik in zip(n,A,phi) 
                                                                                                        for nj,Aj,phij in zip(n,A,phi))
    
        
        



def invariant(A_list,n_list,gemitt_list = [1,1,1],idx=0):
    assert len(A_list) == len(n_list)
    return np.sum([gemitt_plane * pdq_avg_integral(A_plane,n_plane,idx) for A_plane,n_plane,gemitt_plane in zip(A_list,n_list,gemitt_list)])



def invariant_6D(Ax,Ay,Az,nx,ny,nz):
    Ix = invariant([Ax,Ay,Az],[nx,ny,nz],idx=0)
    Iy = invariant([Ax,Ay,Az],[nx,ny,nz],idx=1)
    Iz = invariant([Ax,Ay,Az],[nx,ny,nz],idx=2)
    return Ix,Iy,Iz


def invariant_tunes_6D(Ax,Ay,Az,Qx,Qy,Qz,limit_search=-1,max_n=20,max_alias=5,filter_Q = None,handpicked_QxQy = None,return_optimization=False,force_proper_plane = False,warning_tol = 1e-8):

    # Initialisation
    #---------------------------------------
    # Longitudinal plane is well behaved:
    tune_z      = Qz[0]
    Iz_target   = np.abs(Az[0])**2/2


    if handpicked_QxQy is None:
        # Search for the best combination of tunes, all combinations of Qx,Qy!
        #---------------------------------------------------------------------
        if filter_Q is not None:
            assert filter_Q[0]<filter_Q[1], "filter_Q must have filter_Q[0]<filter_Q[1]"
            Q12_list = list(Qx[(Qx>filter_Q[0])&(Qx<filter_Q[1])])[:limit_search] + list(Qy[(Qy>filter_Q[0])&(Qy<filter_Q[1])])[:limit_search]   
        else:
            Q12_list = list(Qx)[:limit_search] + list(Qy)[:limit_search]    
        Q12_comb = list(itertools.combinations(Q12_list, 2))
    else:
        # handpicked values of (Qx,Qy) to test
        #---------------------------------------------------------------------
        assert len(handpicked_QxQy[0])==2, "handpicked_QxQy must be a list of 2-tuples"


    Iz_estimate = []
    for Q12 in Q12_comb:
        Q123    = list(Q12) + [tune_z]
        nx      = linear_combinations(Qx,Qvec = Q123,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)
        ny      = linear_combinations(Qy,Qvec = Q123,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)
        nz      = linear_combinations(Qz,Qvec = Q123,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)

        Ix = invariant([Ax,Ay,Az],[nx,ny,nz],idx=0)
        Iy = invariant([Ax,Ay,Az],[nx,ny,nz],idx=1)
        Iz = invariant([Ax,Ay,Az],[nx,ny,nz],idx=2)
        if Ix<0 or Iy <0:
            Iz_estimate.append(np.inf)
        else:
            Iz_estimate.append(Iz)
    Iz_estimate = np.array(Iz_estimate)
    #---------------------------------------------------------------------


    # Find the best combination
    #---------------------------------------------------------------------
    idx_match   = np.argmin(np.abs(Iz_estimate-Iz_target))
    Q123_match  = list(Q12_comb[idx_match]) + [tune_z]
    #---------------------------------------------------------------------


    # Check if Q1 is Qx or Qy:
    #---------------------------------------------------------------------
    nx  = linear_combinations(Qx,Qvec = Q123_match,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)
    ny  = linear_combinations(Qy,Qvec = Q123_match,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)
    Ixx = pdq_avg_integral(Ax,nx,idx=0)
    Ixy = pdq_avg_integral(Ay,ny,idx=0)
    if Ixx<Ixy:
        # mislabeled Q! 
        Q123_match = [Q123_match[1],Q123_match[0],Q123_match[2]]
    #---------------------------------------------------------------------


    # Return the line from proper plane if present:
    #---------------------------------------------------------------------
    if force_proper_plane:
        nx  = linear_combinations(Qx,Qvec = Q123_match,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)
        ny  = linear_combinations(Qy,Qvec = Q123_match,max_n=max_n,max_alias=max_alias,warning_tol=warning_tol)
        if (1,0,0,0) in nx:
            Q123_match[0] = Qx[nx.index((1,0,0,0))]
        if (0,1,0,0) in ny:
            Q123_match[1] = Qy[ny.index((0,1,0,0))]
    #---------------------------------------------------------------------


    if return_optimization:
        sort_match  = np.argsort(np.abs(Iz_estimate-Iz_target))
        if Q123_match[0]<Q123_match[1]:
            return np.array([sorted(Q12) + [tune_z] for Q12 in Q12_comb])[sort_match], np.array(np.abs(Iz_estimate-Iz_target))[sort_match]
        else:
            return np.array([sorted(Q12)[::-1] + [tune_z] for Q12 in Q12_comb])[sort_match], np.array(np.abs(Iz_estimate-Iz_target))[sort_match]
        
    else:
        return Q123_match







# NAQT: Numerical Analysis of Quasiperiodic Tori
    
# Possible Names:
#====================
# NAOMI: Numerical Analysis Of Machine Invariants
# NAOMI: Numerical Analysis Of Map Invariants
# IMA  : Integrals of Motion Analysis
# TCIM

# QDIM: Quasiperiodic Description of the Integrals of Motion   
# NDIM: Numerical Description of the Integrals of Motion
# NAIM: Numerical Analysis of the Integrals of Motion
# NAQIM: Numerical Analysis of Quasiperiodic Integrals of Motion
# NAFA: Numerical Analysis of the Fundamental Actions
# NAFI: Numerical Analysis of the Fundamental Invariants
# NAQT: Numerical Analysis of Quasiperiodic Trajectories

# NAQI: Numerical Analysis of Quasiperiodic Invariants
# QIAT: Quasiperiodic Invariant Analysis Toolbox
# NIIM: Numerical Integration of the Integrals of Motion
# NCIM: Numerical Computation of the Integrals of Motion
# NEMI: Numerical Evaluation of the Fundamental Invariants
# NEIM: Numerical Evaluation of the Integrals of Motion
# NEFI: Numerical Evaluation of the Fundamental Invariants
# NAQS: Numerical Analysis of Quasiperiodic Systems

# naht, NAHT: Numerical Analysis of Hamiltonian Tori
# NAKT: Numerical Analysis of KAM Tori
# NAST: Numerical Analysis of Stable Tori
#nast
# nahts
# naff
# naqi
# qdim
# naomi
#    
# neim

# nefi

# naff
# naqt



#naff
#neiv
# naqs





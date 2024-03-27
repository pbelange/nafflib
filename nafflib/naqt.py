import numpy as np
import itertools

from .indexing import linear_combinations


def pdq_avg_integral(A,n,idx=0):
    return np.sum([nk[idx]*(np.abs(Ak)**2)/2 for Ak,nk in zip(A,n)])
    


def invariant(A_list,n_list,idx=0):
    assert len(A_list) == len(n_list)
    return np.sum([pdq_avg_integral(A_plane,n_plane,idx) for A_plane,n_plane in zip(A_list,n_list)])



def invariant_6D(Ax,Ay,Az,nx,ny,nz):
    Ix = invariant([Ax,Ay,Az],[nx,ny,nz],idx=0)
    Iy = invariant([Ax,Ay,Az],[nx,ny,nz],idx=1)
    Iz = invariant([Ax,Ay,Az],[nx,ny,nz],idx=2)
    return Ix,Iy,Iz


def invariant_tunes_6D(Ax,Ay,Az,Qx,Qy,Qz,limit_search=-1,max_n=20,max_alias=5,filter_Q = None,handpicked_QxQy = None,return_optimization=False,warning_tol = 1e-8):

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

        Iz = invariant([Ax,Ay,Az],[nx,ny,nz],idx=2)
        Iz_estimate.append(Iz)
    Iz_estimate = np.array(Iz_estimate)
    #---------------------------------------------------------------------


    # Find the best combination
    #---------------------------------------------------------------------
    idx_match   = np.argsort(np.abs(Iz_estimate-Iz_target))
    Q123_match  = list(Q12_comb[idx_match[0]]) + [tune_z]
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


    if return_optimization:
        if Ixx<Ixy:
            # mislabeled Q! (just ensuring the first element is the matching tune, the rest is irrelevant)
            return np.array([[Q12[1],Q12[0]] + [tune_z] for Q12 in Q12_comb])[idx_match], np.array(np.abs(Iz_estimate-Iz_target))[idx_match]
        else:
            return np.array([list(Q12) + [tune_z] for Q12 in Q12_comb])[idx_match], np.array(np.abs(Iz_estimate-Iz_target))[idx_match]
        
    else:
        return Q123_match







# NAQT: Numerical Analysis of Quasiperiodic Tori
    
# Possible Names:
#====================
# NAOMI: Numerical Analysis Of Machine Invariants
# NAOMI: Numerical Analysis Of Map Invariants
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





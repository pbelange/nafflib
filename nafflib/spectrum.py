import numpy as np
import itertools

from .toolbox import find_linear_combinations
from .invariant import invariant, pdq_avg_integral

def invariant_tunes_6D(Ax,Ay,Az,Qx,Qy,Qz,n_grid = None, limit_search=-1,max_n_search=50,filter_Q = None,handpicked_QxQy = None,return_optimization=False):

    # Initialisation
    #---------------------------------------
    # Longitudinal plane is well behaved:
    tune_z      = Qz[0]
    Iz_target   = np.abs(Az[0])**2/2

    # Preparing n_grid
    if n_grid is None:
        max_n = max_n_search
        n1,n2,n3,n4 = np.mgrid[-max_n:max_n+1,-max_n:max_n+1,
                            -max_n:max_n+1,-max_n:max_n+1]
        n_grid      = [n1, n2, n3, n4]
    #---------------------------------------


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
        Q123   = list(Q12) + [tune_z]
        nx,_,_ = find_linear_combinations(Qx,n_grid=n_grid,fundamental_tunes=Q123)
        ny,_,_ = find_linear_combinations(Qy,n_grid=n_grid,fundamental_tunes=Q123)
        nz,_,_ = find_linear_combinations(Qz,n_grid=n_grid,fundamental_tunes=Q123)

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
    nx,_,_ = find_linear_combinations(Qx,n_grid=n_grid,fundamental_tunes=Q123_match)
    ny,_,_ = find_linear_combinations(Qy,n_grid=n_grid,fundamental_tunes=Q123_match)
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




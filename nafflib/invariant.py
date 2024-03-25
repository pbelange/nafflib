import numpy as np


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





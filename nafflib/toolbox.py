import numpy as np
from .optimize import laskar_dfft,point_dfft,expArray
from .windowing import hann
from .naff import fundamental_frequency


# ---------------------------------------
# Taken from 10.5170/CERN-1994-002, eq. 5.1.2
def henon_map(x, px, Q, num_turns):
    """
    Simulates the Henon map for 2D phase space.

    Parameters
    ----------
    x : float
        Initial position of the particle.
    px : float
        Initial momentum of the particle.
    Q : float
        Tune of the map.
    num_turns : int
        Number of turns to simulate.

    Returns
    -------
    x,px : tuple of ndarray
        Arrays representing the position and momentum of the particle at each turn.
    """
    z_vec = np.nan * np.ones(num_turns) + 1j * np.nan * np.ones(num_turns)
    z_vec[0] = x - 1j * px
    for ii in range(num_turns - 1):
        _z = z_vec[ii]

        z_vec[ii + 1] = np.exp(2 * np.pi * 1j * Q) * (
            _z - 1j / 4 * (_z + np.conjugate(_z)) ** 2
        )
    return np.real(z_vec), -np.imag(z_vec)


# ---------------------------------------


# ---------------------------------------
# Taken from 10.5170/CERN-1994-002, eq. 7.1.5
def henon_map_4D(x, px, y, py, Qx, Qy, coupling, num_turns):
    """
    Simulates the 4D Henon map for phase space, incorporating coupling between two oscillations.

    Parameters
    ----------
    x, y : float
        Initial positions in two orthogonal dimensions.
    px, py : float
        Initial momenta in two orthogonal dimensions.
    Qx, Qy : float
        Tunes of the map in each dimension.
    coupling : float
        Coupling strength between the dimensions.
    num_turns : int
        Number of turns to simulate.

    Returns
    -------
    x,px,y,py : tuple of ndarray
        Arrays representing the position and momentum of the particle in each dimension at each turn.
    """

    z1_vec = np.nan * np.ones(num_turns) + 1j * np.nan * np.ones(num_turns)
    z2_vec = np.nan * np.ones(num_turns) + 1j * np.nan * np.ones(num_turns)
    z1_vec[0] = x - 1j * px
    z2_vec[0] = y - 1j * py
    for ii in range(num_turns - 1):
        _z1 = z1_vec[ii]
        _z2 = z2_vec[ii]

        _z1_sum = _z1 + np.conjugate(_z1)
        _z2_sum = _z2 + np.conjugate(_z2)
        z1_vec[ii + 1] = np.exp(2 * np.pi * 1j * Qx) * (
            _z1 - 1j / 4 * (_z1_sum**2 - coupling * _z2_sum**2)
        )
        z2_vec[ii + 1] = np.exp(2 * np.pi * 1j * Qy) * (
            _z2 + 1j / 2 * coupling * _z1_sum * _z2_sum
        )

    # Returns x,px,y,py
    return np.real(z1_vec), -np.imag(z1_vec), np.real(z2_vec), -np.imag(z2_vec)


# ---------------------------------------




# ---------------------------------------
def dfft(nu,z):
    return np.array([point_dfft(_f, z) for _f in nu])
#----------------------------------------


def naff_dfft(nu, z, N=None, num_harmonics=1, window_order=2, window_type="hann"):

    assert num_harmonics >= 1, "number_of_harmonics needs to be >= 1"

    # initialisation, creating a copy of the signal since we'll modify it
    # ---------------------
    if N is None:
        N = np.arange(len(z))
    _z = z.copy()
    # ---------------------

    # Computing the window function
    # ---------------------
    window_fun = {"hann": hann}[window_type.lower()]
    w_of_N = window_fun(N, order=window_order)
    # ---------------------


    A_dfft = []
    for _ in range(num_harmonics):

        # Computing frequency and amplitude
        amp, freq = fundamental_frequency(_z*w_of_N, N=N)

        # Computing cfft
        _dfft = dfft(nu,_z*w_of_N)

        # Saving results
        A_dfft.append(_dfft)

        # Substraction procedure
        zgs = amp * expArray(2 * np.pi * 1j * freq, len(N))
        _z -= zgs

    return A_dfft
# ---------------------------------------


# ---------------------------------------
def find_linear_combinations(
    frequencies,
    fundamental_tunes=[],
    max_harmonic_order=10,
    n_vec=None,
    to_pandas=False,
):
    """
    Identifies linear combinations of fundamental tunes that closely match the given frequencies.

    Parameters
    ----------
    frequencies : ndarray
        Array of frequencies to analyze.
    fundamental_tunes : list, optional
        List of fundamental tunes for resonance analysis. Length can be 1, 2, or 3.
    max_harmonic_order : int, optional
        Maximum order of harmonics to consider.
    n_vec : list of ndarrays, optional
        List of arrays representing the possible combinations of the fundamental tunes. If not provided, it is generated. (for big arrays, it is better to provide it to avoid memory issues when looping)
    to_pandas : bool, optional
        If True, returns a pandas DataFrame; otherwise, returns lists and an array.

    Returns
    -------
    r_values,error,f_values : DataFrame or tuple
        Resonance values, errors, and corresponding frequencies, either as a DataFrame or separate data structures.
    """
    fundamental_tunes = list(fundamental_tunes)
    assert len(fundamental_tunes) in [
        1,
        2,
        3,
    ], "need 1, 2 or 3 fundamental tunes (2D,4D,6D)"

    # Create a 3D array of all possible combinations of n_vec
    idx = max_harmonic_order
    if n_vec is None:
        if len(fundamental_tunes) == 1:
            n1, n2 = np.mgrid[-idx : idx + 1, -idx : idx + 1]
            n_vec = [n1, n2]
        elif len(fundamental_tunes) == 2:
            n1, n2, n3 = np.mgrid[-idx : idx + 1, -idx : idx + 1, -idx : idx + 1]
            n_vec = [n1, n2, n3]
        else:
            n1, n2, n3, n4 = np.mgrid[
                -idx : idx + 1, -idx : idx + 1, -idx : idx + 1, -idx : idx + 1
            ]
            n_vec = [n1, n2, n3, n4]
    else:
        assert (
            len(n_vec) == len(fundamental_tunes) + 1
        ), "n_vec must be of length fundamental_tunes+1"

    # Computing all linear combinations of n1*Qx + n2*Qy + n3*Qz + n4
    Q_vec = fundamental_tunes + [1]
    all_combinations = sum([_n * _Q for _n, _Q in zip(n_vec, Q_vec)])

    # Find the closest combination for each frequency
    n_values = []
    f_values = []
    err = []
    for freq in frequencies:

        # Find the index of the closest combination
        closest_idx = np.unravel_index(
            np.argmin(np.abs(freq - all_combinations)), all_combinations.shape
        )

        # Get the corresponding values for n1,n2,n3,n4
        closest_combination = tuple(_n[closest_idx] for _n in n_vec)
        closest_value = all_combinations[closest_idx]

        n_values.append(closest_combination)
        f_values.append(closest_value)
        err.append(np.abs(closest_value - freq))

    if to_pandas:
        import pandas as pd

        return pd.DataFrame({"resonance": n_values, "err": err, "frequency": f_values})
    else:
        return [tuple(_n) for _n in n_values], np.array(err), np.array(f_values)


# ---------------------------------------


# ---------------------------------------
def generate_signal(amplitudes, frequencies, N):
    """
    Generates a complex signal from given amplitudes and frequencies over a range of turns.

    Parameters
    ----------
    amplitudes : list or float
        Amplitudes of the components of the signal.
    frequencies : list or float
        Frequencies of the components of the signal.
    N : ndarray
        Array of turn numbers for which the signal is generated.

    Returns
    -------
    x,px : tuple of ndarray
        Arrays representing the position and momentum of the particle at each turn.
    """

    if isinstance(amplitudes, (float, int)):
        amplitudes = [amplitudes]
    if isinstance(frequencies, (float, int)):
        frequencies = [frequencies]

    assert len(amplitudes) == len(
        frequencies
    ), "Amplitudes and frequencies must have the same length"

    signal = sum(
        [
            A * np.exp(1j * (2 * np.pi * (Q) * N))
            for A, Q in zip(amplitudes, frequencies)
        ]
    )
    x = signal.real
    px = -signal.imag

    return x, px


# ---------------------------------------


# ---------------------------------------
def generate_pure_KAM(
    amplitudes, combinations, fundamental_tunes, N, return_frequencies=False
):
    """
    Generates a signal using the Kolmogorov-Arnold-Moser (KAM) theorem, simulating resonances.

    Parameters
    ----------
    amplitudes : list or float
        Amplitudes of the combinations.
    combinations : list of tuples or tuple
        Combination indices for linear combination of the fundamental tunes.
    fundamental_tunes : list
        Fundamental tunes for linear combinations.
    N : ndarray
        Array of turn numbers for signal generation.
    return_frequencies : bool, optional
        If True, also returns the frequencies used for signal generation.

    Returns
    -------
    x,px : tuple
        Arrays representing the position and momentum of the particle at each turn and, optionally, frequencies used for generation.
    """

    if isinstance(amplitudes, (float, int)):
        amplitudes = [amplitudes]
    if isinstance(combinations, (float, int)):
        combinations = [combinations]

    assert len(amplitudes) == len(
        combinations
    ), "amplitudes and resonances must have the same length"

    # Computing the frequencies
    Q_vec = fundamental_tunes + [1]
    n_vec = combinations
    assert (
        len(Q_vec) == np.shape(n_vec)[1]
    ), "combinations should have n+1 indices if n fundamental tunes are provided"
    frequencies = [np.dot(_r, Q_vec) for _r in n_vec]

    # Generating the signal
    signal = sum(
        [
            A * np.exp(1j * (2 * np.pi * (Q) * N))
            for A, Q in zip(amplitudes, frequencies)
        ]
    )
    x = signal.real
    px = -signal.imag

    if return_frequencies:
        return x, px, frequencies
    else:
        return x, px


# ---------------------------------------



#===================================
# Evaluation of tune 6D
    
def remove_from_list(l,elem):
    _l = l.copy()
    if elem in _l:
        _l.remove(elem)
    return _l

def find_sidebands(A,Q,Qs,sideband_tol = 1e-1,power_fraction=1.0,min_separation=None,main_lines_iterations=3):

     # If nans
    if np.all(np.isnan(A)) or np.all(np.isnan(Q)):
        return [np.nan],[np.nan]


    # Keeping relevant power fraction and sorting by frequency
    #-------------------------------------
    n_harm = np.sum(np.cumsum(np.abs(A)**2/np.sum(np.abs(A)**2))<=power_fraction) + 1
    A_srt    = np.array(A)[np.argsort(Q[:n_harm])]
    Q_srt    = np.array(Q)[np.argsort(Q[:n_harm])]


    # Excludng too-close peaks, keeping only main lines (iterate few times, to avoid keeping some noisy peaks)
    #-------------------------------------
    if min_separation is None:
        min_separation = np.abs(Qs)/2
    for _ in range(main_lines_iterations):
        idx_vec     = np.arange(len(Q_srt))
        freq_jumps  = [np.abs(Q_srt-_Q) for _Q in Q_srt]
        main_lines  = list(set([idx_vec[jump<min_separation][np.argmax(np.abs(A_srt[jump<min_separation]))] for jump in freq_jumps]))
        #----
        A_srt    = A_srt[main_lines]
        Q_srt    = Q_srt[main_lines]


    # Keeping only sidebands
    #-------------------------------------
    idx_vec     = np.arange(len(Q_srt))
    freq_jumps  = [np.abs(Q_srt-_Q) for _Q in Q_srt]
    sband_groups= [remove_from_list(idx_vec[np.abs(_jump/Qs - np.round(_jump/Qs))<sideband_tol].tolist(),idx) for idx,_jump in enumerate(freq_jumps)]
    sidebands   = list(set(sum(sband_groups,[])))
    #----
    A_srt    = A_srt[sidebands]
    Q_srt    = Q_srt[sidebands]
    

    # Returning
    return A_srt,Q_srt


def LMS_penalty(residuals):
    return np.log10(1e-16 + np.sqrt(1/(len(residuals)-1)* np.sum(residuals**2)))



def evaluate_penalty(A_srt,Q_srt,As,Qs,sideband_tol=1e-1,q_max=2,w_kink=False):
    idx_vec     = np.arange(len(Q_srt))
    freq_jumps  = [np.abs(Q_srt-_Q) for _Q in Q_srt]


    sband_groups= [idx_vec[np.abs(_jump/Qs - np.round(_jump/Qs))<sideband_tol].tolist() for idx,_jump in enumerate(freq_jumps)]

    penalty_values = []
    for (self_idx,_A),_Q,sband in zip(enumerate(A_srt),Q_srt,sband_groups):


        # Extracting relevant phasors
        observed = A_srt[sband]/np.abs(A_srt[sband])
        q_values = np.round((Q_srt[sband] - _Q)/Qs).astype(int)


        # Keeping only few lines around:
        observed  = observed[np.abs(q_values)<q_max]
        q_values  = q_values[np.abs(q_values)<q_max]

        # Referencing to central line
        observed = observed/observed[q_values==0]

        # Adding penalty for too few lines
        if len(q_values) <= 2:
            penalty_values.append(np.inf)
        
        # Making sure we don't have fully asymmetric lines (linear trend would then be easily satisfied)
        elif np.abs(sum(q_values))>=q_max:
            penalty_values.append(np.inf)
        else:

            if w_kink:
                target = np.exp(1j*(-np.abs(q_values)*np.pi/2)) * (As/np.abs(As))**(q_values) * np.exp(1j*np.pi*(q_values!=0).astype(int))
            else:
                target = np.exp(1j*(-np.abs(q_values)*np.pi/2)) * (As/np.abs(As))**(q_values)
            
            

            residuals = np.abs(np.angle(observed) - np.angle(target))
            penalty_values.append(LMS_penalty(residuals))


    return penalty_values


def find_tune_6D(A,Q,As,Qs,single_line_tol = 0.6,sideband_tol = 1e-1,power_fraction=1.0,min_separation=None,main_lines_iterations=3,force_positive  = False,q_max=2,w_kink=False):
    
    if force_positive:
        A = np.array(A)[Q>10*np.abs(Qs)]
        Q = np.array(Q)[Q>10*np.abs(Qs)]

    # If single dominant like, return it
    if np.abs(A[0])**2/np.sum(np.abs(A)**2) > single_line_tol:
        return Q[0]
    
    # If nans
    if np.all(np.isnan(A)) or np.all(np.isnan(Q)):
        return np.nan


    # Else, let's use the phase difference from the sidebands
    A_srt,Q_srt = find_sidebands(np.array(A)[np.invert(np.isnan(Q))],np.array(Q)[np.invert(np.isnan(Q))],Qs,sideband_tol=sideband_tol,power_fraction=power_fraction,min_separation=min_separation,main_lines_iterations=main_lines_iterations)

    

    # Evaluating the penalty for each Q tested as tune
    penalty_values = evaluate_penalty(A_srt,Q_srt,As,Qs,sideband_tol=sideband_tol,q_max=q_max,w_kink=w_kink)

    # Returning winner
    tune_idx = np.argmin(penalty_values)

    return Q_srt[tune_idx]
   
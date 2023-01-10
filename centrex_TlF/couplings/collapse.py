import copy

import numpy as np
from ..couplings.branching import calculate_BR
from ..couplings.utils_compact import compact_C_array, compact_C_array_indices
from ..states.utils import QuantumSelector, get_indices_quantumnumbers
from ..states.utils_compact import compact_QN_coupled_indices
from tqdm import tqdm

__all__ = ["collapse_matrices"]


def collapse_matrices(
    QN,
    ground_states,
    excited_states,
    gamma=1,
    tol=1e-4,
    progress=False,
    slice_compact=None,
    qn_compact=None,
):
    """
    Function that generates the collapse matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states
    excited_states = list of excited states that are coupled to the ground states
    gamma = decay rate of excited states
    tol = couplings smaller than tol/sqrt(gamma) are set to zero to speed up computation
    progress = boolean flag to display a tqdm progress bar
    slice_compact = np._s of indices to compact into one state
    qn_compact = list of QuantumSelectors or lists of QuantumSelectors with each
                QuantumSelector containing the quantum numbers to compact into a
                single state. Defaults to None.

    outputs:
    C_list = array of collapse matrices
    """
    # Initialize list of collapse matrices
    C_list = []

    # Start looping over ground and excited states
    for excited_state in tqdm(excited_states, disable=not progress):
        j = QN.index(excited_state)
        BRs = calculate_BR(excited_state, ground_states)
        if np.sum(BRs) > 1:
            print(f"Warning: Branching ratio sum > 1, difference = {np.sum(BRs)-1:.2e}")
        for ground_state, BR in zip(ground_states, BRs):
            i = QN.index(ground_state)

            if np.sqrt(BR) > tol:
                # Initialize the coupling matrix
                H = np.zeros((len(QN), len(QN)), dtype=complex)
                H[i, j] = np.sqrt(BR * gamma)

                C_list.append(H)

    C_array = np.array(C_list)

    if slice_compact:
        C_array = compact_C_array(C_array, gamma, slice_compact)
    elif qn_compact:
        if isinstance(qn_compact, QuantumSelector):
            qn_compact = [qn_compact]
        QN_compact = copy.deepcopy(QN)
        for qnc in qn_compact:
            indices_compact = get_indices_quantumnumbers(qnc, QN_compact)
            QN_compact = compact_QN_coupled_indices(QN_compact, indices_compact)
            C_array = compact_C_array_indices(C_array, gamma, indices_compact)
    return C_array

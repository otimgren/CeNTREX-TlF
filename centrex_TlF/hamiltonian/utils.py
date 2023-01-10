from functools import lru_cache

import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j

from ..states.states import State

__all__ = [
    "reorder_evecs",
    "generate_uncoupled_hamiltonian_X_function",
    "generate_coupled_hamiltonian_B_function",
    "matrix_to_states",
    "reduced_basis_hamiltonian",
    "threej_f",
    "sixj_f",
]


@lru_cache(maxsize=int(1e6))
def threej_f(j1, j2, j3, m1, m2, m3):
    return complex(wigner_3j(j1, j2, j3, m1, m2, m3))


@lru_cache(maxsize=int(1e6))
def sixj_f(j1, j2, j3, j4, j5, j6):
    return complex(wigner_6j(j1, j2, j3, j4, j5, j6))


def reorder_evecs(V_in, E_in, V_ref):
    """Reshuffle eigenvectors and eigenergies based on a reference

    Args:
        V_in (np.ndarray): eigenvector matrix to be reorganized
        E_in (np.ndarray): energy vector to be reorganized
        V_ref (np.ndarray): reference eigenvector matrix

    Returns:
        (np.ndarray, np.ndarray): energy vector, eigenvector matrix
    """
    # take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T), V_ref))

    # find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors, axis=1))
    # store energy and state
    E_out = E_in[index]
    V_out = V_in[:, index]

    return E_out, V_out


def generate_uncoupled_hamiltonian_X_function(H):
    ham_func = (
        lambda E, B: 2
        * np.pi
        * (
            H["Hff"]
            + E[0] * H["HSx"]
            + E[1] * H["HSy"]
            + E[2] * H["HSz"]
            + B[0] * H["HZx"]
            + B[1] * H["HZy"]
            + B[2] * H["HZz"]
        )
    )
    return ham_func


def generate_coupled_hamiltonian_B_function(H):
    ham_func = (
        lambda E, B: 2
        * np.pi
        * (
            H["Hrot"]
            + H["H_mhf_Tl"]
            + H["H_mhf_F"]
            + H["H_LD"]
            + H["H_cp1_Tl"]
            + H["H_c_Tl"]
            + 0.01 * H["HZz"]
        )
    )
    return ham_func


def matrix_to_states(V, QN, E=None):
    """Turn a matrix of eigenvectors into a list of state objects

    Args:
        V (np.ndarray): array with columns corresponding to eigenvectors
        QN (list): list of State objects
        E (list, optional): list of energies corresponding to the states.
                            Defaults to None.

    Returns:
        list: list of eigenstates expressed as State objects
    """
    # find dimensions of matrix
    matrix_dimensions = V.shape

    # initialize a list for storing eigenstates
    eigenstates = []

    for i in range(0, matrix_dimensions[1]):
        # find state vector
        state_vector = V[:, i]

        # ensure that largest component has positive sign
        index = np.argmax(np.abs(state_vector))
        state_vector = state_vector * np.sign(state_vector[index])

        data = []

        # get data in correct format for initializing state object
        for j, amp in enumerate(state_vector):
            data.append((amp, QN[j]))

        # store the state in the list
        state = State(data)

        if E is not None:
            state.energy = E[i]

        eigenstates.append(state)

    # return the list of states
    return eigenstates


def reduced_basis_hamiltonian(basis_ori, H_ori, basis_red):
    """Generate Hamiltonian for a sub-basis of the original basis

    Args:
        basis_ori (list): list of states of original basis
        H_ori (np.ndarray): original Hamiltonian
        basis_red (list): list of states of sub-basis

    Returns:
        np.ndarray: Hamiltonian in sub-basis
    """

    # Determine the indices of each of the reduced basis states
    index_red = np.zeros(len(basis_red), dtype=int)
    for i, state_red in enumerate(basis_red):
        index_red[i] = basis_ori.index(state_red)

    # Initialize matrix for Hamiltonian in reduced basis
    H_red = np.zeros((len(basis_red), len(basis_red)), dtype=complex)

    # Loop over reduced basis states and pick out the correct matrix elements
    # for the Hamiltonian in the reduced basis
    for i, state_i in enumerate(basis_red):
        for j, state_j in enumerate(basis_red):
            H_red[i, j] = H_ori[index_red[i], index_red[j]]

    return H_red

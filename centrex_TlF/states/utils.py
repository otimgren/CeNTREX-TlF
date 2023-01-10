from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import List, SupportsFloat, Union

import numpy as np
from centrex_TlF.hamiltonian.utils import reorder_evecs
from centrex_TlF.states.states import CoupledBasisState, State
from sympy.physics.quantum.cg import CG

__all__ = [
    "find_state_idx_from_state",
    "find_exact_states",
    "check_approx_state_exact_state",
    "parity_X",
    "BasisStates_from_State",
    "CGc",
    "find_states_idxs_from_states",
    "get_indices_quantumnumbers",
    "QuantumSelector",
    "get_unique_basisstates",
    "SystemParameters",
]


@lru_cache(maxsize=int(1e6))
def CGc(j1, m1, j2, m2, j3, m3):
    return complex(CG(j1, m1, j2, m2, j3, m3).doit())


def parity_X(J):
    return (-1) ** J


def find_states_idxs_from_states(H, reference_states, QN, V_ref=None):
    # find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)

    if V_ref is not None:
        E, V = reorder_evecs(V, E, V_ref)

    indices = []
    for reference_state in reference_states:
        # determine state vector of reference state
        reference_state_vec = reference_state.state_vector(QN)

        overlaps = np.dot(np.conj(reference_state_vec), V)
        probabilities = overlaps * np.conj(overlaps)

        indices.append(np.argmax(probabilities))
    return indices


def find_state_idx_from_state(H, reference_state, QN, V_ref=None):
    """Determine the index of the state vector most closely corresponding to an
    input state

    Args:
        H (np.ndarray): Hamiltonian to compare to
        reference_state (State): state to find closest state in H to
        QN (list): list of state objects defining the basis for H

    Returns:
        int: index of closest state vector of H corresponding to reference_state
    """

    # determine state vector of reference state
    reference_state_vec = reference_state.state_vector(QN)

    # find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)

    if V_ref is not None:
        E, V = reorder_evecs(V, E, V_ref)

    overlaps = np.dot(np.conj(reference_state_vec), V)
    probabilities = overlaps * np.conj(overlaps)

    idx = np.argmax(probabilities)

    return idx


def find_closest_vector_idx(state_vec, vector_array):
    """ Function that finds the index of the vector in vector_array that most closely
    matches state_vec. vector_array is array where each column is a vector, typically
    corresponding to an eigenstate of some Hamiltonian.

    inputs:
    state_vec = Numpy array, 1D
    vector_array = Numpy array, 2D

    returns:
    idx = index that corresponds to closest matching vector
    """

    overlaps = np.abs(state_vec.conj().T @ vector_array)
    idx = np.argmax(overlaps)

    return idx


def find_exact_states(states_approx, H, QN, V_ref=None):
    """Find closest approximate eigenstates corresponding to states_approx

    Args:
        states_approx (list): list of State objects
        H (np.ndarray): Hamiltonian, diagonal in basis QN
        QN (list): list of State objects defining the basis for H

    Returns:
        list: list of eigenstates of H closest to states_approx
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref)
        states.append(QN[i])

    return states


def check_approx_state_exact_state(approx, exact):
    """Check if the exact found states match the approximate states. The exact
    states are found from the eigenvectors of the hamiltonian and are often a
    superposition of various states.
    The approximate states are used in initial setup of the hamiltonian.

    Args:
        approx (State): approximate state
        exact (State): exact state
    """
    approx = approx.find_largest_component()
    exact = exact.find_largest_component()

    assert approx.electronic_state == exact.electronic_state, (
        f"mismatch in electronic state {approx.electronic_state} != "
        f"{exact.electronic_state}"
    )
    assert approx.J == exact.J, f"mismatch in J {approx.J} != {exact.J}"
    assert approx.F == exact.F, f"mismatch in F {approx.F} != {exact.F}"
    assert approx.F1 == exact.F1, f"mismatch in F1 {approx.F1} != {exact.F1}"
    assert approx.mF == exact.mF, f"mismatch in mF {approx.mF} != {exact.mF}"


def BasisStates_from_State(states):
    if not isinstance(states, (list, np.ndarray, tuple)):
        states = [states]
    unique = []
    for state in states:
        for amp, basisstate in state:
            if basisstate not in unique:
                unique.append(basisstate)
    return np.array(unique)


NumberType = type(SupportsFloat)


@dataclass
class QuantumSelector:
    """Class for setting quantum numbers for selecting a subset of states
    from a larger set of states

    Args:
        J (Union[NumberType, list, np.ndarray]): rotational quantum number
        F1 (Union[NumberType, list, np.ndarray]):
        F (Union[NumberType, list, np.ndarray]):
        mF (Union[NumberType, list, np.ndarray]):
        electronic (Union[str, list, np.ndarray]): electronic state
    """

    J: Union[NumberType, list, np.ndarray] = None
    F1: Union[NumberType, list, np.ndarray] = None
    F: Union[NumberType, list, np.ndarray] = None
    mF: Union[NumberType, list, np.ndarray] = None
    electronic: Union[str, list, np.ndarray] = None
    P: Union[NumberType, list, np.ndarray] = None
    Ω: Union[NumberType, list, np.ndarray] = None

    def get_indices(self, QN, mode="python"):
        return get_indices_quantumnumbers_base(self, QN, mode)

    def get_states(self, QN, mode="python"):
        """
        Returns the states in QN that fit the pattern specified in the QuantumSelector
        """
        indices = self.get_indices(QN)
        return QN[indices]


@dataclass
class SystemParameters:
    nprocs: int
    Γ: float
    ground: Union[list, np.ndarray, QuantumSelector]
    excited: Union[list, np.ndarray, QuantumSelector]


def get_indices_quantumnumbers_base(
    qn_selector: QuantumSelector, QN: Union[list, np.ndarray], mode: str = "python"
) -> np.ndarray:
    """Return the indices corresponding to all states in QN that correspond to
    the quantum numbers in QuantumSelector.
    Entries in QuantumSelector quantum numbers can be either single numbers or
    lists/arrays. States with all possible combinations of quantum numbers in
    QuantumSelector are found

    Args:
        qn_selector (QuantumSelector): QuantumSelector class containing the
                                        quantum numbers to find corresponding
                                        indices for
        QN (Union[list, np.ndarray]): list or array of states

    Raises:
        AssertionError: only supports State and CoupledBasisState types the States list
        or array

    Returns:
        np.ndarray: indices corresponding to the quantum numbers
    """
    assert isinstance(
        qn_selector, QuantumSelector
    ), "supply a QuantumSelector object to select states"
    if isinstance(QN[0], State):
        Js = np.array([s.find_largest_component().J for s in QN])
        F1s = np.array([s.find_largest_component().F1 for s in QN])
        Fs = np.array([s.find_largest_component().F for s in QN])
        mFs = np.array([s.find_largest_component().mF for s in QN])
        estates = np.array([s.find_largest_component().electronic_state for s in QN])
    elif isinstance(QN[0], CoupledBasisState):
        Js = np.array([s.J for s in QN])
        F1s = np.array([s.F1 for s in QN])
        Fs = np.array([s.F for s in QN])
        mFs = np.array([s.mF for s in QN])
        estates = np.array([s.electronic_state for s in QN])
    else:
        raise AssertionError(
            "get_indices_quantumnumbers_base() only supports State and "
            "CoupledBasisState types the States list or array"
        )

    J = qn_selector.J
    F1 = qn_selector.F1
    F = qn_selector.F
    mF = qn_selector.mF
    estate = qn_selector.electronic
    assert estate is not None, "supply the electronic state to select states"

    # generate all combinations
    fields = []
    for par in ["J", "F1", "F", "mF", "electronic"]:
        par = getattr(qn_selector, par)
        fields.append([par] if not isinstance(par, (list, tuple, np.ndarray)) else par)
    combinations = product(*fields)

    mask = np.zeros(len(QN), dtype=bool)
    mask_all = np.ones(len(QN), dtype=bool)
    for J, F1, F, mF, estate in combinations:
        # generate the masks for states in QN where the conditions are met
        mask_J = Js == J if J is not None else mask_all
        mask_F1 = F1s == F1 if F1 is not None else mask_all
        mask_F = Fs == F if F is not None else mask_all
        mask_mF = mFs == mF if mF is not None else mask_all
        mask_es = (
            estates == estate if estate is not None else np.zeros(len(QN), dtype=bool)
        )
        # get the indices of the states in QN to compact
        mask = mask | (mask_J & mask_F1 & mask_F & mask_mF & mask_es)

    if mode == "python":
        return np.where(mask)[0]


def get_indices_quantumnumbers(
    qn_selector: Union[QuantumSelector, list, np.ndarray], QN: Union[list, np.ndarray]
) -> np.ndarray:
    """return the indices corresponding to all states in QN that correspond to
    the quantum numbers in QuantumSelector or a list of QuantumSelector objects.
    Entries in QuantumSelector quantum numbers can be either single numbers or
    lists/arrays. States with all possible combinations of quantum numbers in
    QuantumSelector are found

    Args:
        qn_selector (Union[QuantumSelector, list, np.ndarray]):
                    QuantumSelector class or list/array of QuantumSelectors
                    containing the quantum numbers to find corresponding indices

        QN (Union[list, np.ndarray]): list or array of states

    Raises:
        AssertionError: only supports State and CoupledBasisState types the States list
        or array

    Returns:
        np.ndarray: indices corresponding to the quantum numbers
    """
    if isinstance(qn_selector, QuantumSelector):
        return get_indices_quantumnumbers_base(qn_selector, QN)
    elif isinstance(qn_selector, (list, np.ndarray)):
        return np.unique(
            np.concatenate(
                [get_indices_quantumnumbers_base(qns, QN) for qns in qn_selector]
            )
        )
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )


def get_unique_basisstates(states: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """get a list/array of unique BasisStates in the states list/array

    Args:
        states (Union[list, np.ndarray]): list/array of BasisStates

    Returns:
        Union[list, np.ndarray]: list/array of unique BasisStates
    """
    states_unique = []
    for state in states:
        if state not in states_unique:
            states_unique.append(state)

    if isinstance(states, np.ndarray):
        states_unique = np.asarray(states_unique)

    return states_unique


def matrix_to_states(V: np.ndarray, QN: List)-> List[State]:
    """
    Converts matrix of eigenvectors (each column a vector) to a list of states
    in the basis defined by QN

    Args:
        V (np.ndarray)  :   numpy array to be converted to State-objects.
                            Each column corresponds to a state vector.
        QN (List)       :   List of BasisStates that defines the basis for
                            the state vectors
    Returns:
        List[State]     :   List of States corresponding to columns of V    
    """

    #Find dimensions of matrix
    matrix_dimensions = V.shape
    
    #Initialize a list for storing eigenstates
    eigenstates = []
    
    for i in range(0,matrix_dimensions[1]):
        #Find state vector
        state_vector = V[:,i]

        #Ensure that largest component has positive sign
        index = np.argmax(np.abs(state_vector))
        state_vector = state_vector * np.sign(state_vector[index])
        
        state = State()
        
        #Get data in correct format for initializing state object
        for j, amp in enumerate(state_vector):
            state += amp*QN[j]

        #Store the state in the list
        eigenstates.append(state)
        
    
    #Return the list of states
    return eigenstates

def vector_to_state(state_vector: np.ndarray, QN: List) -> State:
    """
    Converts a state vector into a State in the basis defined by QN.

    Args:
        state_vector (np.ndarray):   
            numpy array to be converted to State.
        QN (List):
            List of BasisStates that defines the basis for
            the state vector
    Returns:
        State:
            State corresponding to state vector       

    """
    state = State()
    
    #Get data in correct format for initializing state object
    for j, amp in enumerate(state_vector):
        state += amp*QN[j]
        
    return state

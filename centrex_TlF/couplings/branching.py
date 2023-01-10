import numpy as np

from ..couplings.matrix_elements import generate_ED_ME_mixed_state

__all__ = ["calculate_BR"]


def calculate_BR(excited_state, ground_states, tol=1e-5):
    """
    Function that calculates branching ratios from the given excited state to
    the given ground states

    inputs:
    excited_state = state object representing the excited state that is
                    spontaneously decaying
    ground_states = list of state objects that should span all the states to
                    which the excited state can decay

    returns:
    BRs = list of branching ratios to each of the ground states
    """

    # Initialize container for matrix elements between excited state and ground
    # states
    MEs = np.zeros(len(ground_states), dtype=complex)

    # loop over ground states
    for i, ground_state in enumerate(ground_states):
        MEs[i] = generate_ED_ME_mixed_state(
            ground_state.remove_small_components(tol=tol),
            excited_state.remove_small_components(tol=tol),
        )

    # Calculate branching ratios
    BRs = np.abs(MEs) ** 2 / (np.sum(np.abs(MEs) ** 2))

    return BRs

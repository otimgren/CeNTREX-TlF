import copy
import numpy as np

from centrex_TlF.couplings.collapse import collapse_matrices
__all__ = [
    ''
]

def delete_row_column_symbolic(arr, sl):
    arr_copy = arr.copy()
    sl = np.s_[sl.start:sl.stop-1]
    deleted = 0
    for idx in range(sl.start, sl.stop):
        arr_copy.row_del(idx-deleted)
        arr_copy.col_del(idx-deleted)
        deleted += 1
    return arr_copy

def compact_symbolic_hamiltonian_indices(hamiltonian, indices_compact):
    """compact a sympy hamiltonian by combining all indices in indices_compact
    into a single state

    Args:
        hamiltonian (sympy matrix): hamiltonian
        indices_compact (list, array): indices to compact

    Returns:
        sympy matrix: compacted hamiltonian
    """
    arr = hamiltonian.copy()
    diagonal = arr.diagonal()
    diagonal = [diagonal[idd] for idd in indices_compact]
    check_free_symbols = np.sum([len(val.free_symbols) for val in diagonal])
    assert check_free_symbols == 0, 'diagonal elements for states to compact have symbols, cannot compact'

    # delete the rows and columns to compact, except a single one that's needed
    # to put the decays into
    deleted = 0
    for idx in indices_compact[1:]:
        row = arr[idx-deleted,:]
        col = arr[:,idx-deleted]
        # check if couplings are present, raise AssertionError if true
        assert (np.sum(row) - row[idx-deleted]) == 0, 'couplings exist for states to compact, cannot compact'
        assert (np.sum(col) - row[idx-deleted]) == 0, 'couplings exist for states to compact, cannot compact'
        arr.row_del(idx-deleted)
        arr.col_del(idx-deleted)
        deleted += 1

    # setting the diagonal element to be the mean off the entire state
    # pretty much irrelevant since the states only have decays and should be
    # far enough away from the others
    arr[idx-deleted, idx-deleted] = np.mean(diagonal)
    return arr

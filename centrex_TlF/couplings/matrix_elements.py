import numpy as np

from ..couplings.utils_sqlite import (retrieve_ED_ME_coupled_sqlite_single,
                                      retrieve_ED_ME_coupled_sqlite_single_rme)
from ..hamiltonian.utils import sixj_f, threej_f


def calculate_ED_ME_mixed_state(
    bra, ket, pol_vec=np.array([1, 1, 1]), reduced=False, normalize_pol=True
):
    """calculate electric dipole matrix elements between mixed states

    Args:
        bra (State): state object
        ket (State): state object
        pol_vec (np.ndarray, optional): polarization vector.
                                        Defaults to np.array([1,1,1]).
        reduced (bool, optional): [description]. Defaults to False.
        normalize_pol (bool, optional): normalize the polarization vector to 1,
                                        defaults to True.

    Returns:
        complex: matrix element between bra and ket
    """
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()

    if normalize_pol:
        pol_vec = np.asarray(pol_vec) / np.linalg.norm(pol_vec)

    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += (
                amp_bra.conjugate()
                * amp_ket
                * ED_ME_coupled(basis_bra, basis_ket, pol_vec=pol_vec, rme_only=reduced)
            )

    return ME


def generate_ED_ME_mixed_state(
    bra, ket, pol_vec=np.array([1, 1, 1]), reduced=False, normalize_pol=True, con=None
):
    """calculate electric dipole matrix elements between mixed states

    Args:
        bra (State): state object
        ket (State): state object
        pol_vec (np.ndarray, optional): polarization vector.
                                        Defaults to np.array([1,1,1]).
        reduced (bool, optional): [description]. Defaults to False.
        normalize_pol (bool, optional): normalize the polarization vector to 1,
                                        defaults to True.

    Returns:
        complex: matrix element between bra and ket
    """
    if not con:
        return calculate_ED_ME_mixed_state(bra, ket, pol_vec, reduced, normalize_pol)
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()

    if normalize_pol:
        pol_vec = np.asarray(pol_vec) / np.linalg.norm(pol_vec)

    if reduced:
        retrieve = retrieve_ED_ME_coupled_sqlite_single_rme
    else:
        retrieve = retrieve_ED_ME_coupled_sqlite_single

    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += (
                amp_bra.conjugate()
                * amp_ket
                * retrieve(basis_bra, basis_ket, pol_vec=pol_vec, con=con)
            )
    return ME


def ED_ME_coupled(bra, ket, pol_vec=np.array([1, 1, 1]), rme_only=False):
    """calculate electric dipole matrix elements between coupled basis states

    Args:
        bra (CoupledBasisState): coupled basis state object
        ket (CoupledBasisState): coupled basis state object
        pol_vec (np.ndarray, optional): polarization vector.
                                        Defaults to np.array([1,1,1]).
        rme_only (bool, optional): set True to return only reduced matrix
                                    element, otherwise angular component is
                                    included. Defaults to False.

    Returns:
        complex: electric dipole matrix element between bra en ket
    """

    # find quantum numbers for ground state
    F = bra.F
    mF = bra.mF
    J = bra.J
    F1 = bra.F1
    I1 = bra.I1
    I2 = bra.I2
    Omega = bra.Omega

    # find quantum numbers for excited state
    Fp = ket.F
    mFp = ket.mF
    Jp = ket.J
    F1p = ket.F1
    Omegap = ket.Omega

    # calculate the reduced matrix element
    q = Omega - Omegap
    ME = (
        (-1) ** (F1 + J + Fp + F1p + I1 + I2)
        * np.sqrt((2 * F + 1) * (2 * Fp + 1) * (2 * F1p + 1) * (2 * F1 + 1))
        * sixj_f(F1p, Fp, I2, F, F1, 1)
        * sixj_f(Jp, F1p, I1, F1, J, 1)
        * (-1) ** (J - Omega)
        * np.sqrt((2 * J + 1) * (2 * Jp + 1))
        * threej_f(J, 1, Jp, -Omega, q, Omegap)
        * float(np.abs(q) < 2)
    )

    # if we want the complete matrix element, calculate angular part
    if not rme_only:

        # calculate elements of the polarization vector in spherical basis
        p_vec = {}
        p_vec[-1] = -1 / np.sqrt(2) * (pol_vec[0] + 1j * pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1 / np.sqrt(2) * (pol_vec[0] - 1j * pol_vec[1])

        # calculate the value of p that connects the states
        p = mF - mFp
        p = p * int(np.abs(p) <= 1)
        # multiply RME by the angular part
        ME = (
            ME
            * (-1) ** (F - mF)
            * threej_f(F, 1, Fp, -mF, p, mFp)
            * p_vec[p]
            * int(np.abs(p) <= 1)
        )

    # return the matrix element
    return ME

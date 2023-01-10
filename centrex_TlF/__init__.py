import logging

from rich.logging import RichHandler

from . import constants, couplings, hamiltonian, states, transitions, utils
from .states.states import CoupledBasisState, State, UncoupledBasisState

FORMAT = "%(message)s"
logging.basicConfig(
    level="WARNING", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

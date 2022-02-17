"""
# liionpack

liionpack is a tool for simulating battery packs with pybamm. It can design the
pack with a combination of batteries connected in series and parallel or can
read a netlist.
"""
from .simulations import basic_simulation
from .simulations import thermal_simulation
from .utils import interp_current
from .utils import build_inputs_dict
from .utils import add_events_to_model
from .netlist_utils import read_netlist
from .netlist_utils import setup_circuit
from .netlist_utils import solve_circuit
from .netlist_utils import solve_circuit_vectorized
from .netlist_utils import make_lcapy_circuit
from .sim_utils import get_initial_stoichiometries
from .sim_utils import update_init_conc
from .solver_utils import solve
from .protocols import generate_protocol_from_experiment
from .plots import draw_circuit
from .plots import plot_pack
from .plots import plot_cells
from .plots import plot_output
from .plots import show_plots
from .plots import simple_netlist_plot
from .plots import compare_solution_output
from .plots import plot_cell_data_image
from .plots import lp_cmap
from .plots import lp_context
from .logger import logger, set_logging_level, log_to_file
from .definitions import ROOT_DIR
from .definitions import MODULE_DIR
from .definitions import CIRCUIT_DIR
from .solvers import casadi_manager
from .solvers import ray_manager
from .solvers import dask_manager
from .solvers import generic_actor
from .solvers import ray_actor

__version__ = "0.3"

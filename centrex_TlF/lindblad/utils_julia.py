import numpy as np
import sympy as smp
from julia import Main
from pathlib import Path
from sympy import Symbol

__all__ = [
    'initialize_julia', 'generate_ode_fun_julia', 'setup_variables_julia',
    'odeParameters', 'setup_parameter_scan_1D', 'setup_ratio_calculation',
    'setup_initial_condition_scan', 'setup_state_integral_calculation',
    'get_indices_diag_flattened', 'setup_parameter_scan_ND', 
    "handle_randomized_ensemble_solution", "setup_discrete_callback_terminate",
    "setup_problem", "solve_problem", "get_results", "do_simulation_single", 
    "setup_problem_parameter_scan", "solve_problem_parameter_scan", 
    "get_results_parameter_scan"
]

def initialize_julia(nprocs):
    Main.eval("""
        using Logging: global_logger
        using TerminalLoggers: TerminalLogger
        global_logger(TerminalLogger())

        using Distributed
    """)

    if Main.eval("nprocs()") < nprocs:
        Main.eval(f"addprocs({nprocs}-nprocs())")

    if Main.eval("nprocs()") > nprocs:
        procs = Main.eval("procs()")
        procs = procs[nprocs:]
        Main.eval(f"rmprocs({procs})")

    Main.eval("""
        @everywhere begin
            using LinearAlgebra
            using Trapz
            using DifferentialEquations
        end
    """)
    # loading common julia functions from julia_common.jl
    path = Path(__file__).parent / "julia_common.jl"
    Main.eval(f'include(raw"{path}")')

    print(f"Initialized Julia with {nprocs} processes")

def generate_ode_fun_julia(preamble, code_lines):
    ode_fun = preamble
    for cline in code_lines:
        ode_fun += "\t\t"+cline+'\n'
    ode_fun += '\t end \n \t nothing \n end'
    Main.eval(f"@everywhere {ode_fun}")
    return ode_fun

def setup_variables_julia(Γ, ρ, vars = None):
    Main.Γ = Γ
    Main.ρ = ρ
    Main.eval("""
        @everywhere begin
            Γ = $Γ
            ρ = $ρ
        end
    """)
    if vars:
        for key, val in vars.items():
            Main.eval(f"@everywhere {key} = {val}")

class odeParameters:
    def __init__(self, *args, **kwargs):
        # if elif statement is for legacy support, where a list of parameters was supplied
        if len(args) > 1:
            raise AssertionError("For legacy support supply a list of strings, one for each parameter")
        elif len(args) == 1:
            assert isinstance(args[0][0], str), "For legacy support supply a list of strings, one for each parameter"
            if 'ρ' not in args[0]: args[0].append('ρ')
            kwargs = {par: 0 for par in args[0]}
            odeParameters(**kwargs)
        

        # kwargs = self._check_for_density(kwargs)
        # kwargs = self._check_for_states(kwargs)
        self._parameters = [key for key,val in kwargs.items() if not isinstance(val, str)]
        self._compound_vars = [key for key,val in kwargs.items() if isinstance(val, str)]
        
        for key,val in kwargs.items():
            # ϕ = 'ϕ') results in different unicode representation
            # replace the rhs with the rep. used on the lhs
            if isinstance(val, str):
                val = val.replace("\u03d5", "\u03c6")
            setattr(self, key, val)
        
        self._check_symbols_defined()
        self._order_compound_vars()
    
    
    def _check_for_density(self, kwargs):
        assert 'ρ' in kwargs.keys(), "Supply an initial density ρ to odeParameters"
        self.ρ = kwargs.get('ρ')
        del kwargs['ρ']
        return kwargs

    def _check_for_states(self, kwargs):
        assert 'ground' in kwargs.keys(), "Supply ground states `ground` to odeParameters"
        self.ground = kwargs.get('ground')
        del kwargs['ground']
        assert 'excited' in kwargs.keys(), "Supply excited states `excited` to odeParameters"
        self.excited = kwargs.get('excited')
        del kwargs['excited']
        return kwargs

    def __setattr__(self, name, value):
        if name in ['_parameters', '_compound_vars']:
            super(odeParameters, self).__setattr__(name, value)
        elif name == 'ρ':
            super(odeParameters, self).__setattr__(name, value)
        elif name in ["ground", "excited"]:
            super(odeParameters, self).__setattr__(name, value)
        elif name in self._parameters:
            assert not isinstance(value, str), "Cannot change parameter from numeric to str"
            super(odeParameters, self).__setattr__(name, value)
        elif name in self._compound_vars:
            assert isinstance(value, str), "Cannot change parameter from str to numeric"
            super(odeParameters, self).__setattr__(name, value)
        else:
            raise AssertionError("Cannot instantiate new parameter on initialized OdeParameters object")
    
    def _get_defined_symbols(self):
        symbols_defined = self._parameters + self._compound_vars
        symbols_defined += ['t']
        symbols_defined = set([Symbol(s) for s in symbols_defined])
        return symbols_defined
    
    def _get_numerical_symbols(self):
        symbols_numerical = self._parameters[:]
        symbols_numerical += ['t']
        symbols_numerical = set([Symbol(s) for s in symbols_numerical])
        return symbols_numerical
    
    def _get_expression_symbols(self):
        symbols_expressions = [smp.parsing.sympy_parser.parse_expr(getattr(self, s))
                                for s in self._compound_vars]
        symbols_expressions = set().union(*[s.free_symbols for s in symbols_expressions])
        return symbols_expressions
    
    def _check_symbols_defined(self):
        symbols_defined = self._get_defined_symbols()
        symbols_expressions = self._get_expression_symbols()
        
        warn_flag = False
        warn_string = f"Symbol(s) not defined: "
        for se in symbols_expressions:
            if se not in symbols_defined:
                warn_flag = True
                warn_string += f"{se}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(' ,'))
    
    def check_symbols_in_parameters(self, symbols_other):
        if not isinstance(symbols_other, (list, np.ndarray, tuple, set)):
            symbols_other = [symbols_other]
        elif isinstance(symbols_other, set):
            symbols_other = list(symbols_other)
        if isinstance(symbols_other[0], smp.Symbol):
            symbols_other = [str(sym) for sym in symbols_other]

        parameters = self._parameters[:]
        parameters += ['t']
        
        warn_flag = False
        warn_string = f"Symbol(s) not defined: "
        for se in symbols_other:
            if se not in parameters:
                warn_flag = True
                warn_string += f"{se}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(' ,'))
            
    def _order_compound_vars(self):
        symbols_num = self._get_numerical_symbols()
        unordered = list(self._compound_vars)
        ordered = []

        while len(unordered) != 0:
            for compound in unordered:
                if compound not in ordered:
                    symbols = smp.parsing.sympy_parser.parse_expr(getattr(self, compound)).free_symbols
                    m = [True if (s in symbols_num) or (str(s) in ordered) else False for s in symbols]
                    if all(m):
                        ordered.append(compound)
            unordered = [val for val in unordered if val not in ordered]
        self._compound_vars = ordered
            
    def _get_index_parameter(self, parameter, mode = 'python'):
        # OdeParameter(ϕ = 'ϕ') results in different unicode representation
        # replace the rhs with the rep. used on the lhs
        parameter = parameter.replace("\u03d5", "\u03c6")
        if mode == 'python':
            return self._parameters.index(parameter)
        elif mode == 'julia':
            return self._parameters.index(parameter)+1
    
    @property
    def p(self):
        return [getattr(self, p) for p in self._parameters]
        
        
    def get_index_parameter(self, parameter, mode = 'python'):
        if isinstance(parameter, str):
            return self._get_index_parameter(parameter, mode)
        elif isinstance(parameter, (list, np.ndarray)):
            return [self._get_index_parameter(par, mode) for par in parameter]
        
    def check_transition_symbols(self, transitions):
        # check Rabi rate and detuning symbols
        to_check = ['Ω', 'δ']
        symbols_defined = [str(s) for s in self._get_defined_symbols()]
        not_defined = []
        for transition in transitions:
            for var in to_check:
                var = str(getattr(transition,var))
                if var is not None: 
                    if var not in symbols_defined:
                        not_defined.append(var)
        if len(not_defined) > 0:
            not_defined = set([str(s) for s in not_defined])
            raise AssertionError(f"Symbol(s) from transitions not defined: {', '.join(not_defined)}")

        # check polarization switching symbols
        to_check = []
        for transition in transitions:
            to_check.extend(transition.polarization_symbols)

        symbols_defined = self._get_defined_symbols()
        
        warn_flag = False
        warn_string = f"Symbol(s) in transition polarization switching not defined: "
        for ch in to_check:
            if (ch not in symbols_defined):
                warn_flag = True
                warn_string += f"{ch}, "
        if warn_flag:
            raise AssertionError(warn_string.strip(' ,'))
        return True
        

    def generate_p_julia(self):
        Main.eval(f"p = {self.p}")
    
    def __repr__(self):
        rep = f"OdeParameters("
        for par in self._parameters:
            rep += f"{par}={getattr(self, par)}, "
        return rep.strip(", ") + ")"

def setup_parameter_scan_1D(odePar, parameter, values):
    if isinstance(parameter, (list, tuple)):
        indices = [odePar.get_index_parameter(par) for par in parameter]
    else:
        indices = [odePar.get_index_parameter(parameter)]

    pars = str(odePar.p).strip('[]').split(',')
    for idx in indices:
        pars[idx] = "params[i]"
    
    pars = "[" + ",".join(pars) + "]"
    
    Main.params = values
    Main.eval(f"""
    @everywhere params = $params
    @everywhere function prob_func(prob,i,repeat)
        remake(prob, p = {pars})
    end
    """)

def setup_parameter_scan_ND(odePar, parameters, values, randomize = False):
    pars = str(odePar.p).strip('[]').split(',')

    for idN, parameter in enumerate(parameters):
        if isinstance(parameter, (list, tuple)):
            indices = [odePar.get_index_parameter(par) for par in parameter]
        else:
            indices = [odePar.get_index_parameter(parameter)]
        for idx in indices:
            pars[idx] = f"params[i,{idN+1}]"
    pars = "[" + ",".join(pars) + "]"
    params = np.array(np.meshgrid(*values)).T.reshape(-1,len(values))
    Main.params = params
    if randomize:
        ind_random = np.random.permutation(len(params))
        Main.params = params[ind_random]

    Main.eval(f"""
    @everywhere params = $params
    @everywhere function prob_func(prob, i, repeat)
        remake(prob, p = {pars})
    end
    """)
    if randomize:
        return ind_random

def setup_ratio_calculation(states):
    cmd = ""
    if isinstance(states[0], (list, np.ndarray, tuple)):
        for state in states:
            cmd += f"sum(real(diag(sol.u[end])[{state}]))/sum(real(diag(sol.u[1])[{state}])), "
        cmd = cmd.strip(', ')
        cmd = "[" + cmd + "]"
    else:
        cmd = f"sum(real(diag(sol.u[end])[{states}]))/sum(real(diag(sol.u[1])[{states}]))"

    Main.eval(f"""
    @everywhere function output_func(sol,i)
        if size(sol.u)[1] == 1
            return NaN, false
        else
            val = {cmd}
            return val, false
        end
    end""")

def handle_randomized_ensemble_solution(ind_random, sol_name = 'sim'):
    order_restored = np.argsort(ind_random)
    # Main.params = Main.params[order_restored]
    return np.asarray(Main.params[order_restored]), np.array(Main.eval(f"{sol_name}.u"))[order_restored]

def setup_initial_condition_scan(values):
    Main.params = values
    Main.eval("@everywhere params = $params")
    Main.eval("""
    @everywhere function prob_func(prob,i,repeat)
        remake(prob,u0=params[i])
    end
    """)

def setup_state_integral_calculation(states, nphotons = False, Γ = None):
    """Setup an integration output_function for an EnsembleProblem. 
    Uses trapezoidal integration to integrate the states.
    
    Args:
        states (list): list of state indices to integrate
        nphotons (bool, optional): flag to calculate the number of photons, 
                                    e.g. normalize with Γ
        Γ (float, optional): decay rate in 2π Hz (rad/s), not necessary if already
                                loaded into Julia globals
    """
    if nphotons & Main.eval("@isdefined Γ"):
        Main.eval(f"""
        @everywhere function output_func(sol,i)
            return Γ.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
        end""")
    else:
        if nphotons:
            assert Γ is not None, "Γ not defined as a global in Julia and not supplied to function"
            Main.eval(f"""
            @everywhere function output_func(sol,i)
                return {Γ}.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end""")
        else:
            Main.eval(f"""
            @everywhere function output_func(sol,i)
                return trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end""")

def setup_discrete_callback_terminate(odepars: odeParameters, 
                                        stop_expression: str):
    # parse expression string to sympy equation
    expression = smp.parsing.sympy_parser.parse_expr(stop_expression)
    # extract symbols in expression and convert to a list of strings 
    symbols_in_expression = list(expression.free_symbols)
    symbols_in_expression = [str(sym) for sym in symbols_in_expression] 
    # check if all symbols are parameters of the ODE
    odepars.check_symbols_in_parameters(symbols_in_expression)

    # remove t
    symbols_in_expression.remove('t')
    # get indices of symbols
    indices = [odepars.get_index_parameter(sym, mode = "julia") for sym in symbols_in_expression]
    for idx, sym in zip(indices, symbols_in_expression):
        stop_expression = stop_expression.replace(str(sym), f"integrator.p[{idx}]")
    Main.eval(f"""
        @everywhere condition(u,t,integrator) = {stop_expression}
        @everywhere affect!(integrator) = terminate!(integrator)
        cb = DiscreteCallback(condition, affect!)
    """)

def setup_problem(odepars: odeParameters, tspan: list, ρ: np.ndarray, 
                    problem_name = "prob"):
    odepars.generate_p_julia()
    Main.ρ = ρ
    Main.tspan = tspan
    assert Main.eval("@isdefined Lindblad_rhs!"), "Lindblad function is not defined in Julia"
    Main.eval(f"""
        {problem_name} = ODEProblem(Lindblad_rhs!,ρ,tspan,p)
    """)

def setup_problem_parameter_scan(odepars: odeParameters, tspan: list, 
                                ρ: np.ndarray, parameters: list, 
                                values: np.ndarray, dimensions: int = 1,
                                problem_name = "prob", 
                                output_func = None):
    setup_problem(odepars, tspan, ρ, problem_name)
    if dimensions == 1:
        setup_parameter_scan_1D(odepars, parameters, values)
    else:
        setup_parameter_scan_ND(odepars, parameters, values)
    if output_func is not None:
        Main.eval(f"""
            ens_{problem_name} = EnsembleProblem({problem_name}, 
                                                    prob_func = prob_func,
                                                    output_func = {output_func}
                                                )
        """)
    else:
        Main.eval(f"""
            ens_{problem_name} = EnsembleProblem({problem_name}, 
                                                    prob_func = prob_func)
        """)

def solve_problem(method = "Tsit5()", abstol = 1e-7, reltol = 1e-4, 
                callback = None, problem_name = "prob", progress = False):
    if callback is not None:
        Main.eval(f"""
            sol = solve({problem_name}, {method}, abstol = {abstol}, 
                        reltol = {reltol}, progress = {str(progress).lower()}, 
                        callback = {callback}    
                    )
        """)
    else:
        Main.eval(f"""
            sol = solve({problem_name}, {method}, abstol = {abstol}, reltol = {reltol},
                        progress = {str(progress).lower()}    
                    )
        """)

def solve_problem_parameter_scan(
        method = "Tsit5()", 
        distributed_method = "EnsembleDistributed()",
        abstol = 1e-7, reltol = 1e-4, save_everystep = True,
        callback = cb, problem_name = "ens_prob",
        trajectories = None
    ):
    if trajectories is None:
        trajectories = "size(params)[1]"
    if callback is not None:
        Main.eval(f"""
            sol = solve({problem_name}, {method}, {distributed_method}, 
                        abstol = {abstol}, reltol = {reltol}, 
                        trajectories = {trajectories}, callback = {callback},
                        save_everystep = {str(save_everystep).lower()}
                    )
        """)
    else:
        Main.eval(f"""
            sol = solve({problem_name}, {method}, {distributed_method}, 
                        abstol = {abstol}, reltol = {reltol}, 
                        trajectories = {trajectories},
                        save_everystep = {str(save_everystep).lower()}
                    )
        """)  

def get_results_parameter_scan(scan_values = None):
    results = np.array(Main.eval("sol.u"))
    if scan_values is not None:
        results = results.reshape([len(val) for val in scan_values])
        X,Y = np.meshgrid(*scan_values)
        return X,Y,results.T
    return results

def get_results():
    """Retrieve the results of a single trajectory OBE simulation solution.

    Returns:
        tuple: tuple containing the timestamps and an n x m numpy arra, where
                n is the number of states, and m the number of timesteps
    """
    results = np.real(np.einsum('jji->ji', np.array(Main.eval("sol[:]")).T))
    t = Main.eval("sol.t")
    return t, results

def do_simulation_single(odepars, tspan, ρ, terminate_expression = None):
    """Perform a single trajectory solve of the OBE equations for a specified 
    TlF system.

    Args:
        odepars (odeParameters): object containing the ODE parameters used in
        the solver
        tspan (list, tuple): time range to solve for
        terminate_expression (str, optional): Expression that determines when to 
                                            stop integration. Defaults to None.

    Returns:
        tuple: tuple containing the timestamps and an n x m numpy arra, where
                n is the number of states, and m the number of timesteps
    """
    callback_flag = False
    if terminate_expression is not None:
        setup_discrete_callback_terminate(odepars, terminate_expression)
        callback_flag = True
    setup_problem(odepars, tspan, ρ)
    if callback_flag:
        solve_problem(callback = "cb")
    else:
        solve_problem()
    return get_results()

def get_indices_diag_flattened(n):
    return np.diag(np.arange(0,n*n).reshape(-1,n))
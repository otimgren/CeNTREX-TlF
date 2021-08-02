from julia import Main

__all__ = [
    'initialize_julia', 'generate_ode_fun_julia', 'setup_variables_julia',
    'odeParameters', 'setup_parameter_scan_1D', 'setup_ratio_calculation',
    'setup_initial_condition_scan', 'setup_state_integral_calculation'
]

def initialize_julia(nprocs):
    Main.eval("""
        using Logging: global_logger
        using TerminalLoggers: TerminalLogger
        global_logger(TerminalLogger())

        using Distributed
        using BenchmarkTools
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
    def __init__(self, parameters):
        self.parameters = parameters

        self._generate_attrs()
        self.p = []
    
    def _generate_attrs(self):
        for par in self.parameters:
            setattr(self, par, 0)
            
    def generate_p(self):
        return [getattr(self, par) for par in self.parameters]
    
    def generate_p_julia(self):
        Main.eval(f"p = {self.generate_p()}")
    
    def get_index_parameter(self, par):
        return self.parameters.index(par)
    
    def __repr__(self):
        rep = "odeParameters("
        for par in self.parameters:
            rep += f"{par}: {getattr(self, par):.2e}, "
        rep = rep.strip(", ")
        rep += ")"
        return rep

def setup_parameter_scan_1D(odePar, parameter, values):
    idx = odePar.get_index_parameter(parameter)
    pars = str(odePar.generate_p()).strip('[]').split(',')
    pars[idx] = "params[i]"
    pars = "[" + ",".join(pars) + "]"
    Main.params = values
    Main.eval(f"""
    @everywhere params = $params
    @everywhere function prob_func(prob,i,repeat)
        remake(prob, p = {pars})
    end
    """)

def setup_ratio_calculation(states):
    Main.eval(f"""
    @everywhere function output_func(sol,i)
        if size(sol.u)[1] != 2
            return NaN, false
        else
            val = [real(sol.u[2][1,1])/real(sol.u[1][1,1]),
                sum(real(diag(sol.u[2])[{states}]))/sum(real(diag(sol.u[1])[{states}]))]
            return val, false
        end
    end""")

def setup_initial_condition_scan(values):
    Main.params = values
    Main.eval("@everywhere params = $params")
    Main.eval("""
    @everywhere function prob_func(prob,i,repeat)
        remake(prob,u0=params[i])
    end
    """)

def setup_state_integral_calculation(states):
    Main.eval(f"""
    @everywhere function output_func(sol,i)
        return trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
    end""")
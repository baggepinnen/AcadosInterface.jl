module AcadosInterface
using Symbolics, PyCall
using LinearAlgebra, Random

@enum AcadosStatus SUCCESS NAN_DETECTED MAXITER MINSTEP QP_FAILURE READY UNBOUNDED

export AcadosStatus

const casadi = PyCall.PyNULL()
const np = PyCall.PyNULL()

function __init__()
    copy!(casadi, pyimport("casadi"))
    copy!(np, pyimport("numpy"))
end

# Add pirate overloads to make CasADi tracing work
for ff in [sin, cos, exp, sqrt]
    f = nameof(ff)
    m = Base.parentmodule(ff)
    eval(quote
        function ($m.$f)(p::$PyObject)
            return casadi.$f(p)
        end
    end)
end

function dynamics2casadi(dx::AbstractVector{Num}, x::AbstractVector{Num}, u::AbstractVector{Num}, p)
    vars = [x; u]

    xd_cas = [casadi.SX.sym(string(v)*"_dot") for v in x]
    x_cas = [casadi.SX.sym(string(v)) for v in x]
    u_cas = [casadi.SX.sym(string(v)) for v in u]

    vars_cas = [x_cas; u_cas]
    subs = Dict(vars .=> vars_cas)
    dx_cas = [Symbolics.substitute(dxi, subs) for dxi in dx]

    XD_cas = casadi.vertcat(xd_cas...)
    X_cas = casadi.vertcat(x_cas...)
    U_cas = casadi.vertcat(u_cas...)
    DX_cas = casadi.vertcat(dx_cas...)

    XD_cas, X_cas, U_cas, DX_cas
end


function symbolics_trace_dynamics(dynamics, nx::Int, nu::Int, p)
    x = Symbolics.variables(:X, 1:nx)
    u = Symbolics.variables(:U, 1:nu)
    vars = [x; u]
    if hasmethod(dynamics, (typeof(x), typeof(u), typeof(p), Real))
        dx = dynamics(x, u, p, 0.0)
    elseif hasmethod(dynamics, (typeof(x), typeof(x), typeof(u), typeof(p), Real))
        dx = similar(x)
        dynamics(dx, x, u, p, 0.0)
    else
        throw(ArgumentError("dynamics must be a function of the form dynamics(x, u, p, t) or dynamics(dx, x, u, p, t)"))
    end

    dx, x, u
end

struct OCP
    model
    ocp
    ocp_solver
    nx::Int
    nu::Int
    N::Int
end


"""
    prob = generate(dynamics;
        nx,
        nu,
        N::Int,
        Tf,
        p = nothing,
        modelname = "model_$(randstring('a':'z', 6))", # Without the random string, changes to settings will have no effect unless julia is restarted
        x_labels = nothing,
        u_labels = nothing,
        Q1 = nothing,
        Q2 = nothing,
        QN = nothing,
        xr = zeros(nx),
        ur = zeros(nu),
        yr = [xr; ur],
        xrN = xr,
        umin = nothing,
        umax = nothing,
        xmin = nothing,
        xmax = nothing,
        xNmin = nothing,
        xNmax = nothing,
        x0 = nothing,
        qp_solver = "PARTIAL_CONDENSING_HPIPM",
        hessian_approx = "GAUSS_NEWTON",
        integrator_type = "ERK",
        print_level = 0,
        nlp_solver_type = "SQP",
        globalization = "MERIT_BACKTRACKING",
        nlp_solver_max_iter = 100,
        generate_ocp = true,
        generate_solver = true,
        verbose = false,
        cost_type = "NONLINEAR_LS",
        cost_type_e = "NONLINEAR_LS",
        Vx = nothing,
        Vu = nothing,
        Vx_e = nothing,
        Vu_e = nothing,
        yref_e = yr,
        model = nothing,
        ocp = nothing,
    )

DOCSTRING

# Arguments:
- `dynamics`: A continuous-time dynamics function on the form `dynamics(x, u, p, t)` or `dynamics(dx, x, u, p, t)` where `x` is the state, `u` is the control, `p` is the parameters, and `t` is the time. The function should return the time derivative of the state.
- `nx`: Dimension of the state
- `nu`: Dimension of the control input
- `N`: Optimization horizon (number of time steps)
- `Tf`: Final time (duration). The time step is thus equal to `Tf/N`
- `p`: Parameters that are passed to the dynamics function
- `modelname`: A descriptive name for the generated code
- `x_labels`: Labels for state variables
- `u_labels`: Labels for control variables
- `Q1`: Quadratic cost matrix for the state along the trajectory
- `Q2`: Quadratic cost matrix for the control along the trajectory
- `QN`: Quadratic cost matrix for the terminal state
- `xr`: Reference state along the trajectory
- `ur`: Reference control along the trajectory
- `yr`: Reference output along the trajectory, used with `cost_type = "LINEAR_LS"` and `Vx, Vu`.
- `xrN`: Reference terminal state
- `umin`: Lower bound on the control input
- `umax`: Upper bound on the control input
- `xmin`: Lower bound on the state
- `xmax`: Upper bound on the state
- `xNmin`: Lower bound on the terminal state
- `xNmax`: Upper bound on the terminal state
- `x0`: Initial state
- `qp_solver`: The QP solver to use, options are "PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_QPOASES", "FULL_CONDENSING_HPIPM", "PARTIAL_CONDENSING_QPDUNES", "PARTIAL_CONDENSING_OSQP", "FULL_CONDENSING_DAQP"
- `hessian_approx`: Hessian-approximation method, options are "GAUSS_NEWTON", "EXACT"
- `integrator_type`: Integrator type, options are "IRK", "ERK"
- `print_level`: How much to print during the optimization. 0-4
- `nlp_solver_type`: Nonlinear programming solver type, options are "SQP_RTI", "SQP"
- `globalization`: Globalization method, options are "FIXED_STEP", "MERIT_BACKTRACKING", "FUNNEL_L1PEN_LINESEARCH"
- `nlp_solver_max_iter`: Maximum number of iterations for the nonlinear programming solver
- `generate_ocp`: Turn off to only generate the model
- `generate_solver`: Turn off to only generate the model and the OCP
- `verbose`: Print status while generating the model and OCP
- `cost_type`: Cost type for the path cost, options are "LINEAR_LS", "NONLINEAR_LS"
- `cost_type_e`: Cost type for the terminal cost, options are "LINEAR_LS", "NONLINEAR_LS"
- `Vx`: Linear output matrix for the path cost, i.e., `y = Vx*x + Vu*u`
- `Vu`: Linear output matrix for the path cost, i.e., `y = Vx*x + Vu*u`
- `model`: A pre-existing model to use
- `ocp`: A pre-existing OCP to use
"""
function generate(dynamics::AbstractVector{Num};
    nx,
    nu,
    x = nothing,
    u = nothing,
    N::Int,
    Tf,
    p = nothing,
    modelname = "model_$(randstring('a':'z', 6))", # Without the random string, changes to settings will have no effect unless julia is restarted
    x_labels = nothing,
    u_labels = nothing,
    Q1 = nothing,
    Q2 = nothing,
    QN = nothing,
    xr = zeros(nx),
    ur = zeros(nu),
    yr = [xr; ur],
    xrN = xr,
    umin = nothing,
    umax = nothing,
    xmin = nothing,
    xmax = nothing,
    xNmin = nothing,
    xNmax = nothing,
    x0 = nothing,
    qp_solver = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx = "GAUSS_NEWTON",
    integrator_type = "ERK",
    print_level = 0,
    nlp_solver_type = "SQP",
    globalization = "MERIT_BACKTRACKING",
    nlp_solver_max_iter = 100,
    generate_ocp = true,
    generate_solver = true,
    verbose = false,
    cost_type = "NONLINEAR_LS",
    cost_type_e = "NONLINEAR_LS",
    Vx = nothing,
    Vu = nothing,
    Vx_e = nothing,
    Vu_e = nothing,
    yref_e = yr,
    model = nothing,
    ocp = nothing,
)

    length(dynamics) == nx || throw(ArgumentError("vector of symbolic expressions for dynamics must have length $nx"))
    length(x) == nx || throw(ArgumentError("x must have length $nx"))
    length(u) == nu || throw(ArgumentError("u must have length $nu"))

    XD_cas, X_cas, U_cas, DX_cas = dynamics2casadi(dynamics, x, u, p)
    
    verbose && @info "Creating model"
    AcadosModel = pyimport("acados_template").AcadosModel
    if model === nothing
        model = AcadosModel()
        model.f_expl_expr = DX_cas
        model.f_impl_expr = XD_cas - DX_cas
        model.xdot = XD_cas
        model.x = X_cas
        model.u = U_cas
        model.name = modelname 
        x_labels === nothing || (model.x_labels = x_labels)
        u_labels === nothing || (model.u_labels = u_labels)
    end


    generate_ocp || return model

    verbose && @info "Creating OCP"
    if ocp === nothing
        AcadosOcp = pyimport("acados_template").AcadosOcp
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = Tf


        # path cost
        if Q1 !== nothing
            # NOTE: this is actually using the nonlinear interface even though we have a linear output function y = [x; u]
            verbose && @info "Adding path cost"
            isnothing(Q2) && throw(ArgumentError("Q2 must be provided if Q1 is provided"))
            ocp.cost.cost_type = cost_type
            ocp.model.cost_y_expr = casadi.vertcat(model.x, model.u) # This is the potentially nonlinear output function, here we use a linear function for now. TODO: trace a nonlinear user-provided output function
            ocp.cost.yref = np.array(yr)
            ocp.cost.W = casadi.diagcat(np.matrix(Q1), np.matrix(Q2)).full()
        end
        if Vx !== nothing
            verbose && @info "Adding linear path cost"
            isnothing(Vu) && throw(ArgumentError("Vu must be provided if Vx is provided"))
            isnothing(yr) && throw(ArgumentError("yr must be provided if Vx is provided"))
            cost_type == "LINEAR_LS" || throw(ArgumentError("cost_type must be 'LINEAR_LS' if Vx and Vu are provided"))
            ocp.cost.cost_type = cost_type
            ocp.cost.yref = np.array(yr)
            ocp.cost.Vx = np.array(Vx) # Cz in z = Cz*x + Dz*u
            ocp.cost.Vu = np.array(Vu) # Dz
        end

        # terminal cost
        if QN !== nothing
            verbose && @info "Adding terminal cost"
            ocp.cost.cost_type_e = cost_type_e
            ocp.cost.yref_e = np.array(xrN)
            ocp.model.cost_y_expr_e = model.x
            ocp.cost.W_e = np.matrix(QN)
        end

        if Vx_e !== nothing
            verbose && @info "Adding linear terminal cost"
            isnothing(Vu_e) && throw(ArgumentError("Vu_e must be provided if Vx_e is provided"))
            isnothing(yref_e) && throw(ArgumentError("yref_e must be provided if Vx_e is provided"))
            cost_type_e == "LINEAR_LS" || throw(ArgumentError("cost_type_e must be 'LINEAR_LS' if Vx_e and Vu_e are provided"))
            ocp.cost.cost_type_e = cost_type_e
            ocp.cost.yref_e = np.array(yref_e)
            ocp.cost.Vx_e = np.array(Vx_e) # Cz in z = Cz*x + Dz*u
            ocp.cost.Vu_e = np.array(Vu_e) # Dz
        end

        # set constraints
        if umin !== nothing
            verbose && @info "Adding control constraints"
            isnothing(umax) && throw(ArgumentError("umax must be provided if umin is provided"))
            ocp.constraints.lbu = np.array(umin)
            ocp.constraints.ubu = np.array(umax)
            ocp.constraints.idxbu = findall(isfinite.(umax) .| isfinite.(umin)) .- 1 # u indices that have bounds declared
        end

        if xmin !== nothing
            verbose && @info "Adding state constraints"
            isnothing(xmax) && throw(ArgumentError("xmax must be provided if xmin is provided"))
            ocp.constraints.lbx = np.array(xmin)
            ocp.constraints.ubx = np.array(xmax)
        end
        
        if x0 !== nothing
            verbose && @info "Adding initial state constraints"
            ocp.constraints.x0 = np.array(x0)
        end

        # terminal constraints
        if xNmin !== nothing
            verbose && @info "Adding terminal state constraints"
            isnothing(xNmax) && throw(ArgumentError("xNmax must be provided if xNmin is provided"))
            ocp.constraints.lbx_e = np.array(xNmin)
            ocp.constraints.ubx_e = np.array(xNmax)

            ocp.constraints.idxbx_e = findall(isfinite.(xNmax) .| isfinite.(xNmin)) .- 1 # x indices that have terminal bounds declared
        end


        # set options
        verbose && @info "Setting solver options"
        ocp.solver_options.qp_solver = qp_solver # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = hessian_approx # "GAUSS_NEWTON", "EXACT"
        ocp.solver_options.integrator_type = integrator_type # IRK, ERK
        ocp.solver_options.print_level = print_level
        ocp.solver_options.nlp_solver_type = nlp_solver_type # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
        ocp.solver_options.globalization = globalization # turns on globalization
    end

    # rm("acados_ocp.json", force=true)
    # rm("c_generated_code", recursive=true, force=true)

    generate_solver || return (; model, ocp)

    verbose && @info "Generating solver"
    AcadosOcpSolver = pyimport("acados_template").AcadosOcpSolver
    ocp_solver = AcadosOcpSolver(ocp)

    OCP(model, ocp, ocp_solver, nx, nu, N)
end

function generate(dynamics; nx, nu, p=nothing, kwargs...)
    dx, x, u = symbolics_trace_dynamics(dynamics, nx, nu, p)
    generate(dx; x, u, nx, nu, p, kwargs...)
end

"""
    X,U = solve_and_extract(prob::OCP, verbose=true)

Solve the optimal control problem through the python interface and extract the state and control trajectories.
"""
function solve_and_extract(prob::OCP, verbose=true)
    ocp_solver = prob.ocp_solver
    @time status = ocp_solver.solve() |> AcadosStatus
    verbose && ocp_solver.print_statistics()
    status == SUCCESS || @warn "Solver failed: status code $status"
    (; nx, nu, N) = prob
    X = zeros(nx, N+1)
    U = zeros(nu, N)
    for i = 1:N
        X[:, i] = ocp_solver.get(i-1, "x")
        U[:, i] = ocp_solver.get(i-1, "u")
    end
    X[:, N+1] = ocp_solver.get(N, "x")
    X, U
end

end

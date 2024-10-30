ENV["PYTHON"] = "/home/fredrikb/repos/acados/bin/python3" # Path to venv python 
ENV["PYCALL_JL_RUNTIME_PYTHON"] = "/home/fredrikb/repos/acados/bin/python3"

using AcadosInterface
using Test, LinearAlgebra


@testset "AcadosInterface.jl" begin

    function cartpole(x, u, p, _=0)
        mc, mp, l, g = 1.0, 0.2, 0.5, 9.81
    
        q  = x[[1, 2]]
        qd = x[[3, 4]]
    
        s = sin(q[2])
        c = cos(q[2])
    
        H = [mc+mp mp*l*c; mp*l*c mp*l^2]
        C = [0 -mp*qd[2]*l*s; 0 0]
        G = [0, mp * g * l * s]
        B = [1, 0]
    
        qdd = -H \ (C * qd + G - B * u[1])
        return [qd; qdd]
    end
    
    nu = 1           # number of controls
    nx = 4           # number of states
    Ts = 0.04        # sample time
    N = 80           # Optimization horizon (number of time steps)
    Tf = N * Ts      # End time
    x0 = zeros(nx)   # Initial state
    xr = [0, π, 0, 0] # Reference state is given by the pendulum upright (π rad)

    umax = [10]
    umin = -umax

    Q1 = Diagonal([1, 1, 1, 1])
    Q2 = Diagonal([0.01])
    QN = Q1



    prob = AcadosInterface.generate(cartpole;
        nx,
        nu,
        N,
        Tf,
        x0,
        xr,
        umin,
        umax,
        Q1,
        Q2,
        QN,
        verbose = true,
        xNmin = xr,
        xNmax = xr,
        nlp_solver_max_iter = 1000,
    )
    X, U = AcadosInterface.solve_and_extract(prob)

    @test X[:, end] ≈ xr atol=1e-3
    
    if isinteractive()
        using Plots
        plot(
            plot(X', layout=1),
            plot(U', layout=nu),
        )
    end
    rm("acados_ocp.json", force=true)
    rm("c_generated_code", recursive=true, force=true)

end

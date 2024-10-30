# AcadosInterface

[![Build Status](https://github.com/baggepinnen/AcadosInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/AcadosInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!CAUTION]
> This package is under development and is not yet expected to work very well and the API is likely to break.


This package provides an interface to the [acados](https://docs.acados.org/index.html) suite of tools for optimal control and model-predictive control. We target the acados [python interface](https://docs.acados.org/python_interface/index.html) by going through [PyCall.jl](https://github.com/JuliaPy/PyCall.jl). Dynamics implemented in Julia are put through the following pipeline
1. [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) is used to build a symbolic representation of the Julia dynamcis.
2. `symbolics.substitute` is used to substitute the Julia symbolic variables for [CasADi](https://web.casadi.org/) symbolic variables.
3. The acados python interface is used to generate the C code for the dynamics.

The main entrypoint to the translation is `AcadosInterface.generate(dynamics; kwargs...)`, where `dynamics` is a julia function on either of the forms
```julia
ẋ = dynamics(x, u, p, t)
dynamics!(ẋ, x, u, p, t)
```

## Installation
1. Follow the [installation instructions for the acados python interface](https://docs.acados.org/python_interface/index.html). Try running one of their examples to make sure everything works.
2. Make sure that PyCall.jl can find the relevant python installation. If you follow the acados advice of creating a virtual environment, this can be done by something like `julia> ENV["PYTHON"] = "<path_to_acados>/acados/bin/python3" # Path to venv python` _before_ loading PyCall.jl or AcadosInterface.jl. If PyCall has already been built, you must rebuild PyCall after having pointed to the correct python installation. See [PyCall: python-virtual-environments](https://github.com/JuliaPy/PyCall.jl?tab=readme-ov-file#python-virtual-environments) if you get lost.
3. Install this package, it's not registered so you may install it from the URL `using Pkg; Pkg.add(url="https://github.com/baggepinnen/AcadosInterface.jl")`


## Example: Cartpole swing-up
This example follows [JuliaSimControl: Pendulum swing-up](https://help.juliahub.com/juliasimcontrol/dev/examples/optimal_control/#Pendulum-swing-up)
```julia
using AcadosInterface
using Test, LinearAlgebra

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

nu = 1              # number of control inputs
nx = 4              # dimension of state
Ts = 0.04           # sample time
N = 80              # Optimization horizon (number of time steps)
Tf = N * Ts         # End time
x0 = zeros(nx)      # Initial state
xr = [0, π, 0, 0]   # Reference state is given by the pendulum upright (π rad)

umax = [10]         # Control input limits
umin = -umax

Q1 = Diagonal([1, 1, 1, 1]) # Quadratic cost on state
Q2 = Diagonal([0.01])       # Quadratic cost on control
QN = Q1                     # Quadratic cost on terminal state

x_labels = ["x", "θ", "dx", "dθ"]


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
    xNmin = xr,
    xNmax = xr,
    x_labels,
    verbose = true,
    nlp_solver_max_iter = 1000,
)

X, U = AcadosInterface.simulate(prob)

@test X[:, end] ≈ xr atol=1e-3 # Test that the pendulum swing-up worked

using Plots
plot(
    plot(X', label=permutedims(x_labels), title="State trajectory"),
    plot(U', label="u", title="Control trajectory"),
)
```
![result](https://private-user-images.githubusercontent.com/3797491/381536977-1e9437bb-acde-4e43-b7f4-0f763ffedc01.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAyOTAyMTAsIm5iZiI6MTczMDI4OTkxMCwicGF0aCI6Ii8zNzk3NDkxLzM4MTUzNjk3Ny0xZTk0MzdiYi1hY2RlLTRlNDMtYjdmNC0wZjc2M2ZmZWRjMDEucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MTAzMCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDEwMzBUMTIwNTEwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YzZiODFhMjk3YmU3ODUzYjU4MDM3OWExMGRhOTNhYjBlNzkyYjdlMTNiNWUyZWNiNDliODFiMDUyZTAzNmI2YyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.m5NPTTQYyQY8Vr5V1tiCJoZeKahJyZA5gFfnojoF0F8)
```julia
rm("acados_ocp.json", force=true)
rm("c_generated_code", recursive=true, force=true)
```
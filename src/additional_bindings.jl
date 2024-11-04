using AcadosBindings

# Step 1: Create the ocp_nlp_plan_t object
function ocp_nlp_plan_create(N::Int)
    plan = ccall((:ocp_nlp_plan_create, libacados), Ptr{ocp_nlp_plan_t}, (Cint,), N)
    return plan
end

# Step 2: Create the ocp_nlp_config object
function ocp_nlp_config_create(plan::Ptr{ocp_nlp_plan_t})
    config = ccall((:ocp_nlp_config_create, libacados), Ptr{Cint}, (Ptr{ocp_nlp_plan_t},), plan)
    return config
end

# Step 3: Create the ocp_nlp_dims object
function ocp_nlp_dims_create(config::Ptr{Cint})
    dims = ccall((:ocp_nlp_dims_create, libacados), Ptr{Cint}, (Ptr{Cint},), config)
    return dims
end

# Step 4: Create the opts object
function ocp_nlp_solver_opts_create(config::Ptr{Cint}, dims::Ptr{Cint})
    opts = ccall((:ocp_nlp_solver_opts_create, libacados), Ptr{Cvoid}, (Ptr{Cint}, Ptr{Cint}), config, dims)
    return opts
end

# Step 5: Create the mem object
function ocp_nlp_solver_create(config::Ptr{Cint}, dims::Ptr{Cint})
    mem = ccall((:ocp_nlp_solver_create, libacados), Ptr{Cvoid}, (Ptr{Cint}, Ptr{Cint}), config, dims)
    return mem
end

# Step 6: Create the work object
function ocp_nlp_solver_workspace_create(config::Ptr{Cint}, dims::Ptr{Cint})
    work = ccall((:ocp_nlp_solver_workspace_create, libacados), Ptr{Cvoid}, (Ptr{Cint}, Ptr{Cint}), config, dims)
    return work
end

# Example usage
N = 20  # Horizon length
plan = ocp_nlp_plan_create(N)
config = ocp_nlp_config_create(plan)
dims = ocp_nlp_dims_create(config)
opts = ocp_nlp_solver_opts_create(config, dims)
mem = ocp_nlp_solver_create(config, dims)
work = ocp_nlp_solver_workspace_create(config, dims)

# Now you can use the config and dims objects to create the ocp_nlp_solver struct
solver = ocp_nlp_solver(config, dims, opts, mem, work)


have a look at /home/fredrikb/repos/acados/examples/c_ext_dep_off/pendulum/acados_solver_pendulum_ode.h
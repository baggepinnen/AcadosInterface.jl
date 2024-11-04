module AcadosBindings

using CEnum

@cenum condensing_t::UInt32 begin
    PARTIAL_CONDENSING = 0
    FULL_CONDENSING = 1
end

struct condensing_plan
    condensing_type::condensing_t
end

struct condensing_module
    config::Ptr{Cint}
    dims::Ptr{Cvoid}
    opts::Ptr{Cvoid}
    mem::Ptr{Cvoid}
    work::Ptr{Cvoid}
end

function ocp_qp_condensing_config_create(plan)
    ccall((:ocp_qp_condensing_config_create, libacados), Ptr{Cint}, (Ptr{condensing_plan},), plan)
end

function ocp_qp_condensing_opts_create(config, dims_)
    ccall((:ocp_qp_condensing_opts_create, libacados), Ptr{Cvoid}, (Ptr{Cint}, Ptr{Cvoid}), config, dims_)
end

function ocp_qp_condensing_calculate_size(config, dims_, opts_)
    ccall((:ocp_qp_condensing_calculate_size, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}), config, dims_, opts_)
end

function ocp_qp_condensing_assign(config, dims_, opts_, raw_memory)
    ccall((:ocp_qp_condensing_assign, libacados), Ptr{condensing_module}, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), config, dims_, opts_, raw_memory)
end

function ocp_qp_condensing_create(config, dims_, opts_)
    ccall((:ocp_qp_condensing_create, libacados), Ptr{condensing_module}, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}), config, dims_, opts_)
end

function ocp_qp_condense(_module, qp_in, qp_out)
    ccall((:ocp_qp_condense, libacados), Cint, (Ptr{condensing_module}, Ptr{Cvoid}, Ptr{Cvoid}), _module, qp_in, qp_out)
end

function ocp_qp_expand(_module, qp_in, qp_out)
    ccall((:ocp_qp_expand, libacados), Cint, (Ptr{condensing_module}, Ptr{Cvoid}, Ptr{Cvoid}), _module, qp_in, qp_out)
end

@cenum dense_qp_solver_t::UInt32 begin
    DENSE_QP_HPIPM = 0
    DENSE_QP_QORE = 1
    DENSE_QP_QPOASES = 2
    DENSE_QP_OOQP = 3
    DENSE_QP_DAQP = 4
end

struct dense_qp_solver_plan
    qp_solver::dense_qp_solver_t
end

struct dense_qp_solver
    config::Ptr{Cint}
    dims::Ptr{Cvoid}
    opts::Ptr{Cvoid}
    mem::Ptr{Cvoid}
    work::Ptr{Cvoid}
end

function dense_qp_config_create(plan)
    ccall((:dense_qp_config_create, libacados), Ptr{Cint}, (Ptr{dense_qp_solver_plan},), plan)
end

# no prototype is found for this function at dense_qp_interface.h:59:16, please use with caution
function dense_qp_dims_create()
    ccall((:dense_qp_dims_create, libacados), Ptr{Cint}, ())
end

function dense_qp_in_create(config, dims)
    ccall((:dense_qp_in_create, libacados), Ptr{Cint}, (Ptr{Cint}, Ptr{Cint}), config, dims)
end

function dense_qp_out_create(config, dims)
    ccall((:dense_qp_out_create, libacados), Ptr{Cint}, (Ptr{Cint}, Ptr{Cint}), config, dims)
end

function dense_qp_opts_create(config, dims)
    ccall((:dense_qp_opts_create, libacados), Ptr{Cvoid}, (Ptr{Cint}, Ptr{Cint}), config, dims)
end

function dense_qp_calculate_size(config, dims, opts_)
    ccall((:dense_qp_calculate_size, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}), config, dims, opts_)
end

function dense_qp_assign(config, dims, opts_, raw_memory)
    ccall((:dense_qp_assign, libacados), Ptr{dense_qp_solver}, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}), config, dims, opts_, raw_memory)
end

function dense_qp_create(config, dims, opts_)
    ccall((:dense_qp_create, libacados), Ptr{dense_qp_solver}, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}), config, dims, opts_)
end

function dense_qp_solve(solver, qp_in, qp_out)
    ccall((:dense_qp_solve, libacados), Cint, (Ptr{dense_qp_solver}, Ptr{Cint}, Ptr{Cint}), solver, qp_in, qp_out)
end

function dense_qp_inf_norm_residuals(dims, qp_in, qp_out, res)
    ccall((:dense_qp_inf_norm_residuals, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}), dims, qp_in, qp_out, res)
end

function dense_qp_set_field_double_array(field, arr, qp_in)
    ccall((:dense_qp_set_field_double_array, libacados), Cint, (Ptr{Cchar}, Ptr{Cdouble}, Ptr{Cint}), field, arr, qp_in)
end

function dense_qp_set_field_int_array(field, arr, qp_in)
    ccall((:dense_qp_set_field_int_array, libacados), Cint, (Ptr{Cchar}, Ptr{Cint}, Ptr{Cint}), field, arr, qp_in)
end

function dense_qp_get_field_double_array(field, qp_in, arr)
    ccall((:dense_qp_get_field_double_array, libacados), Cint, (Ptr{Cchar}, Ptr{Cint}, Ptr{Cdouble}), field, qp_in, arr)
end

function dense_qp_get_field_int_array(field, qp_in, arr)
    ccall((:dense_qp_get_field_int_array, libacados), Cint, (Ptr{Cchar}, Ptr{Cint}, Ptr{Cint}), field, qp_in, arr)
end

function external_function_param_generic_create(fun, np, opts_)
    ccall((:external_function_param_generic_create, libacados), Cvoid, (Ptr{Cint}, Cint, Ptr{Cint}), fun, np, opts_)
end

function external_function_param_generic_free(fun)
    ccall((:external_function_param_generic_free, libacados), Cvoid, (Ptr{Cint},), fun)
end

function external_function_casadi_create(fun, opts_)
    ccall((:external_function_casadi_create, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}), fun, opts_)
end

function external_function_casadi_free(fun)
    ccall((:external_function_casadi_free, libacados), Cvoid, (Ptr{Cint},), fun)
end

function external_function_casadi_create_array(size, funs, opts_)
    ccall((:external_function_casadi_create_array, libacados), Cvoid, (Cint, Ptr{Cint}, Ptr{Cint}), size, funs, opts_)
end

function external_function_casadi_free_array(size, funs)
    ccall((:external_function_casadi_free_array, libacados), Cvoid, (Cint, Ptr{Cint}), size, funs)
end

function external_function_param_casadi_create(fun, np, opts_)
    ccall((:external_function_param_casadi_create, libacados), Cvoid, (Ptr{Cint}, Cint, Ptr{Cint}), fun, np, opts_)
end

function external_function_param_casadi_free(fun)
    ccall((:external_function_param_casadi_free, libacados), Cvoid, (Ptr{Cint},), fun)
end

function external_function_param_casadi_create_array(size, funs, np, opts_)
    ccall((:external_function_param_casadi_create_array, libacados), Cvoid, (Cint, Ptr{Cint}, Cint, Ptr{Cint}), size, funs, np, opts_)
end

function external_function_param_casadi_free_array(size, funs)
    ccall((:external_function_param_casadi_free_array, libacados), Cvoid, (Cint, Ptr{Cint}), size, funs)
end

function external_function_external_param_casadi_create(fun, opts_)
    ccall((:external_function_external_param_casadi_create, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}), fun, opts_)
end

function external_function_external_param_casadi_free(fun)
    ccall((:external_function_external_param_casadi_free, libacados), Cvoid, (Ptr{Cint},), fun)
end

function external_function_external_param_generic_create(fun, opts_)
    ccall((:external_function_external_param_generic_create, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}), fun, opts_)
end

function external_function_external_param_generic_free(fun)
    ccall((:external_function_external_param_generic_free, libacados), Cvoid, (Ptr{Cint},), fun)
end

@cenum ocp_nlp_solver_t::UInt32 begin
    SQP = 0
    SQP_RTI = 1
    DDP = 2
    INVALID_NLP_SOLVER = 3
end

@cenum ocp_nlp_dynamics_t::UInt32 begin
    CONTINUOUS_MODEL = 0
    DISCRETE_MODEL = 1
    INVALID_DYNAMICS = 2
end

@cenum ocp_nlp_constraints_t::UInt32 begin
    BGH = 0
    BGP = 1
    INVALID_CONSTRAINT = 2
end

@cenum ocp_nlp_reg_t::UInt32 begin
    NO_REGULARIZE = 0
    MIRROR = 1
    PROJECT = 2
    PROJECT_REDUC_HESS = 3
    CONVEXIFY = 4
    INVALID_REGULARIZE = 5
end

@cenum ocp_nlp_globalization_t::UInt32 begin
    FIXED_STEP = 0
    MERIT_BACKTRACKING = 1
    FUNNEL_L1PEN_LINESEARCH = 2
end

struct ocp_nlp_plan_t
    ocp_qp_solver_plan::Cint
    sim_solver_plan::Ptr{Cint}
    nlp_solver::ocp_nlp_solver_t
    regularization::ocp_nlp_reg_t
    globalization::ocp_nlp_globalization_t
    nlp_cost::Ptr{Cint}
    nlp_dynamics::Ptr{ocp_nlp_dynamics_t}
    nlp_constraints::Ptr{ocp_nlp_constraints_t}
    N::Cint
end

struct ocp_nlp_solver
    config::Ptr{Cint}
    dims::Ptr{Cvoid}
    opts::Ptr{Cvoid}
    mem::Ptr{Cvoid}
    work::Ptr{Cvoid}
end

function ocp_nlp_plan_destroy(plan_)
    ccall((:ocp_nlp_plan_destroy, libacados), Cint, (Ptr{Cvoid},), plan_)
end

function ocp_nlp_config_destroy(config_)
    ccall((:ocp_nlp_config_destroy, libacados), Cint, (Ptr{Cvoid},), config_)
end

function ocp_nlp_dims_destroy(dims_)
    ccall((:ocp_nlp_dims_destroy, libacados), Cint, (Ptr{Cvoid},), dims_)
end

function ocp_nlp_in_destroy(in)
    ccall((:ocp_nlp_in_destroy, libacados), Cint, (Ptr{Cvoid},), in)
end

function ocp_nlp_in_set(config, dims, in, stage, field, value)
    ccall((:ocp_nlp_in_set, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, value)
end

function ocp_nlp_dynamics_model_set_external_param_fun(config, dims, in, stage, field, ext_fun)
    ccall((:ocp_nlp_dynamics_model_set_external_param_fun, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, ext_fun)
end

function ocp_nlp_cost_model_set_external_param_fun(config, dims, in, stage, field, ext_fun)
    ccall((:ocp_nlp_cost_model_set_external_param_fun, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, ext_fun)
end

function ocp_nlp_constraints_model_set_external_param_fun(config, dims, in, stage, field, ext_fun)
    ccall((:ocp_nlp_constraints_model_set_external_param_fun, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, ext_fun)
end

function ocp_nlp_in_set_params_sparse(config, dims, in, stage, idx, p, n_update)
    ccall((:ocp_nlp_in_set_params_sparse, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Ptr{Cdouble}, Cint), config, dims, in, stage, idx, p, n_update)
end

function ocp_nlp_in_get(config, dims, in, stage, field, value)
    ccall((:ocp_nlp_in_get, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, value)
end

function ocp_nlp_dynamics_model_set(config, dims, in, stage, fun_type, fun_ptr)
    ccall((:ocp_nlp_dynamics_model_set, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, fun_type, fun_ptr)
end

function ocp_nlp_cost_model_set(config, dims, in, stage, field, value)
    ccall((:ocp_nlp_cost_model_set, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, value)
end

function ocp_nlp_constraints_model_set(config, dims, in, stage, field, value)
    ccall((:ocp_nlp_constraints_model_set, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, value)
end

function ocp_nlp_constraints_model_get(config, dims, in, stage, field, value)
    ccall((:ocp_nlp_constraints_model_get, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, in, stage, field, value)
end

function ocp_nlp_out_destroy(out)
    ccall((:ocp_nlp_out_destroy, libacados), Cint, (Ptr{Cvoid},), out)
end

function ocp_nlp_out_set(config, dims, out, stage, field, value)
    ccall((:ocp_nlp_out_set, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, out, stage, field, value)
end

function ocp_nlp_out_get(config, dims, out, stage, field, value)
    ccall((:ocp_nlp_out_get, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, out, stage, field, value)
end

function ocp_nlp_get_at_stage(config, dims, solver, stage, field, value)
    ccall((:ocp_nlp_get_at_stage, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{ocp_nlp_solver}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, dims, solver, stage, field, value)
end

function ocp_nlp_get_from_iterate(dims, solver, iter, stage, field, value)
    ccall((:ocp_nlp_get_from_iterate, libacados), Cint, (Ptr{Cint}, Ptr{ocp_nlp_solver}, Cint, Cint, Ptr{Cchar}, Ptr{Cvoid}), dims, solver, iter, stage, field, value)
end

function ocp_nlp_dims_get_from_attr(config, dims, out, stage, field)
    ccall((:ocp_nlp_dims_get_from_attr, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}), config, dims, out, stage, field)
end

function ocp_nlp_constraint_dims_get_from_attr(config, dims, out, stage, field, dims_out)
    ccall((:ocp_nlp_constraint_dims_get_from_attr, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cint}), config, dims, out, stage, field, dims_out)
end

function ocp_nlp_cost_dims_get_from_attr(config, dims, out, stage, field, dims_out)
    ccall((:ocp_nlp_cost_dims_get_from_attr, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cint}), config, dims, out, stage, field, dims_out)
end

function ocp_nlp_qp_dims_get_from_attr(config, dims, out, stage, field, dims_out)
    ccall((:ocp_nlp_qp_dims_get_from_attr, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cint}), config, dims, out, stage, field, dims_out)
end

function ocp_nlp_solver_opts_create(config, dims)
    ccall((:ocp_nlp_solver_opts_create, libacados), Ptr{Cint}, (Ptr{Cint}, Ptr{Cint}), config, dims)
end

function ocp_nlp_solver_opts_destroy(opts)
    ccall((:ocp_nlp_solver_opts_destroy, libacados), Cint, (Ptr{Cvoid},), opts)
end

function ocp_nlp_solver_opts_set(config, opts_, field, value)
    ccall((:ocp_nlp_solver_opts_set, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}), config, opts_, field, value)
end

function ocp_nlp_solver_opts_set_at_stage(config, opts_, stage, field, value)
    ccall((:ocp_nlp_solver_opts_set_at_stage, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, opts_, stage, field, value)
end

function ocp_nlp_solver_opts_update(config, dims, opts_)
    ccall((:ocp_nlp_solver_opts_update, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}), config, dims, opts_)
end

function ocp_nlp_solver_destroy(solver)
    ccall((:ocp_nlp_solver_destroy, libacados), Cint, (Ptr{ocp_nlp_solver},), solver)
end

function ocp_nlp_solve(solver, nlp_in, nlp_out)
    ccall((:ocp_nlp_solve, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cint}), solver, nlp_in, nlp_out)
end

function ocp_nlp_solver_reset_qp_memory(solver, nlp_in, nlp_out)
    ccall((:ocp_nlp_solver_reset_qp_memory, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cint}), solver, nlp_in, nlp_out)
end

function ocp_nlp_precompute(solver, nlp_in, nlp_out)
    ccall((:ocp_nlp_precompute, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cint}), solver, nlp_in, nlp_out)
end

function ocp_nlp_eval_cost(solver, nlp_in, nlp_out)
    ccall((:ocp_nlp_eval_cost, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cint}), solver, nlp_in, nlp_out)
end

function ocp_nlp_eval_params_jac(solver, nlp_in, nlp_out)
    ccall((:ocp_nlp_eval_params_jac, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cint}), solver, nlp_in, nlp_out)
end

function ocp_nlp_eval_residuals(solver, nlp_in, nlp_out)
    ccall((:ocp_nlp_eval_residuals, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cint}), solver, nlp_in, nlp_out)
end

function ocp_nlp_eval_param_sens(solver, field, stage, index, sens_nlp_out)
    ccall((:ocp_nlp_eval_param_sens, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cchar}, Cint, Cint, Ptr{Cint}), solver, field, stage, index, sens_nlp_out)
end

function ocp_nlp_eval_lagrange_grad_p(solver, nlp_in, field, out)
    ccall((:ocp_nlp_eval_lagrange_grad_p, libacados), Cint, (Ptr{ocp_nlp_solver}, Ptr{Cint}, Ptr{Cchar}, Ptr{Cdouble}), solver, nlp_in, field, out)
end

function ocp_nlp_get(config, solver, field, return_value_)
    ccall((:ocp_nlp_get, libacados), Cint, (Ptr{Cint}, Ptr{ocp_nlp_solver}, Ptr{Cchar}, Ptr{Cvoid}), config, solver, field, return_value_)
end

function ocp_nlp_set(config, solver, stage, field, value)
    ccall((:ocp_nlp_set, libacados), Cint, (Ptr{Cint}, Ptr{ocp_nlp_solver}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, solver, stage, field, value)
end

@cenum ocp_qp_solver_t::UInt32 begin
    PARTIAL_CONDENSING_HPIPM = 0
    PARTIAL_CONDENSING_HPMPC_NOT_AVAILABLE = 1
    PARTIAL_CONDENSING_OOQP_NOT_AVAILABLE = 2
    PARTIAL_CONDENSING_OSQP_NOT_AVAILABLE = 3
    PARTIAL_CONDENSING_QPDUNES_NOT_AVAILABLE = 4
    FULL_CONDENSING_HPIPM = 5
    FULL_CONDENSING_QPOASES_NOT_AVAILABLE = 6
    FULL_CONDENSING_DAQP_NOT_AVAILABLE = 7
    FULL_CONDENSING_QORE_NOT_AVAILABLE = 8
    FULL_CONDENSING_OOQP_NOT_AVAILABLE = 9
    INVALID_QP_SOLVER = 10
end

struct ocp_qp_solver_plan_t
    qp_solver::ocp_qp_solver_t
end

struct ocp_qp_solver
    config::Ptr{Cint}
    dims::Ptr{Cint}
    opts::Ptr{Cvoid}
    mem::Ptr{Cvoid}
    work::Ptr{Cvoid}
end

function ocp_qp_xcond_solver_config_initialize_from_plan(solver_name, solver_config)
    ccall((:ocp_qp_xcond_solver_config_initialize_from_plan, libacados), Cvoid, (ocp_qp_solver_t, Ptr{Cint}), solver_name, solver_config)
end

function ocp_qp_xcond_solver_config_create(plan)
    ccall((:ocp_qp_xcond_solver_config_create, libacados), Ptr{Cint}, (ocp_qp_solver_plan_t,), plan)
end

function ocp_qp_xcond_solver_config_free(config)
    ccall((:ocp_qp_xcond_solver_config_free, libacados), Cvoid, (Ptr{Cint},), config)
end

function ocp_qp_dims_create(N)
    ccall((:ocp_qp_dims_create, libacados), Ptr{Cint}, (Cint,), N)
end

function ocp_qp_dims_free(dims)
    ccall((:ocp_qp_dims_free, libacados), Cvoid, (Ptr{Cvoid},), dims)
end

function ocp_qp_xcond_solver_dims_create(config, N)
    ccall((:ocp_qp_xcond_solver_dims_create, libacados), Ptr{Cint}, (Ptr{Cint}, Cint), config, N)
end

function ocp_qp_xcond_solver_dims_create_from_ocp_qp_dims(config, dims)
    ccall((:ocp_qp_xcond_solver_dims_create_from_ocp_qp_dims, libacados), Ptr{Cint}, (Ptr{Cint}, Ptr{Cint}), config, dims)
end

function ocp_qp_xcond_solver_dims_free(dims_)
    ccall((:ocp_qp_xcond_solver_dims_free, libacados), Cvoid, (Ptr{Cint},), dims_)
end

function ocp_qp_xcond_solver_dims_set(config_, dims, stage, field, value)
    ccall((:ocp_qp_xcond_solver_dims_set, libacados), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cint}), config_, dims, stage, field, value)
end

function ocp_qp_in_create(dims)
    ccall((:ocp_qp_in_create, libacados), Ptr{Cint}, (Ptr{Cint},), dims)
end

function ocp_qp_in_set(config, in, stage, field, value)
    ccall((:ocp_qp_in_set, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cchar}, Ptr{Cvoid}), config, in, stage, field, value)
end

function ocp_qp_in_free(in_)
    ccall((:ocp_qp_in_free, libacados), Cvoid, (Ptr{Cvoid},), in_)
end

function ocp_qp_out_create(dims)
    ccall((:ocp_qp_out_create, libacados), Ptr{Cint}, (Ptr{Cint},), dims)
end

function ocp_qp_out_free(out_)
    ccall((:ocp_qp_out_free, libacados), Cvoid, (Ptr{Cvoid},), out_)
end

function ocp_qp_out_get(out, field, value)
    ccall((:ocp_qp_out_get, libacados), Cvoid, (Ptr{Cint}, Ptr{Cchar}, Ptr{Cvoid}), out, field, value)
end

function ocp_qp_xcond_solver_opts_create(config, dims)
    ccall((:ocp_qp_xcond_solver_opts_create, libacados), Ptr{Cvoid}, (Ptr{Cint}, Ptr{Cint}), config, dims)
end

function ocp_qp_xcond_solver_opts_free(opts)
    ccall((:ocp_qp_xcond_solver_opts_free, libacados), Cvoid, (Ptr{Cint},), opts)
end

function ocp_qp_xcond_solver_opts_set(config, opts, field, value)
    ccall((:ocp_qp_xcond_solver_opts_set, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cchar}, Ptr{Cvoid}), config, opts, field, value)
end

function ocp_qp_calculate_size(config, dims, opts_)
    ccall((:ocp_qp_calculate_size, libacados), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}), config, dims, opts_)
end

function ocp_qp_assign(config, dims, opts_, raw_memory)
    ccall((:ocp_qp_assign, libacados), Ptr{ocp_qp_solver}, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}), config, dims, opts_, raw_memory)
end

function ocp_qp_create(config, dims, opts_)
    ccall((:ocp_qp_create, libacados), Ptr{ocp_qp_solver}, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}), config, dims, opts_)
end

function ocp_qp_solver_destroy(solver)
    ccall((:ocp_qp_solver_destroy, libacados), Cvoid, (Ptr{ocp_qp_solver},), solver)
end

function ocp_qp_x_cond_solver_free(config, dims, opts_)
    ccall((:ocp_qp_x_cond_solver_free, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}), config, dims, opts_)
end

function ocp_qp_solve(solver, qp_in, qp_out)
    ccall((:ocp_qp_solve, libacados), Cint, (Ptr{ocp_qp_solver}, Ptr{Cint}, Ptr{Cint}), solver, qp_in, qp_out)
end

function ocp_qp_inf_norm_residuals(dims, qp_in, qp_out, res)
    ccall((:ocp_qp_inf_norm_residuals, libacados), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}), dims, qp_in, qp_out, res)
end

@cenum sim_solver_t::UInt32 begin
    ERK = 0
    IRK = 1
    GNSF = 2
    LIFTED_IRK = 3
    INVALID_SIM_SOLVER = 4
end

struct sim_solver_plan_t
    sim_solver::sim_solver_t
end

struct sim_solver
    config::Ptr{Cint}
    dims::Ptr{Cvoid}
    opts::Ptr{Cvoid}
    mem::Ptr{Cvoid}
    work::Ptr{Cvoid}
end

function sim_config_destroy(config)
    ccall((:sim_config_destroy, libacados), Cint, (Ptr{Cvoid},), config)
end

function sim_dims_create(config_)
    ccall((:sim_dims_create, libacados), Ptr{Cint}, (Ptr{Cvoid},), config_)
end

function sim_dims_destroy(dims)
    ccall((:sim_dims_destroy, libacados), Cint, (Ptr{Cvoid},), dims)
end

function sim_dims_set(config, dims, field, value)
    ccall((:sim_dims_set, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cint}), config, dims, field, value)
end

function sim_dims_get(config, dims, field, value)
    ccall((:sim_dims_get, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cint}), config, dims, field, value)
end

function sim_dims_get_from_attr(config, dims, field, dims_out)
    ccall((:sim_dims_get_from_attr, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cint}), config, dims, field, dims_out)
end

function sim_in_destroy(out)
    ccall((:sim_in_destroy, libacados), Cint, (Ptr{Cvoid},), out)
end

function sim_in_set(config_, dims_, in, field, value)
    ccall((:sim_in_set, libacados), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cchar}, Ptr{Cvoid}), config_, dims_, in, field, value)
end

function sim_out_destroy(out)
    ccall((:sim_out_destroy, libacados), Cint, (Ptr{Cvoid},), out)
end

function sim_out_get(config, dims, out, field, value)
    ccall((:sim_out_get, libacados), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cchar}, Ptr{Cvoid}), config, dims, out, field, value)
end

function sim_opts_create(config, dims)
    ccall((:sim_opts_create, libacados), Ptr{Cint}, (Ptr{Cint}, Ptr{Cvoid}), config, dims)
end

function sim_opts_destroy(opts)
    ccall((:sim_opts_destroy, libacados), Cint, (Ptr{Cvoid},), opts)
end

function sim_opts_set(config, opts, field, value)
    ccall((:sim_opts_set, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}), config, opts, field, value)
end

function sim_opts_get(config, opts, field, value)
    ccall((:sim_opts_get, libacados), Cint, (Ptr{Cint}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}), config, opts, field, value)
end

function sim_solver_destroy(solver)
    ccall((:sim_solver_destroy, libacados), Cint, (Ptr{Cvoid},), solver)
end

function sim_solve(solver, in, out)
    ccall((:sim_solve, libacados), Cint, (Ptr{sim_solver}, Ptr{Cint}, Ptr{Cint}), solver, in, out)
end

function sim_precompute(solver, in, out)
    ccall((:sim_precompute, libacados), Cint, (Ptr{sim_solver}, Ptr{Cint}, Ptr{Cint}), solver, in, out)
end

function sim_solver_set(solver, field, value)
    ccall((:sim_solver_set, libacados), Cint, (Ptr{sim_solver}, Ptr{Cchar}, Ptr{Cvoid}), solver, field, value)
end

# exports
const PREFIXES = ["ocp", "sim"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module

using Clang.Generators

cd(joinpath("..", @__DIR__()))
LIB_PATH = "/home/fredrikb/repos/acados/interfaces/acados_c"
headers =  [joinpath(LIB_PATH, header) for header in filter(x -> endswith(x, ".h"), readdir(LIB_PATH))] # include all .h files in the directory
output = "AcadosBindings.jl"

# make sure Clang.jl can find necessary headers included by your header file
clang_args = ["-I", LIB_PATH]

# setup generator options
args = get_default_args()
push!(args, "-I$LIB_PATH")

# wrapper generator options
options = load_options(joinpath(@__DIR__, "Generator.toml"))

# create context
ctx = create_context(headers, args, options)

# run generator
build!(ctx)

println("Bindings generated in $output")

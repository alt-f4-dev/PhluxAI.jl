module JupyTools

using JSON3

import ..ArtifactStore
import ..ProtocolTypes: ArtifactRef, ToolCall, ToolExecution, ToolSpec
import ..Sandboxing

@enum ToolMode begin
    TOOLS_OFF
    TOOLS_ASK
    TOOLS_AUTO
end

@enum ToolRisk begin
    TOOL_READ
    TOOL_WRITE
    TOOL_EXECUTE
    TOOL_NETWORK
end

struct RegisteredTool
    spec::ToolSpec
    risk::ToolRisk
end

struct ToolRegistry
    tools::Dict{String,RegisteredTool}
end

ToolRegistry() = ToolRegistry(Dict{String,RegisteredTool}())

function register!(
    registry::ToolRegistry,
    tool::RegisteredTool,
)::ToolRegistry
    registry.tools[tool.spec.name] = tool
    return registry
end

function tool_specs(registry::ToolRegistry)::Vector{ToolSpec}
    names = sort!(collect(keys(registry.tools)))
    return ToolSpec[registry.tools[name].spec for name in names]
end

function tool_risk(
    registry::ToolRegistry,
    name::AbstractString,
)::Union{ToolRisk,Nothing}
    tool = get(registry.tools, String(name), nothing)
    return isnothing(tool) ? nothing : tool.risk
end

mutable struct ToolContext
    sandbox::Sandboxing.SandboxConfig
    artifacts::ArtifactStore.Store
    registry::ToolRegistry
    jupy_executable::String
    jupip_executable::String
end

function _executable(name::String)::String
    path = Sys.which(name)
    isnothing(path) && throw(ArgumentError(
        "required executable '$name' was not found on PATH",
    ))
    return String(path)
end

function ToolContext(
    sandbox::Sandboxing.SandboxConfig,
    artifacts::ArtifactStore.Store;
    registry::ToolRegistry = default_registry(),
    jupy_executable::Union{AbstractString,Nothing} = nothing,
    jupip_executable::Union{AbstractString,Nothing} = nothing,
)::ToolContext
    jupy = isnothing(jupy_executable) ? _executable("jupy") : String(jupy_executable)
    jupip = isnothing(jupip_executable) ? _executable("jupip") : String(jupip_executable)
    return ToolContext(sandbox, artifacts, registry, jupy, jupip)
end

function _schema(value)::String
    return String(JSON3.write(value))
end

function _tool(
    name::String,
    description::String,
    properties::Dict{String,Any};
    required::Vector{String} = String[],
    risk::ToolRisk = TOOL_READ,
)::RegisteredTool
    parameters = Dict{String,Any}(
        "type" => "object",
        "properties" => properties,
        "additionalProperties" => false,
    )
    isempty(required) || (parameters["required"] = required)
    return RegisteredTool(ToolSpec(name, description, _schema(parameters)), risk)
end

function default_registry()::ToolRegistry
    registry = ToolRegistry()
    register!(registry, _tool(
        "write_text_file",
        "Write UTF-8 text to a relative path in the sandbox workspace.",
        Dict{String,Any}(
            "path" => Dict(
                "type" => "string",
                "description" => "Workspace-relative output path.",
            ),
            "content" => Dict(
                "type" => "string",
                "description" => "Complete UTF-8 file contents.",
            ),
            "overwrite" => Dict(
                "type" => "boolean",
                "description" => "Whether an existing file may be replaced.",
            ),
        );
        required = ["path", "content"],
        risk = TOOL_WRITE,
    ))
    register!(registry, _tool(
        "read_text_file",
        "Read a UTF-8 text file from the sandbox workspace.",
        Dict{String,Any}(
            "path" => Dict("type" => "string"),
            "max_bytes" => Dict(
                "type" => "integer",
                "minimum" => 1,
            ),
        );
        required = ["path"],
        risk = TOOL_READ,
    ))
    register!(registry, _tool(
        "list_files",
        "List files under a relative sandbox workspace directory.",
        Dict{String,Any}(
            "path" => Dict("type" => "string"),
            "recursive" => Dict("type" => "boolean"),
        );
        risk = TOOL_READ,
    ))
    register!(registry, _tool(
        "run_julia_script",
        "Run a Julia script through jupy in the sandbox. Optionally collect " *
        "specified output files as artifacts.",
        Dict{String,Any}(
            "path" => Dict(
                "type" => "string",
                "description" => "Workspace-relative Julia script path.",
            ),
            "threads" => Dict(
                "type" => "string",
                "description" => "Julia thread count or auto.",
            ),
            "artifacts" => Dict(
                "type" => "array",
                "items" => Dict("type" => "string"),
                "description" => "Workspace-relative files to collect.",
            ),
        );
        required = ["path"],
        risk = TOOL_EXECUTE,
    ))
    register!(registry, _tool(
        "install_python_packages",
        "Install Python packages with jupip in the sandbox project.",
        Dict{String,Any}(
            "packages" => Dict(
                "type" => "array",
                "items" => Dict("type" => "string"),
                "minItems" => 1,
            ),
        );
        required = ["packages"],
        risk = TOOL_NETWORK,
    ))
    register!(registry, _tool(
        "inspect_python_package",
        "Inspect one Python package with jupip show.",
        Dict{String,Any}(
            "package" => Dict("type" => "string"),
        );
        required = ["package"],
        risk = TOOL_READ,
    ))
    register!(registry, _tool(
        "collect_artifacts",
        "Copy workspace files into the session artifact store.",
        Dict{String,Any}(
            "paths" => Dict(
                "type" => "array",
                "items" => Dict("type" => "string"),
                "minItems" => 1,
            ),
        );
        required = ["paths"],
        risk = TOOL_READ,
    ))
    return registry
end

function _parse_arguments(call::ToolCall)
    try
        return JSON3.read(call.arguments_json)
    catch error
        throw(ArgumentError(
            "invalid JSON arguments for tool $(call.name): " *
            sprint(showerror, error),
        ))
    end
end

function _string_argument(
    arguments,
    key::Symbol;
    required::Bool = true,
    default::String = "",
)::String
    value = get(arguments, key, nothing)
    if isnothing(value)
        required && throw(ArgumentError("missing argument '$key'"))
        return default
    end
    result = String(value)
    required && isempty(strip(result)) && throw(ArgumentError(
        "argument '$key' must not be empty",
    ))
    return result
end

function _bool_argument(
    arguments,
    key::Symbol,
    default::Bool,
)::Bool
    value = get(arguments, key, nothing)
    return isnothing(value) ? default : Bool(value)
end

function _int_argument(
    arguments,
    key::Symbol,
    default::Int,
)::Int
    value = get(arguments, key, nothing)
    return isnothing(value) ? default : Int(value)
end

function _string_vector(arguments, key::Symbol)::Vector{String}
    value = get(arguments, key, nothing)
    isnothing(value) && return String[]
    return String[String(item) for item in value]
end

function _safe_package(value::AbstractString)::String
    package = String(strip(value))
    isempty(package) && throw(ArgumentError(
        "package name must not be empty",
    ))
    occursin(r"^[A-Za-z0-9_.@/+=<>!~:-]+$", package) ||
        throw(ArgumentError("unsafe package specification '$package'"))
    return package
end

function _format_process(result::Sandboxing.ProcessResult)::String
    buffer = IOBuffer()
    println(buffer, "success: ", result.success)
    println(buffer, "timed_out: ", result.timed_out)
    println(buffer, "exit_code: ", something(result.exit_code, "unknown"))
    println(
        buffer,
        "elapsed_seconds: ",
        round(result.elapsed_seconds; digits = 3),
    )
    if !isempty(result.stdout)
        println(buffer, "stdout:")
        println(buffer, result.stdout)
    end
    if !isempty(result.stderr)
        println(buffer, "stderr:")
        println(buffer, result.stderr)
    end
    return String(take!(buffer))
end

function _collect(
    context::ToolContext,
    paths::Vector{String},
)::Vector{ArtifactRef}
    artifacts = ArtifactRef[]
    for relative in paths
        source = Sandboxing.resolve_workspace_path(
            context.sandbox,
            relative;
            must_exist = true,
        )
        isfile(source) || throw(ArgumentError(
            "artifact path is not a file: $relative",
        ))
        push!(
            artifacts,
            ArtifactStore.import_artifact(context.artifacts, source),
        )
    end
    return artifacts
end

function _execute_write(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    relative = _string_argument(arguments, :path)
    content = _string_argument(arguments, :content)
    overwrite = _bool_argument(arguments, :overwrite, false)
    path = Sandboxing.resolve_workspace_path(context.sandbox, relative)
    isfile(path) && !overwrite && throw(ArgumentError(
        "file already exists and overwrite=false: $relative",
    ))
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    try
        open(temporary, "w") do io
            write(io, content)
            flush(io)
        end
        mv(temporary, path; force = true)
    catch
        isfile(temporary) && rm(temporary; force = true)
        rethrow()
    end
    return ToolExecution(
        call.id,
        call.name,
        true,
        "wrote $(ncodeunits(content)) bytes to $relative",
    )
end

function _execute_read(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    relative = _string_argument(arguments, :path)
    limit = _int_argument(arguments, :max_bytes, 200_000)
    limit > 0 || throw(ArgumentError("max_bytes must be positive"))
    path = Sandboxing.resolve_workspace_path(
        context.sandbox,
        relative;
        must_exist = true,
    )
    isfile(path) || throw(ArgumentError("path is not a file: $relative"))
    filesize(path) <= limit || throw(ArgumentError(
        "file exceeds max_bytes=$limit: $relative",
    ))
    return ToolExecution(
        call.id,
        call.name,
        true,
        read(path, String),
    )
end

function _execute_list(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    relative = _string_argument(
        arguments,
        :path;
        required = false,
        default = ".",
    )
    recursive = _bool_argument(arguments, :recursive, true)
    root = Sandboxing.resolve_workspace_path(
        context.sandbox,
        relative;
        must_exist = true,
    )
    isdir(root) || throw(ArgumentError("path is not a directory: $relative"))
    names = String[]
    if recursive
        for (directory, directories, files) in walkdir(root)
            sort!(directories)
            sort!(files)
            for file in files
                push!(names, relpath(joinpath(directory, file), root))
            end
        end
    else
        names = sort!(readdir(root))
    end
    return ToolExecution(
        call.id,
        call.name,
        true,
        join(names, '\n'),
    )
end

function _execute_julia(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    relative = _string_argument(arguments, :path)
    script = Sandboxing.resolve_workspace_path(
        context.sandbox,
        relative;
        must_exist = true,
    )
    isfile(script) || throw(ArgumentError("script is not a file: $relative"))
    threads = _string_argument(
        arguments,
        :threads;
        required = false,
        default = "auto",
    )
    occursin(r"^(auto|[1-9][0-9]*)$", threads) || throw(ArgumentError(
        "threads must be 'auto' or a positive integer",
    ))
    argv = String[
        context.jupy_executable,
        "-t",
        threads,
        relative,
    ]
    result = Sandboxing.run_sandboxed(context.sandbox, argv)
    requested = _string_vector(arguments, :artifacts)
    artifacts = result.success ? _collect(context, requested) : ArtifactRef[]
    return ToolExecution(
        call.id,
        call.name,
        result.success,
        _format_process(result);
        exit_code = result.exit_code,
        artifacts,
    )
end

function _execute_install(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    context.sandbox.network || throw(ArgumentError(
        "sandbox network access is disabled",
    ))
    packages = _string_vector(arguments, :packages)
    isempty(packages) && throw(ArgumentError(
        "packages must contain at least one package",
    ))
    safe_packages = _safe_package.(packages)
    argv = String[context.jupip_executable, "install"]
    append!(argv, safe_packages)
    result = Sandboxing.run_sandboxed(context.sandbox, argv)
    return ToolExecution(
        call.id,
        call.name,
        result.success,
        _format_process(result);
        exit_code = result.exit_code,
    )
end

function _execute_inspect(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    package = _safe_package(_string_argument(arguments, :package))
    result = Sandboxing.run_sandboxed(
        context.sandbox,
        [context.jupip_executable, "show", package],
    )
    return ToolExecution(
        call.id,
        call.name,
        result.success,
        _format_process(result);
        exit_code = result.exit_code,
    )
end

function _execute_collect(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    paths = _string_vector(arguments, :paths)
    isempty(paths) && throw(ArgumentError(
        "paths must contain at least one file",
    ))
    artifacts = _collect(context, paths)
    output = join(
        ("$(artifact.path) $(artifact.mime_type)" for artifact in artifacts),
        '\n',
    )
    return ToolExecution(
        call.id,
        call.name,
        true,
        output;
        artifacts,
    )
end

function _execute_registered(
    context::ToolContext,
    call::ToolCall,
    arguments,
)::ToolExecution
    call.name == "write_text_file" &&
        return _execute_write(context, call, arguments)
    call.name == "read_text_file" &&
        return _execute_read(context, call, arguments)
    call.name == "list_files" &&
        return _execute_list(context, call, arguments)
    call.name == "run_julia_script" &&
        return _execute_julia(context, call, arguments)
    call.name == "install_python_packages" &&
        return _execute_install(context, call, arguments)
    call.name == "inspect_python_package" &&
        return _execute_inspect(context, call, arguments)
    call.name == "collect_artifacts" &&
        return _execute_collect(context, call, arguments)
    throw(ArgumentError("unknown tool '$(call.name)'"))
end

function _approved(
    mode::ToolMode,
    tool::RegisteredTool,
    call::ToolCall,
    approval_callback::Union{Nothing,Function},
)::Bool
    mode == TOOLS_OFF && return false
    if mode == TOOLS_AUTO && tool.risk != TOOL_NETWORK
        return true
    end

    # Network-capable package installation always requires an explicit
    # approval callback, including in automatic tool mode.
    isnothing(approval_callback) && return false
    return Bool(approval_callback(call, tool.risk))
end

"""Validate, authorize, and execute one model-requested tool call."""
function execute_tool_call(
    context::ToolContext,
    call::ToolCall;
    mode::ToolMode = TOOLS_ASK,
    approval_callback::Union{Nothing,Function} = nothing,
)::ToolExecution
    tool = get(context.registry.tools, call.name, nothing)
    isnothing(tool) && return ToolExecution(
        call.id,
        call.name,
        false,
        "tool is not registered",
    )
    _approved(mode, tool, call, approval_callback) ||
        return ToolExecution(
            call.id,
            call.name,
            false,
            "tool execution was not approved",
        )
    try
        arguments = _parse_arguments(call)
        return _execute_registered(context, call, arguments)
    catch error
        return ToolExecution(
            call.id,
            call.name,
            false,
            sprint(showerror, error),
        )
    end
end

export ToolMode,
       TOOLS_OFF,
       TOOLS_ASK,
       TOOLS_AUTO,
       ToolRisk,
       TOOL_READ,
       TOOL_WRITE,
       TOOL_EXECUTE,
       TOOL_NETWORK,
       RegisteredTool,
       ToolRegistry,
       ToolContext,
       register!,
       tool_specs,
       tool_risk,
       default_registry,
       execute_tool_call

end # module JupyTools

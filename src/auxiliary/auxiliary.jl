include("configurators.jl")
include("helpers.jl")

# Some default settings for the package
function load_default_settings()
    if DEFAULT_SETTING
        # TODO: Check how to better utilize custom `log` and `sqrt` functions from Trixi.jl
        set_log_type("log_Base")
        set_sqrt_type("sqrt_Base")
    else
        # To be implemented
        println("Using custom setting...")
    end
end

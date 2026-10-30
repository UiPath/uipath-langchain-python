"""The static-args display contract shared by the runtime and the graphs."""

# Run metadata key carrying the display values of static args, keyed by tool
# name, so conversation events show the args the tool is actually called with.
STATIC_ARGS_DISPLAY_METADATA_KEY = "uipath_static_args_display"

# Shown in place of sensitive static arg values.
SENSITIVE_ARG_PLACEHOLDER = "<hidden>"

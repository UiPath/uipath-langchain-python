"""Import-time warm-up for hosts that preload modules while a process boots.

``uipath server`` imports a configurable list of modules before a pooled
instance accepts its first job. Importing this module runs
:func:`warm_code_interpreter` at that point, so the first advanced agent on the
instance does not pay the WebAssembly compile itself.
"""

from uipath_langchain.agent.advanced.code_interpreter import warm_code_interpreter

warm_code_interpreter()

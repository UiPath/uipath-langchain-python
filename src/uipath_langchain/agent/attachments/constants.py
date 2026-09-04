"""Names shared between an output-file's schema handling and its tool.

A leaf module so neither side has to import the other: the tool module builds
the tool, and the attachments module builds the prompt and the corrective
messages that name it.
"""

OUTPUT_FILE_TOOL_NAME = "create_output_file"

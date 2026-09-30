"""v2: v1 plus rules for running the operations entities declare.

Used only when at least one entity declares operations; the operations
themselves are listed per entity by the prompt builder.
"""

from .v1 import TEMPLATE as V1_TEMPLATE

_DOMAIN_GUIDANCE = "{domain_guidance}"

_OPERATIONS = """
ENTITY OPERATIONS:
Some entities declare operations, listed under "Operations for <table>" in the \
entity schemas below. Run one with the ``execute_operation`` tool.
- Questions about the data go to ``execute_sql``.
- Use ``execute_operation`` when the request asks for the action an operation \
performs, or when a Read operation answers the question directly.
- Pass the entity's SQL table name, the operation name and the parameter names \
exactly as listed. Never invent an operation or a parameter.
- A Mutation changes data. ``execute_operation`` refuses one unless the request \
allows changes. If it refuses, say so in a plain text reply and do not look for \
another way to make the change.
- Never run a Mutation again after it returned ``Wrote``.
- ``Returned``, ``Wrote`` and ``NoChange`` are final. On ``Refused`` or \
``Faulted``, read ``errors``, then fix the arguments or explain the failure in a \
plain text reply.
"""

TEMPLATE = V1_TEMPLATE.removesuffix(_DOMAIN_GUIDANCE) + _OPERATIONS + _DOMAIN_GUIDANCE

"""Library data in a user turn (Story 2.8): the document delimiter every prompt puts it in, for every provider.

Azure's jailbreak check (Prompt Shields) reads only the user turn outside `DOCUMENTS_OPEN` … `DOCUMENTS_CLOSE`; the
documents inside get its indirect-attack check only when the deployment's content filter enables it. To any other
provider it is plain text. Every piece of library text inside is escaped by its caller, so it can end neither the
delimiter nor its own block. Story 4.2 moved these here from `lanes.answer`, so the ingestion records step shares them.
"""

from __future__ import annotations

DOCUMENTS_OPEN = '""" <documents>'
DOCUMENTS_CLOSE = '</documents> """'
# How a section path is written in a block's `section` attribute.
SECTION_SEPARATOR = " > "

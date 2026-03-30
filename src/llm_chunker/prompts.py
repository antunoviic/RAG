BOUNDARY_PROMPT = """\
Given these numbered sentences, return the index (0-based) after which the most natural semantic boundary occurs. Return ONLY valid JSON, no other text.

{sentences}

Return: {{"split_after": <sentence_index>, "reason": "<brief reason>"}}"""

FILTER_PROMPT = """\
Rate this text chunk's information density for a RAG knowledge base (1=useless filler, 10=highly informative). Return ONLY valid JSON, no other text.

Chunk: "{chunk}"

Return: {{"score": <1-10>, "keep": <true/false>}}"""

METADATA_PROMPT = """\
Generate a short descriptive title and exactly 3 keywords for this text chunk. Return ONLY valid JSON, no other text.

Chunk: "{chunk}"

Return: {{"title": "<title>", "keywords": ["<kw1>", "<kw2>", "<kw3>"]}}"""

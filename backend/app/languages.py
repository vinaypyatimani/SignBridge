"""Supported languages. Sign languages use ISO 639-3 codes, spoken languages ISO 639-1."""

SIGN_LANGUAGES = {
    "ase": {"name": "American Sign Language", "short": "ASL", "gloss_language": "en"},
    "gsg": {"name": "German Sign Language (Deutsche Gebärdensprache)", "short": "DGS", "gloss_language": "de"},
    "ssp": {"name": "Spanish Sign Language (Lengua de Signos Española)", "short": "LSE", "gloss_language": "es"},
}

SPOKEN_LANGUAGES = {
    "en": {"name": "English"},
    "de": {"name": "German"},
    "es": {"name": "Spanish"},
}

# Notes the LLM gets about each sign language's structure when building glosses.
GRAMMAR_NOTES = {
    "ase": "Common ASL order is time - topic - comment; WH-question signs often come at the end; "
           "no articles or forms of 'to be'; plural and tense are shown by separate signs (e.g. FINISH, WILL).",
    "gsg": "DGS is verb-final (subject - object - verb); time markers come first; no articles; "
           "questions are marked by facial expression and a question sign, often at the end.",
    "ssp": "LSE commonly places time first and the verb late (subject - object - verb); no articles; "
           "questions use a question sign, often at the end, with facial marking.",
}


def check(sign_lang: str, spoken_lang: str) -> None:
    if sign_lang not in SIGN_LANGUAGES:
        raise ValueError(f"unknown sign language {sign_lang!r}; use one of {sorted(SIGN_LANGUAGES)}")
    if spoken_lang not in SPOKEN_LANGUAGES:
        raise ValueError(f"unknown spoken language {spoken_lang!r}; use one of {sorted(SPOKEN_LANGUAGES)}")

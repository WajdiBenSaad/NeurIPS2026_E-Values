from __future__ import annotations


class FixedLanguageRouter:
    def __init__(self, translated_languages: list[str]):
        self.translated = set(translated_languages)

    def route(self, language_code: str) -> str:
        return 'translate' if language_code in self.translated else 'direct'

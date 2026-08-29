from __future__ import annotations


class DirectOnlyRouter:
    def route(self, language_code: str) -> str:
        return 'direct'


class TranslationOnlyRouter:
    def __init__(self, eligible_languages: set[str]):
        self.eligible = eligible_languages

    def route(self, language_code: str) -> str:
        return 'translate' if language_code in self.eligible else 'direct'


class SelectedLanguageRouter:
    def __init__(self, selected_languages: list[str]):
        self.selected = set(selected_languages)

    def route(self, language_code: str) -> str:
        return 'translate' if language_code in self.selected else 'direct'

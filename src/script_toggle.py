import re
from typing import Literal

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


ScriptMode = Literal["auto", "devanagari", "roman"]

DEVANAGARI_CHAR_PATTERN = re.compile(r"[\u0900-\u097F]")
DEVANAGARI_TOKEN_PATTERN = re.compile(r"([\u0900-\u097F]+)")
LATIN_TOKEN_PATTERN = re.compile(r"([a-zA-Z]+)")


class ScriptToggler:
    @staticmethod
    def contains_devanagari(text: str) -> bool:
        return bool(DEVANAGARI_CHAR_PATTERN.search(text))

    @staticmethod
    def devanagari_to_roman(text: str) -> str:
        if not text:
            return ""

        def _convert_match(match: re.Match[str]) -> str:
            token = match.group(1)
            roman = transliterate(token, sanscript.DEVANAGARI, sanscript.ITRANS)
            roman = roman.lower()
            roman = roman.replace("~n", "n").replace(".n", "n").replace("m^", "m")
            roman = roman.replace("aa", "a").replace("ii", "i").replace("uu", "u")
            roman = roman.replace("shh", "sh").replace("chh", "chh")
            return roman

        converted = DEVANAGARI_TOKEN_PATTERN.sub(_convert_match, text)
        return re.sub(r"\s+", " ", converted).strip()

    @staticmethod
    def roman_to_devanagari(text: str) -> str:
        if not text:
            return ""

        def _convert_match(match: re.Match[str]) -> str:
            token = match.group(1)
            return transliterate(token.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)

        converted = LATIN_TOKEN_PATTERN.sub(_convert_match, text)
        return re.sub(r"\s+", " ", converted).strip()

    def convert(self, text: str, target_script: ScriptMode = "auto") -> str:
        if not text or target_script == "auto":
            return text.strip()
        if target_script == "roman":
            return self.devanagari_to_roman(text)
        if target_script == "devanagari":
            return self.roman_to_devanagari(text)
        raise ValueError(f"Unsupported target_script: {target_script}")

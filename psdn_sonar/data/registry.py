"""Known public HuggingFace dataset/config mappings per language.

Each entry maps a short dataset name to its HuggingFace ID, the config
template (with ``{lang}`` as a placeholder), the column holding the
transcription text, and the column holding audio data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .catalog import load_catalog


@dataclass(frozen=True)
class DatasetSpec:
    """Blueprint for a single public HuggingFace dataset."""

    hf_id: str
    config_template: str
    text_column: str = "sentence"
    audio_column: str = "audio"
    splits: tuple[str, ...] = ("train", "validation", "test")
    # When set, for these languages the dataset has no config (single default config).
    no_config_langs: Optional[frozenset[str]] = None
    revision: str = ""
    enabled: bool = True


@dataclass
class AvailableDataset:
    """Result of a discovery check — a dataset confirmed to exist for a language."""

    name: str
    hf_id: str
    config: str
    splits: list[str] = field(default_factory=list)
    num_examples: Optional[dict[str, int]] = None
    text_column: str = "sentence"
    audio_column: str = "audio"
    revision: str = ""


FLEURS_CONFIG: dict[str, str] = {
    "af": "af_za",
    "ar": "ar_eg",
    "bn": "bn_in",
    "cs": "cs_cz",
    "da": "da_dk",
    "de": "de_de",
    "el": "el_gr",
    "en": "en_us",
    "es": "es_419",
    "fi": "fi_fi",
    "fr": "fr_fr",
    "he": "he_il",
    "hi": "hi_in",
    "hu": "hu_hu",
    "id": "id_id",
    "it": "it_it",
    "ja": "ja_jp",
    "ko": "ko_kr",
    "mr": "mr_in",
    "ms": "ms_my",
    "nl": "nl_nl",
    "no": "nb_no",
    "pl": "pl_pl",
    "pt": "pt_br",
    "ro": "ro_ro",
    "ru": "ru_ru",
    "sv": "sv_se",
    "sw": "sw_ke",
    "ta": "ta_in",
    "te": "te_in",
    "th": "th_th",
    "tr": "tr_tr",
    "uk": "uk_ua",
    "ur": "ur_pk",
    "vi": "vi_vn",
    "zh": "cmn_hans_cn",
}

MLS_CONFIG: dict[str, str] = {
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
}

_BENCHMARK_CATALOG = load_catalog()


def _hf_dataset_spec(name: str, **kwargs) -> DatasetSpec:
    benchmark = _BENCHMARK_CATALOG.get(name)
    if benchmark.source.kind != "huggingface":
        raise ValueError(f"catalog benchmark {name!r} is not a Hugging Face source")
    return DatasetSpec(
        hf_id=benchmark.source.identifier,
        config_template=benchmark.config_template,
        revision=benchmark.source.revision,
        enabled=benchmark.runtime == "enabled" and benchmark.availability == "active",
        splits=benchmark.splits,
        **kwargs,
    )


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    # The former Hugging Face mirror is now a tombstone. Keep the name for
    # explicit disabled errors, but never attempt runtime discovery/loading.
    "common_voice": DatasetSpec(
        hf_id="mozilla-foundation/common_voice_17_0",
        config_template="{lang}",
        enabled=False,
        splits=(),
        text_column="sentence",
        audio_column="audio",
    ),
    "fleurs": _hf_dataset_spec(
        "fleurs",
        text_column="transcription",
        audio_column="audio",
    ),
    "zeroth": _hf_dataset_spec(
        "zeroth",
        text_column="text",
        audio_column="audio",
        no_config_langs=frozenset({"ko"}),
    ),
    "voxpopuli": _hf_dataset_spec(
        "voxpopuli",
        text_column="raw_text",
        audio_column="audio",
    ),
    "multilingual_librispeech": _hf_dataset_spec(
        "multilingual_librispeech",
        text_column="transcript",
        audio_column="audio",
    ),
}

VOXPOPULI_LANGS = frozenset(
    {
        "cs",
        "de",
        "en",
        "es",
        "fi",
        "fr",
        "hu",
        "it",
        "nl",
        "pl",
        "ro",
    }
)

MLS_LANGS = frozenset(
    {
        "de",
        "es",
        "fr",
        "it",
        "nl",
        "pl",
        "pt",
    }
)

_DATASET_LANG_GATES: dict[str, frozenset[str]] = {
    "voxpopuli": VOXPOPULI_LANGS,
    "multilingual_librispeech": MLS_LANGS,
    "zeroth": frozenset({"ko"}),
}


def resolve_config(spec: DatasetSpec, lang: str) -> Optional[str]:
    """Return the HF config string for *lang*, or ``None`` if unsupported.

    For datasets with no_config_langs (e.g. Zeroth), returns "" when lang is in that set,
    meaning load_dataset should be called without a config argument.
    """
    if spec.no_config_langs and lang in spec.no_config_langs:
        return ""
    tpl = spec.config_template
    if "{fleurs}" in tpl:
        return FLEURS_CONFIG.get(lang)
    if "{mls}" in tpl:
        return MLS_CONFIG.get(lang)
    if "{lang}" in tpl:
        return tpl.replace("{lang}", lang)
    return None

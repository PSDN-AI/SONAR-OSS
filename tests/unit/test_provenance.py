"""Tests for model training-corpus provenance and domain markers (issue #119)."""

from psdn_sonar.models.provenance import (
    IN_DOMAIN,
    NOT_DECLARED,
    UNKNOWN,
    declared_training_datasets,
    evaluation_domain,
)
from psdn_sonar.models.registry import list_models


class TestEvaluationDomain:
    def test_declared_overlaps_are_in_domain(self):
        # The three card-declared overlaps from issue #119.
        assert evaluation_domain("khushids_bengali", "fleurs") == IN_DOMAIN
        assert evaluation_domain("wav2vec2_bengali", "openslr53") == IN_DOMAIN
        assert evaluation_domain("kresnik_wav2vec2_large_xlsr_korean", "zeroth") == IN_DOMAIN

    def test_registry_alias_shares_checkpoint_declaration(self):
        # wav2vec2_xlsr_korean is the same kresnik checkpoint under another name.
        assert evaluation_domain("wav2vec2_xlsr_korean", "zeroth") == IN_DOMAIN

    def test_audited_model_on_other_dataset_is_not_declared(self):
        assert evaluation_domain("khushids_bengali", "zeroth") == NOT_DECLARED
        assert evaluation_domain("kresnik_wav2vec2_large_xlsr_korean", "fleurs") == NOT_DECLARED

    def test_hosted_apis_are_unknown(self):
        for api in ("whisper_api", "elevenlabs_api", "assemblyai_api"):
            assert evaluation_domain(api, "fleurs") == UNKNOWN
            assert evaluation_domain(api, "openslr53") == UNKNOWN

    def test_unaudited_and_custom_models_are_unknown(self):
        assert evaluation_domain("whisper_base_en", "fleurs") == UNKNOWN
        assert evaluation_domain("custom_some_model", "fleurs") == UNKNOWN


class TestDeclaredTrainingDatasets:
    def test_returns_none_when_unaudited(self):
        assert declared_training_datasets("whisper_base_en") is None

    def test_returns_declared_set(self):
        assert declared_training_datasets("khushids_bengali") == frozenset({"fleurs"})

    def test_all_audited_names_are_registered_models(self):
        # Guards against the provenance map drifting from the registry when
        # models are renamed or removed.
        from psdn_sonar.models.provenance import _DECLARED_TRAINING_DATASETS

        registered = set(list_models())
        for name in _DECLARED_TRAINING_DATASETS:
            assert name in registered, f"provenance entry {name!r} is not a registered model"

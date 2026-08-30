# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `psdn-sonar leaderboard` renders a comparison table from the
  `scores_<model>.json` artifacts that evaluation runs write (#117). It shows
  measured numbers only: metrics absent from every contributing run render as
  `—` and are never derived or back-solved from other columns — the failure
  mode behind #117, where the published web leaderboard back-solved WER/CER
  from POSEIDON medians and rendered generated distributions as measurements.
  Runs are grouped per model and language, multiple runs are averaged with
  the run count shown, rows whose artifacts carry configuration warnings are
  marked with `!`, and `--sort`/`--language`/`--json` control the view. The
  fabricated web leaderboard is being removed from `PSDN-AI/psdn-portals`;
  this command is its measured-data replacement.
- Documented what the published scores measure and made the caveats
  machine-readable (#119): `docs/SCORE_INTERPRETATION.md` states the
  project's position on the three reader traps — multi-speaker WER/CER
  measures the preprocessing + ASR pipeline end to end (the multi CLI now
  logs this scope at run start), scores are comparable within a dataset
  only (`precompute_benchmarks.py` now publishes per-dataset
  utterance-length stats as `public_length_stats_<language>.json`), and
  cells where a model is evaluated on a corpus its card declares as
  training data are marked (`psdn_sonar.models.provenance` records the
  audited card declarations; the precompute script writes per-cell
  `domain_markers.json` and warns when running an in-domain pair).
  Leaderboard rendering of the markers is owned by `PSDN-AI/psdn-portals`.
- `scores.json` now carries a `lineage` block recording the facts needed to
  compare two runs like-for-like (#120): the HuggingFace repo id and
  checkpoint commit SHA actually loaded (`hf_model_id`/`hf_revision` — the
  model registry pins no revisions, so this is the only record of which
  weights produced the numbers) and the WER normalization contract in force
  (`normalization`, e.g. `bn:v1+bnlp`). The Bengali contract marks whether
  the optional `bnlp` tokenizer was active, because its silent
  whitespace-split fallback changes tokenization — and therefore absolute
  WER/POSEIDON — between otherwise identical environments.

### Changed

- Fixed the three obstacles a new reader hits on the README → USAGE path
  (#112). `make setup-ml` now installs the frozen environment with the
  `[ml]` extra the USAGE examples require, and the README contributor
  section shows it (plus `pip install -e ".[dev,ml]"`) instead of leaving
  `[ml]` to a footnote. The plain-pip block now carries the same
  create-a-venv prerequisite as the TestPyPI section, naming the PEP 668
  `externally-managed-environment` failure it prevents. And the bundled
  example now says what it produces: `examples/sample_audio/single/
  sample.wav` is a 0.4 s synthetic sine tone, so the recommended command
  scores WER ~1.0 by design — the disclosure sits next to the command and
  the fixture description, with a pointer to `psdn-sonar discover` for real
  speech.
- Documented what a first run actually costs and which device it runs on
  (#111). The README now states the order of magnitude up front — several GB
  of downloads and tens of minutes before the first number (measured: FLEURS
  Bengali ~3.4 GB, `[ml]` extra ~1.5 GB on disk, the ~64 MB semantic scorer
  and ~390 MB UTMOS checkpoint fetched lazily on first use) — with matching
  notes in `docs/USAGE.md` and the FAQ pre-run checklist, including measured
  CPU throughput (~1 s/sample CTC vs ~20 s/sample Whisper-class). The
  `--max-samples` help for `discover` and `custom` now says it bounds
  processing only: each requested split is still downloaded to the
  HuggingFace cache in full on first run.

### Removed

- The end-to-end QA testing guide (`docs/SONAR-OSS-E2E-QA-Testing-Guide.md`).
  It was an internal QA workflow document — environment setup runbooks, test
  matrices, and sign-off checklists for the QA team — and does not belong in
  the published repository. QA runbooks are distributed to the QA team
  directly; user-facing usage documentation lives in `docs/USAGE.md` and
  `docs/FAQ.md`.

### Fixed

- `--sweep` can now reach a method set other than the packaged config's, and
  its caution matches what the run actually did (#210). The `multi` subcommand
  had no option for either the method list or the config file, so a sweep
  always covered whatever `psdn_sonar/multi_speaker_config.yaml` listed — one
  method, `no_trim` — while the help described "all methods" and the run
  printed the oracle-bias warning regardless; editing the file inside the
  installed package was the only way to change it. `multi` now takes
  `--methods` (an explicit list, validated against the known methods, since it
  bypasses the config loader's own check) and `--preprocessing-config` (a path
  to another YAML, whose silence/timestamp/pyannote settings are honoured too).
  `run_multispeaker_evaluation` already accepted `methods`; the CLI simply
  never passed it. The oracle-bias caution now fires only when the sweep has
  more than one active method, and names the methods actually swept rather than
  the ones requested — with a single method it says the run is equivalent to
  `--method <name>` and how to widen the set. Runs without `--sweep` log the
  active set. The fallback method list is no longer declared in two places:
  `core` uses `config_loader.DEFAULT_METHODS` instead of its own copy, so which
  set a caller got no longer depends on which declaration it reached. And
  `--method`'s help names all six values it accepts, not four. The packaged
  config is unchanged: its method list is also the candidate set for per-clip
  auto-selection on ordinary runs, so widening it would re-baseline every
  `multi` run and is a separate decision.

  Making the method set reachable exposed four ways it could go wrong quietly,
  all closed here. A config file the caller *names* is now used or the run
  stops — a missing path, an unreadable file or a config with no known methods
  used to warn and silently evaluate with the default methods instead, and a
  malformed one (`methods: 5`, `silence: oops`) escaped as a bare `TypeError`;
  the no-argument path stays lenient so a damaged install still runs.
  `--method` and `--methods` are mutually exclusive, and a pinned `--method` is
  now the active set: it previously lost to a configured per-clip method that
  ran in its place, or aborted the run with "No valid preprocessing methods"
  while the pinned method was perfectly usable. Repeated methods are collapsed
  with a warning — under `--sweep` a repeat doubled the ASR calls per clip and
  counted itself twice in the caution while still producing one score.
- The judge-model guidance in the `llm_metrics` module docstring now names a
  mechanism that exists (#211). It told the reader to opt into a stronger
  judge "via `--judge-model gemini-3.1-pro-preview` on the analysis script";
  no subcommand takes that flag, and `scripts/` holds no such script. These
  metrics are a library surface with no CLI entry point, so the docstring
  now names what actually selects a judge — the `model=` argument on
  `evaluate_sample` and its siblings, and the `judge_model` argument that
  keeps `make_cache_key`'s rows for different judges apart. The paid-tier
  caveat on `gemini-3.1-pro-preview` was independently confirmed and is kept.
- Number verbalization no longer half-converts a digit run glued to a Latin
  letter (#209). `_DIGIT_RUN_RE` guarded both sides against a Latin letter but
  not against another digit, so the engine matched a *proper sub-run* of a
  glued run from either direction: a greedy `\d+` backtracked out of `"15m"`
  to `"1"` and emitted `one5m` (`일5m`, `एक5m`; `"100MB"` -> `ten0mb`), and on
  `"iPhone15"` it started the match at `"5"` and emitted `iphone1five`. Both
  contradicted the module's stated contract, and since normalization runs on
  the reference and the hypothesis before WER and CER, a unit written in
  Latin on one side and in the target script on the other was driven further
  apart by the normalizer than by the transcription. Every case the tests
  covered used a single-digit run (`"v2"`, `"H2O"`, `"2nd"`), which has no
  sub-run to fall back on, so none of them could catch it. Both lookarounds
  now count a digit as token glue. English, Korean and Hindi were affected;
  Bengali was not.
- `timestamp_trim` no longer scores the second speaker on 100 ms of padding
  (#205). Transcript `start`/`end` offsets are on the combined-recording
  timeline (the shipped fixtures' and FAQ schema's convention, now stated in
  `docs/FAQ.md`), but they were clamped against each speaker's own channel
  file — for the speaker who talks second, the start lay past the end of
  their file, every segment was dropped, and the trailing padding was
  exported, transcribed, and scored as their turn: `Samples: 2, Failed: 0`,
  empty `error` column, exit 0, WER 1.0 fabricated from silence. The trim
  now picks its source by timeline fit: a channel file that spans the
  offsets is trimmed in place (keeping channel isolation), otherwise the
  segments are cut from the `<audio_id>_Combined_Audio.wav` recording the
  offsets actually describe — `run_single_method`/`run_sweep` now hand it to
  the strategy. When neither source can hold the offsets, or no segment
  overlaps the chosen source, the method fails for that speaker with an
  error naming the mismatch instead of exporting padding as a success.
- Evaluations no longer make an unbounded network call to the NLTK data host
  at import time (#204). With the `[bengali]` extra installed,
  `text_processing` imported `bnlp` at module scope, and `bnlp`'s module
  body checks NLTK for `tokenizers/punkt` but downloads `punkt_tab` —
  mismatched names, so on an environment holding `punkt_tab` (what the
  tokenizer actually uses) the check failed forever and `nltk.download` ran
  on every import: a timeout-less round trip that landed on English, Hindi
  and Korean runs too, and was observed hanging a run indefinitely when the
  remote closed the connection mid-transfer. The bnlp import is now lazy
  (non-Bengali runs never touch it) and every bnlp import in the package
  goes through a guard: when `punkt_tab` is already local the stale `punkt`
  probe is answered with it, so the import is fully offline — no download,
  no misleading "punkt not found. downloading..." line; when the resource is
  genuinely missing the download runs under a 60-second socket timeout, so
  a dead connection fails the run instead of hanging it. The Bengali
  normalization contract's `+bnlp`/`-bnlp` marker now reflects the tokenizer
  that actually loaded, not merely whether the package is importable.
- Three registered Whisper fine-tunes transcribe again when a language is
  requested (#203) — a regression from #186's `--language` forwarding.
  `tugstugi_bengali`, `tugstugi_bengali_regional`, and
  `whisper_hindi_large_v2` ship a `generation_config.json` with no
  `lang_to_id`/`task_to_id` maps (single-language fine-tunes, the language
  baked into the weights), so `generate(language=...)` raised "The
  generation config is outdated" on every utterance — and since the CLI
  substitutes `bn` when `--language` is omitted, no CLI path avoided it for
  the two Bengali defaults. The pipeline adapter now passes
  `language`/`task` only to checkpoints whose generation config can resolve
  them — mirroring the exact `hasattr` check transformers performs — and
  otherwise logs one load-time warning naming the model and the dropped
  language; the checkpoint transcribes in its fine-tuned language, as it
  did before forwarding existed. `whisper_small_hi` (a multilingual base
  pinned to `hi` in the registry) keeps receiving its language. The same
  guard covers `--hf-model` custom Whisper checkpoints, which take the same
  per-utterance failure through `CustomHuggingFaceModel`.
- The LLM judge's default model resolves again, and the test that guards it
  finally runs somewhere (#187). Google retired the entire stable Gemini 2.5
  generation for new users, taking both names the module hardcoded with it:
  `gemini-2.5-pro` (the default) and `gemini-2.5-flash` (the documented cheap
  alternative) now 404 with "no longer available to new users", so the judge
  failed on its own default. `DEFAULT_MODEL` is now `gemini-3.6-flash`, the
  name verified live on a fresh key; `gemini-3.1-pro-preview` is the
  documented stronger opt-in (paid tier only — its free-tier quota is 0). The
  old docstring preferred stable tier over recency precisely to avoid this
  404, and events inverted it: the stable generation retired first while the
  previews stayed up. The real guard was never a naming heuristic but
  `TestLiveSmoke::test_default_model_string_is_live` — which was opt-in and
  wired to nothing (zero occurrences of `RUN_LIVE_GEMINI_TESTS` under
  `.github/workflows/`). A scheduled `live-gemini` workflow now runs it
  weekly against the live API (needs the `GEMINI_API_KEY` repository secret;
  until that's configured the run warns instead of silently passing), and a
  unit test pins the wiring so the workflow can't be deleted unnoticed.
  Cached judgments key on the model string, so the default change
  auto-invalidates them — no stale cross-judge comparisons.
- `--language` now reaches a registered model's constructor (#186).
  `create_model` used its `language` argument only on the `custom_hf_model`
  branch; registered models were built from registry kwargs alone, and the
  three hosted-API adapters are registered with empty kwargs — so their
  constructor defaults always won. Every AssemblyAI request said Bengali
  (`--language en` runs included: Bengali audio came back phonetically right
  but in Devanagari, WER 1.0), and every ElevenLabs request sent `ben`.
  `create_model` now forwards `language` to any constructor that declares
  the parameter — unless the registry entry pins one (`whisper_small_hi`
  stays Hindi whatever `--language` says) — and the ElevenLabs adapter owns
  the ISO 639-1 → vendor-code conversion, so the `custom` subcommand's
  duplicate mapping is gone. The same dropped-kwargs line made
  `AssemblyAIAPIModel`'s documented streaming mode unreachable, and with it
  `ttft_s`: `psdn-sonar single` now has `--streaming`, which requests the
  streaming protocol from adapters that have one and records `ttft_s` plus
  the TTFT percentiles; a model without a streaming mode logs a warning and
  runs batch, and `scores.json` records the protocol actually used rather
  than reading only the `SONAR_PROTOCOL` env var. Relatedly (from the issue
  discussion): the unused `elevenlabs` SDK is out of the `[apis]` extra —
  the ElevenLabs adapter deliberately speaks the REST API via `requests` (a
  core dependency), so `elevenlabs_api` works without the extra; the
  pyproject now says so.
- `scores.json` no longer misdescribes the run it records (#184) — three
  instances of the same defect class, the artifact asserting things the run
  didn't do. A hosted-API run recorded `provider: local` / `region: local`
  because the submission block read undocumented `SONAR_PROVIDER`/
  `SONAR_REGION` env vars with a `local` default, never the model that ran;
  `provider` now comes from the adapter that actually served inference
  (`openai`/`elevenlabs`/`assemblyai`, or `local` for in-process models),
  `model_snapshot` records the provider-side model id actually requested
  (e.g. `whisper-1`) instead of the registry alias the artifact already
  carries as `model_name`, and `region` is null unless explicitly supplied —
  hosted providers do not disclose one, so the toolkit no longer invents it.
  The env overrides remain and are now documented in `.env.example`. The
  fallback-normalizer caveat (a run scored with the generic normalization,
  e.g. `--language sw`) is now recorded in the `warnings` array with the same
  wording the terminal prints — it makes the same
  these-numbers-carry-a-caveat claim as the script-mismatch warning (#148)
  that was already recorded, and only one of the two was auditable from the
  output files. And `prompt_version` is no longer stamped onto every
  POSEIDON run: it was gated on `compute_sem`, which is local
  sentence-transformers similarity and has nothing to do with the LLM judge,
  so every ordinary run asserted an LLM-judge rubric hash for judgments that
  never happened — alongside a `judge_model` field that echoed
  `SONAR_JUDGE_MODEL`/`GEMINI_MODEL` env vars never wired to the judge. Both
  fields now stay null on this path; a caller that actually runs the judge
  supplies its own submission block.
- The LLM-judge path now reads `.env` (#188) — the same gap #167 closed for
  the `multi` subcommand, on the one path that remained.
  `llm_metrics.get_client()` read `os.getenv` directly without ever calling
  `load_env()`, and since the LLM-judged metrics have no CLI entry point, the
  direct library call was both the only way to reach them and the only path
  in the package that never loaded `.env`: a `GEMINI_API_KEY` configured
  there was present and loadable yet invisible, and only a shell-exported
  key worked. `get_client()` now loads `.env` before the credential check.
  The four places describing this contract also disagreed with each other —
  the README and `.env.example` did not mention Gemini at all, the error
  message said "in the environment" while the ElevenLabs adapter in the same
  codebase instructed ".env or as env var" for the identical contract, and a
  test docstring promised the preferred env name was "documented in README".
  All four now agree: `.env.example` lists `GEMINI_API_KEY` (preferred) and
  `GOOGLE_API_KEY` (alternative), the README documents them next to the
  other API keys, the error message names both mechanisms and points at
  `.env.example`, and a doc-contract test pins the agreement so the four
  cannot silently drift apart again.
- `pyannote_diarize` no longer assigns every word to one speaker and drops the
  other (#189). pyannote.audio 4.x returns a `DiarizeOutput` dataclass whose
  `speaker_diarization` attribute holds the `Annotation`; only 3.x returned the
  `Annotation` directly, and the reader was guarded on `hasattr(..., "itertracks")`,
  so under 4.x every speech turn was silently discarded. With no segments, each
  word fell into an `"unknown"` bucket, one speaker held both references' words,
  and the other left the evaluation with no error — on a clean non-overlapping
  two-channel input, which is why the reporter's level-matched control
  reproduced it exactly: the audio never entered into it. Both output shapes are
  now read, an unrecognised one raises instead of reporting "no speakers", and
  words that fall between turns go to the nearest turn rather than inventing a
  speaker that competes for a reference. A clip whose diarization yields no
  turns, fewer than the two speakers requested, or no word timestamps now fails
  with that reason instead of scoring one speaker against both references.
  Per-clip methods also gained the capability precheck they never had:
  `supports_word_timestamps` previously had no reader anywhere, so a model
  without it (any `whisper_*` adapter) reached the strategy and failed on a bare
  `NotImplementedError`, whose `str()` is empty — the run reported
  `pyannote_diarize failed:` with no reason and recorded that method as the best
  one. Relatedly, an undefined standard deviation (fewer than two samples) is
  now reported as `n/a (n<2)` in both the summary file and the CLI table, which
  previously disagreed (`Std 0.0000` versus `nan`) about the same quantity.
- A run without the `[ml]` extra no longer reports clean success while
  semantic similarity and POSEIDON are silently null (#191). The
  `ModuleNotFoundError` for sentence-transformers was swallowed by the
  batch-semantics `except Exception` as a warning-with-traceback, so a
  core-only install evaluating a hosted API model (reachable with no extras
  at all) produced `successful: N, failed: 0`, null headline metrics, and an
  empty `warnings` array — nothing in any artifact said the metrics were
  missing or why. The evaluator now checks for the dependency up front when
  semantics are requested and emits the same one-actionable-line style as
  the other dependency paths (#169/#177) — naming
  `pip install "psdn-sonar[ml]"` — before any transcription time or API
  spend, and records that line in the `scores.json` `warnings` array so a
  reader of the artifact alone can tell. A genuine runtime failure in the
  semantics batch (which has no known remedy) keeps its traceback in the
  log and is now also recorded in `warnings` instead of vanishing. WER/CER
  behavior is unchanged.
- `discover` now stops the moment the disk cannot fit the next download,
  instead of downloading for hours past huggingface_hub's "not enough free
  disk space" warning until the disk is 99% full (#183). The hub warns with
  the expected file size *before* each download; the preparer promotes that
  warning to an error, and any disk-full evidence in a failed split's
  exception chain (`ENOSPC` errno, "No space left on device", the Rust-style
  "os error 28") now aborts the run rather than being downgraded to a
  per-split warning. The failure surfaces through the CLI's clean one-line
  `OSError` path (#149) and names what the old traceback did not: the disk,
  the partial download left in the HuggingFace cache and where to delete it,
  and that `--max-samples` bounds preparation, not the download. The
  "No samples could be loaded" RuntimeError raised when every split fails
  for other reasons is no longer unchained — it names and carries the last
  real failure as its cause.
- An intact M4A/MP4 file is no longer reported as malformed by the pipeline
  adapters (#182). Given a path, the transformers ASR pipeline pushes the
  file's bytes to ffmpeg on stdin, and an MP4-family container — the default
  recording format on iOS — cannot be demuxed from a non-seekable pipe: the
  decode returned an empty buffer and transformers raised "Soundfile is
  either not in the correct format or is malformed", advice (check the
  extension, check for corruption) that did not apply, while the same ffmpeg
  decoded the same file completely when given its path. The three pipeline
  adapters (`StandardHuggingFaceASR`, `KhushiDSBengaliModel`, and
  `CustomHuggingFaceModel`'s generic branch) now decode the audio themselves
  by handing ffmpeg the path — seekable, so every container the installed
  binary reads now works — at the pipeline's own sampling rate, and pass the
  raw waveform to the pipeline, so transformers neither re-decodes nor
  resamples. When decoding genuinely fails, the error names ffmpeg, the
  path, and ffmpeg's own stderr instead of claiming the file is malformed.
  The `wav2vec2_*` family and non-pipeline Whisper fine-tunes still decode
  via libsndfile, which reads no M4A/AAC/ALAC; the FAQ pre-run checklist now
  states the split per adapter family instead of a flat format list.
- The gated-model instructions now name all three pyannote repos diarization
  actually needs, and the 403 headline names the repo that was refused
  (#190). The guidance added for #171 listed `pyannote/segmentation-3.0` and
  `pyannote/speaker-diarization-3.1`, but under the `pyannote.audio >= 4.0.7`
  pin from #129 the diarization pipeline also downloads
  `pyannote/speaker-diarization-community-1` — a gated dependency no command
  names — so following the instructions to the letter still ended in a 403.
  The README, FAQ pre-run checklist, `.env.example`, and the runtime
  `GATED_MODEL_HINT` all list it now (with a note that VAD alone needs only
  `segmentation-3.0`). And when the refusal belongs to a dependency rather
  than the requested pipeline, the error's headline no longer repeats the
  already-authorized model id: it reads `access was refused for '<repo>', a
  gated repo this pipeline depends on`, extracting the repo from the 403's
  own URL or prose while ignoring the documentation/settings links
  HuggingFace errors also carry.
- The `multi` path now distinguishes a failed transcription from an empty
  one, the same way `single` does (#181). Adapters return `None` after
  recording the cause, but the manifest loop coerced that to `""` and scored
  it as a real empty hypothesis: an authentication failure produced two
  fully scored rows (CER/WER 1.0), a `Samples: 2` summary with no
  successful/failed split, no error field anywhere in the CSV, and exit 0 —
  while the same key on `single` wrote `Transcription failed: <cause>` and
  exited 1. The multi CSV now carries an `error` column; a transcription
  with a recorded cause becomes a failed row holding that cause instead of a
  scored one, and the rows the pipeline already wrote for preprocessing
  failures now say why (including the pyannote install hint the selector
  recorded but had nowhere to put). Failed rows no longer count as
  processed, so the `.txt` summary gains a `Failed:` line and an all-failure
  run trips the existing zero-rows guard and exits non-zero. A genuinely
  empty prediction with no recorded cause still scores WER/CER 1.0, per the
  benchmark README convention, and the stale-cause clearing protocol from
  #170 applies before every transcription.
- A failed transcription now records its cause in the artifacts the CLI
  points at (#170). Every adapter catches broadly and returns `None` by
  design, so one bad clip cannot abort a long run — but that also meant an
  authentication failure reached the CSV `error` column and
  `scores_<model>.json` as a bare `Empty prediction`, with the real 401
  visible only in the terminal. `ASRModel` now keeps the cause on the
  adapter (`last_transcribe_error`) and every catch-and-return-`None` path
  records it: the three vendor API adapters, the `_retry` decorator's
  exhaustion path, AssemblyAI's errored-transcript object (the SDK reports
  some failures without raising), and all seven HuggingFace adapter failure
  paths. The single-speaker evaluator clears the attribute before each clip,
  so a stale cause is never attributed to the next row, and writes
  `Transcription failed: <cause>` into the row; the scores JSON exports the
  same field from the same results list. A genuinely empty transcription
  with no recorded failure still reads `Empty prediction`, preserving the
  distinction automation needs.
- The type-check gate reaches the same verdict whether or not the optional
  extras are installed (#172). Its two standing `unused-ignore-comment`
  warnings were "unused" only while torch was absent — the `torch.load`
  suppression is live once `[ml]` is installed — so the fix removes the need
  for the suppressions rather than silencing the rule. All three sites are
  the same idiom, a monkeypatch or optional import whose type error exists
  only when the optional package's types are visible: the two
  `pyannote_utils.py` patch sites now use `setattr`, and `llm_metrics.py`
  imports the two `httpx` transport-error classes into a tuple instead of
  rebinding the module to `None`, because `isinstance(exc, ())` is `False`
  on its own — exactly what the discarded `httpx is not None` guard did.
- A missing optional dependency names the extra that ships it (#169). In a
  core-only install, asking for any local model failed with the bare `No
  module named 'torch'`, and the run-level error then closed by suggesting
  `--models` and `--hf-model`, two remedies that fail identically in that
  environment. The registry now translates a `ModuleNotFoundError` raised
  while importing an adapter into the existing `MissingDependencyError`,
  preserving the original error and naming the extra —
  `torch`/`torchaudio`/`transformers`/`sentence_transformers`/`speechmos`/`onnxruntime`
  → `[ml]`, `peft` → `[bengali]`, `pyannote` → `[pyannote]`,
  `openai`/`elevenlabs`/`assemblyai` → `[apis]` — while modules with no
  known extra re-raise unchanged rather than guessing. Both import points in
  `create_model` are covered, so `--models`, `--hf-model`, and the `multi`
  pipeline get the message from one place, and the closing advice now sends
  the user at the per-model reasons first and reserves the flags for
  mistyped ids.
- A hosted-API model with no key is no longer reported as "not found in the
  registry" (#168). `_model_factory` caught every `ValueError` to translate
  the registry's unknown-model error, but the ElevenLabs adapter and the
  AssemblyAI SDK also raise `ValueError` when their key is missing, so their
  actionable "set this environment variable" messages were swallowed and the
  run then listed the exact id the user had passed. The registry now raises
  a dedicated `UnknownModelError` — a `ValueError` subclass, so every
  existing `except ValueError` caller and test keeps working — and
  `_model_factory` catches only that, letting a credential failure reach the
  constructor-failure branch that logs the adapter's own message and
  continues the multi-model run.
- Accepting the gated pyannote models' user conditions is documented as the
  required step it is (#171). A valid `HF_TOKEN` alone returns HuggingFace's
  bare `403 ... not in the authorized list`: the token's account must also
  accept each model's conditions once on its model page, and the one
  sentence in the codebase that mentioned the terms sat on the `not
  PYANNOTE_AVAILABLE` branch, which an affected user never reaches. Every
  place that tells the user to set `HF_TOKEN` — `.env.example`,
  `docs/FAQ.md`, `README.md` — now names the acceptance step with links to
  both model pages and the `403` symptom, and the VAD and diarization
  loaders wrap auth and gating rejections from `from_pretrained` in a
  `RuntimeError` that preserves the original error and appends the same
  guidance, so it lands in the run's log warnings and per-row error results
  where the bare 403 used to. Unrelated failures re-raise unchanged.
- The `multi` subcommand loads `.env` before anything reads credentials
  (#167). It never called `load_env()`, so a credential configured only in
  `.env` was invisible to it — and the failure it produced told the user to
  set the key in `.env`, which is exactly where it already was.
  `run_multispeaker_evaluation()` now mirrors the single-speaker path, and
  one call site covers both entry points (the CLI `multi` subcommand and the
  standalone `psdn_sonar.multispeaker_pipeline` script) because every
  downstream credential read goes through `os.getenv`: the hosted-API
  adapter keys read at `create_model()` time, and the pyannote `HF_TOKEN`
  reads in VAD and diarization, which is also the 401-with-a-valid-token
  failure the issue reported.
- The `[pyannote]` extra installs a pair that can actually import (#129).
  The extra pinned `pyannote.audio` 3.x, which references
  `torchaudio.AudioMetaData` — removed in the torchaudio ≥ 2.9 that `[ml]`
  locks — so `import pyannote.audio` crashed with an `AttributeError` and
  every advertised pyannote feature (`--method pyannote_vad`, diarization)
  was unusable as shipped. The extra now requires `pyannote.audio`
  ≥ 4.0.7, which decodes audio through torchcodec instead of the removed
  torchaudio APIs, and the two `from_pretrained` call sites use the
  renamed `token` argument (4.x removed `use_auth_token`). pyannote 4.x
  needs a system `ffmpeg` (via torchcodec) — the same binary the
  pipeline-based ASR adapters already require — and the README says so.
  With the import fixed, the missing-`HF_TOKEN` path is reachable again
  and produces the actionable gated-model error instead of dying at
  import.
- `EVAL_REPORT.md` no longer claims a public-benchmark comparison the repo
  cannot make (#113). Every benchmark claim in the report is now gated on
  benchmark data actually being present, probed through the same loaders
  the plots read — so the wording and the pictures cannot disagree. In a
  stock install (no benchmark data ships, by deliberate import-gate policy)
  the setup table now says "None included … every number and plot in this
  report describes your dataset only", the performance section is titled
  "Performance Distributions" instead of "Cross-Dataset Comparison", and
  the hardcoded per-language coverage constant is gone — when benchmark
  CSVs are present, the coverage row names exactly the datasets found on
  disk. The hard-negatives plots no longer render fabricated data: the
  hardcoded per-language "benchmark" tables (one commented as placeholder
  values, with invented error bars, and a silent Bengali fallback for
  unknown languages) are deleted, so those plots show only the measured
  user data and title themselves accordingly. Canned "Key Insights"
  verdicts ("your dataset shows strong diversity metrics compared to
  public benchmarks") are replaced with pointers that assert nothing the
  report does not compute. The `normalize_bengali_for_wer` docstring no
  longer promises comparability "against published results", and the FAQ
  no longer directs users to baselines in `psdn_sonar/benchmarks/`, which
  does not ship.
- CI now exercises what the README advertises (#114). A green run
  previously validated Linux only, and no job installed the `[ml]` extra,
  so the 8 HuggingFace adapter tests skipped on every PR — no ASR model
  adapter was exercised anywhere in the pipeline. One new job closes both
  gaps: macOS arm64 (Python 3.12) runs the full suite with `[ml]`
  installed, with a preflight import check so the suite can never go
  green with the adapter tests silently skipped. The `[ml]` install lives
  on macOS deliberately — its torch wheels are small CPU-only builds,
  while the lockfile resolves Linux torch to the CUDA-bundled stack
  (several GB), too heavy for PR CI when the adapter tests are fully
  mocked. The README's supported-environments note now matches: macOS is
  CI-validated, Windows remains expected-to-work but unvalidated.
- Swept the six places where docstrings, config, and packaging did not
  match the code (#115). The shipped loanword caches are now actually
  CI-validated: `tests/test_loanword_cache_integrity.py` — the file the
  validator's docstring had always pointed at — exists and hard-asserts
  all six integrity checks on the bn/hi/ko caches (a polluted entry
  silently shifts WER, since replacement applies to both reference and
  hypothesis). The `BengaliProcessor` docstring now states the class is
  not on any production path and that wiring it into scoring would
  require a contract bump, instead of claiming "validation, dataset
  preparation" uses it. The two unreachable Tugstugi adapter classes are
  deleted (the registry routes those model ids through
  `StandardHuggingFaceASR`). The `llm_metrics` module docstring now names
  stable `gemini-2.5-pro` as the default judge — matching
  `DEFAULT_MODEL` — with `gemini-3.1-pro-preview` framed as the opt-in it
  is. A checked-in `.python-version` pins `uv` to CPython 3.12 so
  `make setup` cannot build an interpreter outside the documented
  3.10–3.12 range (CI passes `--python` explicitly and is unaffected).
  And the top-level `data/` directory the documented discover workflow
  writes into is now gitignored (root-anchored; the `psdn_sonar/data/`
  package is untouched) plus excluded from sdists, closing the path to a
  455 MB local `uv build` and accidental audio commits.
- Invisible characters no longer score as transcription errors (#140). The
  punctuation strip filters Unicode categories `P*`/`S*` only, so format
  characters (category `Cf`) survived normalization in every language — a
  zero-width space in the reference scored as a substitution against a
  byte-identical prediction — and English applied no Unicode normalization
  at all, so the NFC and NFD encodings of the same word (`café` composed vs
  `e` + combining acute) counted as different words. Every normalization
  path now starts with one shared fold: ZWSP/ZWNBSP become a space (they
  mark word boundaries, so an invisible separator cannot glue words
  together), all other format characters (ZWJ/ZWNJ, soft hyphen,
  directional marks) are removed, and NFC composition applies. In the
  Bengali canonical pipeline the fold runs before loanword replacement, so
  a zero-width character inside a Latin token cannot defeat the cache
  lookup. This changes scoring on affected corpora, so every WER
  normalization contract is bumped (`en`/`hi`/`ko` v3, `bn` v4) and
  recorded per run in `scores.json` lineage as before.
- Single-speaker runs now write the normalized text WER was computed over
  (#143). `asr_detailed_<model>.csv` (both `single` and `custom`) gains
  `normalized_reference` / `normalized_hypothesis` columns — the exact
  strings CER/WER scored after language normalization. Previously the single
  path wrote no normalized text at all, so a reference whose score was
  poisoned by an invisible character (e.g. a zero-width space surviving
  normalization) was undiagnosable from the artifact, while the multi path
  did expose it under `transcription_norm` / `asr_transcription_norm_*`.
  Rows where CER/WER were uncomputable also carry the pair, since an empty
  `normalized_reference` is itself the diagnosis. The
  `normalize_bengali_for_wer` docstring, which promised these column names
  while no path produced them, now states the actual per-path columns.
- ASR adapters now use MPS on Apple Silicon (#111). Every HuggingFace adapter
  selected its device with `torch.cuda.is_available()` only, so on Apple
  Silicon diarization ran on the GPU while ASR inference silently fell back
  to CPU — measured ~16x slower for Whisper-class models — and the `device`
  field in `scores.json`, which does report `mps`, misstated where inference
  actually ran. All adapters now share one resolver (CUDA, then MPS, then
  CPU, matching the diarization pipeline); explicit `device=` arguments in
  any accepted form (int, string, `torch.device`) pass through unchanged,
  and fp16 remains CUDA-only.
- The catalog's acquisition-vs-redistribution posture is now documented and
  surfaced at the point of download (#116). All eight benchmark catalog
  entries carry `review.decision: pending`, seven of them enabled and
  downloadable through `psdn-sonar discover`, which read as a contradiction
  of the import gate's "confirmed redistribution rights" bar. The position
  is now written into `docs/import-gate.md` ("Acquisition vs.
  redistribution") and the catalog file header: the catalog stores pinned
  pointers, never dataset content; `discover` performs user-initiated
  acquisition from the upstream source under the per-entry license; and
  `review.decision` gates project redistribution/publishing, which
  `catalog.identity(publishable=True)` already refuses for non-approved
  data. `discover` now prints a data-rights note listing each discovered
  dataset's upstream license and review state, so a pending review is
  visible to the user acquiring the data. New tests pin the posture
  invariants (license evidence on every entry, `prohibited` implies
  disabled, enabled-but-pending refuses the publish gate). Closing the
  pending reviews themselves — approver, date, evidence URL, verified
  fingerprints — remains a maintainer/rights action recorded in the
  catalog, not a code change.
- A wrong-but-supported `--language` no longer produces a silent,
  healthy-looking scorecard (#148). `--language ko` on English data used to
  run with zero warnings and emit plausible scores with self-consistent
  provenance — indistinguishable downstream from a correct run, even though
  the language selects the normalization branch and every WER/CER in the
  run was computed with the wrong rules. The single-speaker evaluator now
  compares the dominant Unicode script of the reference transcriptions
  against the script the selected language is written in (bn→Bengali,
  hi→Devanagari, ko→Hangul, en→Latin; `psdn_sonar.language.script_check`).
  On a clear majority mismatch it logs a WARNING naming both scripts and
  the likely correct code, and records the same text in a new top-level
  `warnings` list in `scores.json`, so the artifact itself carries the
  signal. Code-switched references where the expected script keeps the
  majority do not trip the check; same-script confusions (en vs sw) are out
  of its reach by design. Runs stay warnings-only — nothing is rejected.
- `scores.json` provenance no longer misattributes the code and now records
  the score-changing inputs (#110). `git_sha` used to be resolved with
  `git rev-parse HEAD` in the caller's working directory, so a run started
  from inside an unrelated repository recorded that repository's commit — a
  well-formed 40-char SHA pointing at a different codebase. It is now
  resolved against the package's own directory and recorded only when the
  package file is tracked by that repository (so a venv nested inside an
  unrelated repo records `unknown` rather than the host repo's HEAD, as do
  wheel/pip installs; the `SONAR_GIT_SHA` override — now documented in
  `.env.example` and the benchmark README — stamps a known commit, e.g. in
  CI). `SubmissionConfig.from_env()` additionally records the POSEIDON
  weights and semantic-similarity model actually in effect (including
  `POSEIDON_*_WEIGHT` / `SIMILARITY_MODEL` env overrides) plus
  `os_platform`, `python_version`, and `device`, so two runs whose scores
  differ across environments no longer have identical-looking provenance.
  The new fields default to null; pre-existing artifacts still validate.
- `discover` no longer prints a traceback for an unwritable output
  directory (#149): the run already failed correctly (exit 1, no download,
  two actionable ERROR lines), but a full traceback appeared between them
  whose exception chain named a `FileNotFoundError` before the real
  `PermissionError` — an artifact of `pathlib.mkdir(parents=True)`
  internals that led the reader to the wrong diagnosis. Dataset-preparation
  `OSError`s (unwritable `--output`, disk full, network) are now reported
  as the single clean ERROR line the other user-error paths emit; the
  message carries the exception that actually propagated, i.e. the real
  cause. Unexpected non-OS errors keep their traceback.
- One unconstructible model no longer aborts a multi-model run, and the
  missing `[bengali]` extra is named instead of a bare traceback (#108):
  the documented `--language bn` default run downloaded ~3 GB for its
  first model, then died at `khushids_bengali` with
  `ModuleNotFoundError: No module named 'peft'` — and because the model
  constructor ran outside any try/except, the whole 9-model loop ended
  and models already evaluated in the run lost their output.
  `khushids_bengali` now raises `MissingDependencyError` (new, in
  `psdn_sonar.models.base`) naming `peft` and
  `pip install "psdn-sonar[bengali]"` before any download, and
  `run_evaluation` isolates constructor failures per model: the failing
  model is skipped with its reason logged and the remaining models still
  evaluate (results are written per model, so completed work survives).
  A run where every model fails to construct still errors loudly with
  exit 1. README and `docs/USAGE.md` now document that the Bengali
  defaults include one model requiring the `[bengali]` extra.
- All scoring paths now share one missing-value convention, and semantic
  similarity is reported on one range everywhere (#107). Previously three
  paths handled an uncomputable metric three contradictory ways: the
  single-speaker evaluator scored it as best case (CER/WER 0.0, deflating
  run averages), `PoseidonScorer` substituted worst case (WER/CER 1.0,
  similarity 0.0, inflating them), and `significant_wer_rate` excluded it —
  so the same batch yielded systematically shifted aggregates depending on
  the entry point. The convention is now: a metric that cannot be computed
  is `null`/`None`/`NaN`, the row is counted as failed with the reason in
  its `error` field (the transcription is preserved), and aggregates are
  computed only over present values. `ensure_poseidon_score` (reporting
  backfill) likewise leaves `NaN` instead of fabricating worst-case scores,
  and no longer invents a POSEIDON score for runs that never computed
  semantic similarity. Separately, semantic similarity was clamped to
  `[0, 1]` inside the POSEIDON formula but stored and averaged raw, so
  `semantic_similarity_mean` could go negative while `poseidon_score_mean`
  could not; cosine similarity is now clamped to `[0, 1]` once at the point
  of computation, so the CSV, the means, POSEIDON, and the leaderboard all
  report the same value. Both conventions are documented in
  `psdn_sonar/benchmark/README.md` ("Missing values and metric ranges").
- The `psdn-sonar` exit code is now stable across repeat runs (#139): a
  run with zero successful samples intermittently died with SIGABRT
  (exit 134) instead of the exit 1 it had already reported, because a
  torch-family native extension aborted in C++ static destructors during
  interpreter teardown (`recursive_mutex lock failed`) — after all
  results, scores, and the error message were written. The console
  script now enters through `psdn_sonar.cli:entrypoint`, which resolves
  the exit code, flushes logging and the std streams, and leaves via
  `os._exit`, skipping interpreter finalization entirely. Codes are now
  deterministic: 0 success, 1 evaluation/data failure (also unexpected
  exceptions, which still print their traceback), 2 argparse usage
  error, 130 Ctrl-C. In-process callers of `cli.main()` keep normal
  `SystemExit` behavior.
- Missing `ffmpeg` now fails once at model load instead of once per
  utterance (#109): the adapters that hand audio file paths to the
  `transformers` ASR pipeline — `StandardHuggingFaceASR` (both English
  default models), `KhushiDSBengaliModel`, and `CustomHuggingFaceModel`'s
  generic-pipeline branch — need the `ffmpeg` binary to decode **any**
  path they are given, WAV included. Without it a run used to emit one
  identical `Transcription failed ... ffmpeg was not found` error per
  utterance and still print its normal summary. These adapters now
  preflight for `ffmpeg` before downloading the checkpoint and raise
  `MissingFfmpegError` (new, in `psdn_sonar.models.base`) naming the
  binary, install commands, and the adapters that decode audio
  themselves; `psdn-sonar single` reports it as a clean error and exits
  1. The README claim that "WAV evaluation works without ffmpeg" was
  wrong for these adapters and has been corrected.
- TSV parsing no longer truncates references or mis-reports a BOM
  (#141): a data row carrying more fields than the header (a literal
  tab inside the transcription) used to have everything after the tab
  silently discarded and the truncated reference scored (exit 0, no
  warning); such rows are now counted as failed with a warning naming
  the line and field counts, per the issue-#102 failed-not-dropped
  pattern. Evaluation TSVs are also read as ``utf-8-sig``, so the UTF-8
  BOM Excel prepends to exports is stripped instead of corrupting the
  first column name and raising ``TSV missing required columns:
  audio_path`` for a column that is present.
- Bengali suffix splitting no longer cuts whole words in two (#142):
  `_split_suffixes` matched on trailing characters alone, so ordinary
  words ending in a suffix-lookalike were split (`মাটি` → `মা টি`,
  `ছেলে` → `ছেল এ`) and `ঘণ্টা` was cut inside its conjunct, leaving a
  virama-terminated fragment that is not a well-formed Bengali word.
  Applied to reference and hypothesis alike this did not create word
  errors by itself, but it systematically inflated the Bengali token
  count — the WER denominator. A split is now taken only when the stem
  could stand alone (at least two grapheme clusters, never ending in a
  virama) and the token is not in a small protected whole-word lexicon
  covering cases structure cannot decide (`ছেলে` and `দেশ`+`এ` have
  identical shapes). Genuine suffixes still split (`প্যাকেটটা` →
  `প্যাকেট টা`, `বইগুলো` → `বই গুলো`), and locatives like `হাতে` now
  split at the true morpheme boundary (`হাত এ`) instead of the nonsense
  `হা তে`. The Bengali normalization contract recorded in `scores.json`
  is bumped to `bn:v3` — v2 and v3 numbers must not be compared as
  like-for-like.
- Thousands separators no longer split number verbalization (#135):
  `1,000 dollars` used to normalize to `one000 dollars` for English,
  Hindi and Korean — the digit-run regex saw `1` and `000` as separate
  runs, the second hit the leading-zero skip, and the later punctuation
  strip glued them into a token that can never match, silently inflating
  WER on corpora with separated numerals. The shared `verbalize_digits`
  now strips separators inside well-formed grouped numbers (Western
  `1,234,567`, Indian `12,34,567`, and space/thin-space grouping) before
  digit-run extraction — the handling the canonical Bengali pipeline
  already had. Enumerations (`1,2,3`) and adjacent independent runs
  (`2020 100`) are not merged; joined numbers longer than 4 digits stay
  as digits under the existing phone/ID skip. `BengaliProcessor`'s own
  digit path received the same comma strip (`১,০০০` now verbalizes like
  `১০০০`). Because absolute WER changes, the English, Hindi and Korean
  normalization contracts recorded in `scores.json` are bumped to
  `en:v2`/`hi:v2`/`ko:v2` — v1 and v2 numbers must not be compared as
  like-for-like.
- Bengali now verbalizes semantic symbols like the other three languages
  (#136): Bengali was the only supported language without a symbol map,
  so `%` survived normalization as a literal — `৫০%` normalized to
  `50 %` and could never match a hypothesis written `৫০ শতাংশ`, and the
  leftover symbol's spacing depended on whether the `[bengali]` extra
  (`bnlp`) was installed, making the same text score differently across
  otherwise identical installs. `BENGALI_SYMBOL_MAP` (key-for-key
  parallel with the English/Hindi/Korean maps) is applied in both the
  canonical WER pipeline and `BengaliProcessor`, before digit
  verbalization and punctuation handling. Because this changes absolute
  Bengali WER, the Bengali normalization contract recorded in
  `scores.json` is bumped to `bn:v2` — v1 and v2 numbers must not be
  compared as like-for-like.
- A TSV `audio_path` with `../` can no longer escape the dataset
  directory (#127): the boundary guard in the single-speaker evaluator
  existed but was gated behind a parameter that defaulted off and had no
  CLI exposure, so a manifest received from someone else could read
  audio from anywhere on disk. Relative paths that resolve outside the
  TSV's directory are now always rejected with a clear error before any
  model loads. Absolute paths remain allowed by default (they are
  explicit in the TSV and `discover` output writes them); the new
  `psdn-sonar single --strict-audio-paths` flag additionally rejects
  absolute paths and requires every path to be an existing regular file.
- Multi-speaker assignment no longer treats a perfect score as missing
  (#106): the `dual_assignment_score` selection heuristic defaulted
  missing metrics with `or`, which also fires on a legitimate `0.0` —
  a perfect transcription (CER/WER 0.0) was scored as worst case, so a
  swapped speaker-to-reference pairing with real errors but slightly
  higher similarity could win and corrupt both speakers' reported
  WER/CER. Missing metrics now default via explicit `None` checks,
  matching the equivalent computation in `core.py`.
- Fully silent audio no longer passes the quality gate (#105):
  `calculate_silence_ratio` measures frames relative to the file's own
  loudest frame, so uniformly quiet files scored `0.0` — identical to
  all-speech. Files whose loudest RMS frame is below an absolute silence
  floor (`SONAR_SILENCE_FLOOR_AMP`, default 1e-3 ≈ -60 dBFS) now score
  `1.0` and trip the `high_silence` warning; mixed-content behavior is
  unchanged. `calculate_snr` returns `None` instead of `inf` for
  signal-less audio (SNR is undefined; the old `inf` leaked into CSVs,
  gave silent files an SNR tier of "High", and poisoned plot columns) and
  caps noise-free audio at 100 dB.
- `psdn-sonar discover` no longer reports failure as success (#104):
  `--datasets` entries are validated per name with distinct reasons — unknown
  name (with the discoverable list), catalogued-but-disabled
  (`common_voice`), catalogued non-HuggingFace source (the OpenSLR Bengali
  corpora), or catalogued-but-not-wired (`multilingual_librispeech`) — and
  exit 1 instead of warning and exiting 0. A valid filter that matches
  nothing for the language also exits 1, and the error blames the filter
  (naming each dataset's supported languages) rather than the language. The
  summary now prints a catalog scope note so a one-row listing is no longer
  presented as the complete catalog.
- `psdn-sonar single`/`multi` now validate `--language` before scoring (#103):
  unrecognized codes (e.g. `xx`) exit 1 with an actionable error instead of
  silently scoring with the fallback normalizer, recognized ISO codes without
  a dedicated normalizer (e.g. `pt`) log an explicit fallback warning, and
  long names / mixed case (`Bengali`, `KO`) are canonicalized to the ISO code
  so they select the intended normalizer. The CLI epilogue, examples, and QA
  guide no longer pair Bengali with generic `openai/whisper-small`, which
  hallucinates non-Bengali script (WER ≈ 1.04); they recommend the registered
  `wav2vec2_bengali` instead.
- Runs that evaluate zero samples no longer exit 0 looking like clean runs
  (#102): unknown model names raise a `ValueError` listing the registered ids,
  a run with zero successful samples raises `NoSamplesEvaluatedError` after
  writing artifacts, `wer_mean`/`cer_mean` are `null` (not `0.0`) when
  `successful == 0`, TSV rows with missing/blank fields are counted in
  `failed` with per-row error entries instead of being silently dropped,
  per-utterance CSVs always carry a header, a multi run that processes zero
  clips raises instead of logging completion, and `--method pyannote_vad`
  without the `[pyannote]` extra fails fast with the install hint.

- `calculate_poseidon_score` now raises an actionable `TypeError` naming the
  missing `psdn-sonar[ml]` extra when given the `None` that
  `compute_semantic_similarity` returns without sentence-transformers, instead
  of an opaque comparison error (#101). README and USAGE now flag that the
  `[dev]` install does not include `[ml]`.
- `psdn-sonar multi --hf-model` now loads the HuggingFace checkpoint instead of
  calling `create_model(None)`. `--language` is passed through to scoring.
- Language-default `single` runs skip hosted API models unless their keys are
  set, so `psdn-sonar single --language en` no longer tries ElevenLabs/OpenAI/
  AssemblyAI on a fresh install.
- Omitting `--language` logs a warning that scoring defaults to Bengali.
- Multi-speaker FAQ/examples now use the runtime manifest schema
  (`audio_filepaths`, JSON `transcript_filepath`) and ship a tiny fixture.
- `docs/USAGE.md` section 1 now scores through
  `UtteranceEvaluator.score_single_variant` — the evaluation path, which
  normalizes before computing CER/WER — and labels `calculate_cer_wer` as a
  raw-text primitive, so the first documented number matches what an
  evaluation run reports (#100).

## [0.1.0] - 2026-08-17

### Added

- Multi-language text normalization pipeline (Bengali, Hindi, English, Korean)
  with loanword replacement, symbol/number verbalization, and script checks.
- Config-driven ASR backends: HuggingFace models, hosted APIs (OpenAI Whisper,
  AssemblyAI, ElevenLabs), and a central model registry.
- Evaluation metrics: CER, WER, semantic similarity, and the composite
  PSDN score with configurable weights.
- Audio-quality metrics: SNR, clipping ratio, silence ratio, and optional
  reference-free MOS scorers (DNSMOS, UTMOS, SQUIM).
- Single-speaker, multi-speaker (manifest-driven), and custom (YAML-configured)
  evaluation pipelines, exposed via the `psdn-sonar` CLI.
- Dataset discovery and preparation for public corpora (`psdn-sonar discover`).
- Reporting suite: markdown report generator plus lexical-diversity,
  cross-dataset, demographic, audio-quality, and latency plots.
- Precomputed public benchmark data and aggregation scripts
  (`precompute_benchmarks.py`, `extract_benchmarks.py`, `build_macro_summary.py`).
- Data utilities: YAML-driven cloud sync, Common Voice and Zeroth-Korean
  TSV converters, transcript-JSON data prep, SNR-vs-WER analysis.
- Runnable examples for every pipeline under `examples/`.

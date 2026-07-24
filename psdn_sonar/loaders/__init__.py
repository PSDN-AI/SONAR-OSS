"""Dataset loaders for public ASR corpora."""

from psdn_sonar.loaders.base import DatasetLoader, OpenSLRBaseLoader, SingleSpeakerLoaderBase
from psdn_sonar.loaders.common_voice import CommonVoiceLoader
from psdn_sonar.loaders.fleurs import FleursLoader
from psdn_sonar.loaders.openslr import OpenSLR37BDLoader, OpenSLR53Loader, OpenSLRLineIndexLoader
from psdn_sonar.loaders.resolution import resolve_dataset_dir

__all__ = [
    "DatasetLoader",
    "SingleSpeakerLoaderBase",
    "OpenSLRBaseLoader",
    "CommonVoiceLoader",
    "FleursLoader",
    "OpenSLR37BDLoader",
    "OpenSLR53Loader",
    "OpenSLRLineIndexLoader",
    "resolve_dataset_dir",
]

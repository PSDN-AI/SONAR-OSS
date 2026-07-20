"""Configuration loader with OmegaConf support."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and merging."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.

        Args:
            config_dir: Path to config directory (defaults to repo root/conf)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "conf"

        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        logger.info(f"Config directory: {self.config_dir}")

    def load(
        self,
        config_name: str = "config",
        overrides: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        backend: Optional[str] = None,
        validation: Optional[str] = None,
    ) -> Any:
        """
        Load configuration with optional overrides.

        Args:
            config_name: Name of main config file (without .yaml)
            overrides: Dictionary of config overrides
            language: Language config to use (e.g., 'bn', 'ko')
            backend: Backend config to use (e.g., 'huggingface', 'whisper')
            validation: Validation config to use (e.g., 'lenient', 'strict')

        Returns:
            Configuration object
        """
        try:
            from omegaconf import OmegaConf
        except ImportError:
            logger.warning("OmegaConf not installed, falling back to simple YAML loader")
            return self._load_simple(config_name, overrides, language, backend, validation)

        # Load main config
        main_config_path = self.config_dir / f"{config_name}.yaml"
        if not main_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {main_config_path}")

        cfg = OmegaConf.load(main_config_path)

        # Load language config
        if language:
            lang_config_path = self.config_dir / "language" / f"{language}.yaml"
            if lang_config_path.exists():
                lang_cfg = OmegaConf.load(lang_config_path)
                cfg = OmegaConf.merge(cfg, lang_cfg)
            else:
                logger.warning(f"Language config not found: {lang_config_path}")

        # Load backend config
        if backend:
            backend_config_path = self.config_dir / "backend" / f"{backend}.yaml"
            if backend_config_path.exists():
                backend_cfg = OmegaConf.load(backend_config_path)
                cfg = OmegaConf.merge(cfg, backend_cfg)
            else:
                logger.warning(f"Backend config not found: {backend_config_path}")

        # Load validation config
        if validation:
            val_config_path = self.config_dir / "validation" / f"{validation}.yaml"
            if val_config_path.exists():
                val_cfg = OmegaConf.load(val_config_path)
                cfg = OmegaConf.merge(cfg, val_cfg)
            else:
                logger.warning(f"Validation config not found: {val_config_path}")

        # Apply overrides
        if overrides:
            override_cfg = OmegaConf.create(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)

        # Resolve interpolations
        OmegaConf.resolve(cfg)

        return cfg

    def _load_simple(
        self,
        config_name: str,
        overrides: Optional[Dict[str, Any]],
        language: Optional[str],
        backend: Optional[str],
        validation: Optional[str],
    ) -> Dict[str, Any]:
        """Simple YAML loader fallback (without OmegaConf)."""
        try:
            import yaml
        except ImportError:
            raise ImportError("Neither OmegaConf nor PyYAML is installed")

        # Load main config
        main_config_path = self.config_dir / f"{config_name}.yaml"
        with open(main_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Load and merge language config
        if language:
            lang_config_path = self.config_dir / "language" / f"{language}.yaml"
            if lang_config_path.exists():
                with open(lang_config_path, encoding="utf-8") as f:
                    lang_cfg = yaml.safe_load(f)
                    cfg.update(lang_cfg)

        # Load and merge backend config
        if backend:
            backend_config_path = self.config_dir / "backend" / f"{backend}.yaml"
            if backend_config_path.exists():
                with open(backend_config_path, encoding="utf-8") as f:
                    backend_cfg = yaml.safe_load(f)
                    cfg.update(backend_cfg)

        # Load and merge validation config
        if validation:
            val_config_path = self.config_dir / "validation" / f"{validation}.yaml"
            if val_config_path.exists():
                with open(val_config_path, encoding="utf-8") as f:
                    val_cfg = yaml.safe_load(f)
                    cfg.update(val_cfg)

        # Apply overrides
        if overrides:
            cfg.update(overrides)

        # Convert to namespace for dot notation access
        return self._dict_to_namespace(cfg)

    def _dict_to_namespace(self, d: Dict) -> Any:
        """Convert dict to namespace for dot notation access."""
        from types import SimpleNamespace

        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_namespace(item) for item in d]
        else:
            return d


def load_config(
    config_dir: Optional[Path] = None,
    language: str = "bn",
    backend: str = "huggingface",
    validation: str = "lenient",
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function to load configuration.

    Args:
        config_dir: Path to config directory
        language: Language code (e.g., 'bn', 'ko')
        backend: Backend name (e.g., 'huggingface', 'whisper')
        validation: Validation mode (e.g., 'lenient', 'strict')
        overrides: Dictionary of config overrides

    Returns:
        Configuration object

    Example:
        >>> config = load_config(language='ko', backend='huggingface')
        >>> print(config.language.name)
        Korean
    """
    manager = ConfigManager(config_dir)
    return manager.load(
        config_name="config", overrides=overrides, language=language, backend=backend, validation=validation
    )


_DEFAULT_RUN_SEED = 42


def get_run_seed(config: Any = None) -> int:
    """Return ``run.seed`` from the main OmegaConf config (default 42).

    Single entry point for reproducible sampling in evaluators and reporting.
    ``psdn_sonar.config`` remains env-based POSEIDON weights; preprocessing keeps
    its own YAML loader for multi-speaker trim settings.
    """
    if config is None:
        config = load_config()
    try:
        seed = config.run.seed
    except (AttributeError, KeyError, TypeError):
        return _DEFAULT_RUN_SEED
    try:
        return int(seed)
    except (TypeError, ValueError):
        return _DEFAULT_RUN_SEED

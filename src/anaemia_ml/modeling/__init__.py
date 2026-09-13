"""Model construction, registry, and leakage-safe runner utilities."""

from anaemia_ml.modeling.imbalance import (
    DEFAULT_CLASSES,
    FoldSafeClassWeightClassifier,
    ImbalanceError,
    ImbalanceSettings,
    SMOTENCSpec,
    balanced_class_weights,
    balanced_sample_weights,
    build_primary_imbalance_classifier,
    imbalance_settings,
    make_smotenc_spec,
    smotenc_categorical_indices,
    smotenc_sampling_targets,
)
from anaemia_ml.modeling.registry import (
    ModelDependencyError,
    ModelRegistryError,
    ModelSpec,
    build_estimator,
    model_spec,
    registered_models,
)
from anaemia_ml.modeling.runner import (
    DEFAULT_MODEL_NAMES,
    ModelRun,
    RunnerError,
    compare_models,
    fit_evaluate_model,
)

__all__ = [
    "DEFAULT_CLASSES",
    "DEFAULT_MODEL_NAMES",
    "FoldSafeClassWeightClassifier",
    "ImbalanceError",
    "ImbalanceSettings",
    "ModelDependencyError",
    "ModelRegistryError",
    "ModelRun",
    "ModelSpec",
    "RunnerError",
    "SMOTENCSpec",
    "balanced_class_weights",
    "balanced_sample_weights",
    "build_estimator",
    "build_primary_imbalance_classifier",
    "compare_models",
    "fit_evaluate_model",
    "imbalance_settings",
    "make_smotenc_spec",
    "model_spec",
    "registered_models",
    "smotenc_categorical_indices",
    "smotenc_sampling_targets",
]

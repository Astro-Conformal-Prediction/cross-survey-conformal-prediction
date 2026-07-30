"""Shared constants for the cross-survey embedding alignment project."""

FOUNDATION_MODELS = [
    "astropt_15m", "astropt_850m", "astropt_95m",
    "convnext_base", "convnext_large", "convnext_nano", "convnext_tiny",
    "dino_base", "dino_giant", "dino_large", "dino_small",
    "ijepa_giant", "ijepa_huge",
    "vit_base", "vit_huge", "vit_large",
]

SURVEY_CONFIGS = {
    "legacy": {
        "hf_repo": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "source_col": 2,
        "target_col": 1,
        "streaming": True,
        "requires_token": True,
    },
    "sdss": {
        "hf_repo": "astro-conformal-project/split_sdss_hsc_embeddings",
        "source_col": 1,
        "target_col": 2,
        "streaming": True,
        "requires_token": True,
    },
    "jwst": {
        "hf_repo": "astro-conformal-project/split_jwst_hsc_embeddings",
        "source_col": 1,
        "target_col": 2,
        "streaming": True,
        "requires_token": True,
    },
}

"""Smoke-check the pinned local WOFOST maize parameters."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Load and print the selected maize variety without refreshing the cache."""
    from pcse.input import YAMLCropDataProvider
    from pcse.models import Wofost81_NWLP_CWB_CNB

    crop_parameter_dir = PROJECT_ROOT / "data" / "crop_parameters" / "wofost81"
    cropdata = YAMLCropDataProvider(
        model=Wofost81_NWLP_CWB_CNB,
        fpath=str(crop_parameter_dir),
        force_reload=False,
    )
    cropdata.set_active_crop("maize", "Grain_maize_201")
    print(cropdata)


if __name__ == "__main__":
    main()

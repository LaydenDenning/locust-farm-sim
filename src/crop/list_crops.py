from pathlib import Path

from pcse.input import YAMLCropDataProvider
from pcse.models import Wofost81_PP


PROJECT_ROOT = Path(__file__).resolve().parents[2]

crop_parameter_dir = (
    PROJECT_ROOT
    / "data"
    / "crop_parameters"
    / "wofost81"
)

cropdata = YAMLCropDataProvider(
    model=Wofost81_PP,
    fpath=str(crop_parameter_dir),
    force_reload=True
)

cropdata.set_active_crop(
    "maize",
    "Grain_maize_201"
)

print(cropdata)

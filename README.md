# Crop Monitoring Drone Farm Simulation

This repository is building a reproducible crop-simulation baseline for later
comparison of conventional scouting and UAV-assisted crop monitoring.

Phase 1 is deliberately narrower than that eventual comparison: it creates a
spatial ground-truth farm with coherent synthetic crop trajectories. It does
not yet simulate a drone, scouting observations, management decisions,
economics, or evidence of drone value.

## Phase 1 model

The field is an 800 m by 800 m square divided into 25 contiguous 160 m by
160 m zones. Each zone runs a fresh PCSE 6.0.13
`Wofost81_NWLP_CWB_CNB` engine using:

- WOFOST 8.1 `maize` / `Grain_maize_201` crop parameters;
- the classic water balance and classic soil-nitrogen balance;
- a base sowing date of 2022-05-01 and a 200-day maximum duration;
- 420 ppm atmospheric CO2;
- fixed 2022 weather for Ames, Iowa (42.03 N, 93.63 W); and
- no irrigation, fertilizer events, or injected stress events.

The tracked `maize.yaml` originated in the
[WOFOST crop-parameter repository](https://github.com/ajwdewit/WOFOST_crop_parameters)
and declares parameter-file version 1.0.0 (metadata date 2022-02-13). The
local `crops.yaml` pins maize as the only available crop so normal runs do not
depend on a changing remote parameter repository.

The official WOFOST 8.1 crop catalog excludes maize because the legacy
temperate-maize set lacks 13 inputs required by the final WOFOST 8.1 model
interface. The local `temperate_maize` block therefore includes a clearly
marked compatibility section: `AMAX_REF` is derived from its existing
`AMAXTB`; two assimilation coefficients use the only values present in the
file; the remaining nitrogen-stress and reallocation defaults match the same
file's tropical-maize block. Reallocation is disabled before maturity. These
values make `Grain_maize_201` operational, but they are synthetic and have not
been calibrated for central Iowa.

The three soil profiles are synthetic low-, reference-, and high-water-holding
profiles. They are not USDA texture classifications or field-calibrated soils.
Zone differences are explicit in `data/phase1/zones.csv`: soil profile,
planting offset, initial available nitrogen, stand density, and a central
slow-drainage flag. Stand density is represented by the establishment proxy
`TDWI = 50 * stand_density / 8` kg/ha.

## Setup on this Windows machine

Create the pinned environment without changing the PowerShell profile:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe env create -f environment.yml
```

PCSE creates runtime settings, logs, caches, and `pcse.db` under
`C:\Users\layde\.pcse` on first import. Confirm the installed versions and run
PCSE's built-in tests with:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe run -n py3_pcse python -c "import sys, pcse; print(sys.version); print(pcse.__version__); pcse.test()"
```

## Run Phase 1

From the repository root:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe run -n py3_pcse python -m src.simulation.run_phase1 --config config/phase1.yaml
```

The command validates all configuration, zone geometry, crop inputs, and
weather coverage before starting the first zone. It runs the 25 zones
sequentially with fresh mutable model state and writes results only after every
zone succeeds. Existing result files are protected; use `--overwrite` only
when replacing them is intentional:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe run -n py3_pcse python -m src.simulation.run_phase1 --config config/phase1.yaml --overwrite
```

## Inputs and units

| Input | Unit or meaning |
|---|---|
| Zone geometry | metres |
| `SMW`, `SMFCF`, `SM0` | volumetric soil-water fraction, cm3/cm3 |
| `RDMSOL` | cm |
| `K0`, `SOPE`, `KSUB` | cm/day |
| `WAV` | cm water in the potentially rooted zone |
| Initial available N (`NAVAILI`) | kg N/ha |
| Stand density | plants/m2 |
| `TDWI` | kg dry matter/ha |
| `CO2` | ppm |
| Weather `IRRAD` | kJ/m2/day in the CSV; converted by PCSE |
| Weather `TMIN`, `TMAX` | degrees C |
| Weather `VAP` | kPa |
| Weather `WIND` | m/s |
| Weather `RAIN` | mm/day |
| Weather `SNOWDEPTH` | cm |

The weather file header records its NASA POWER provenance, coordinates,
retrieval date, and units. Normal simulations use only this local CSV through
`CSVWeatherDataProvider`; they do not retrieve weather from the internet. The
small provenance utility `scripts/fetch_phase1_weather.py` records the exact
NASA POWER variables and unit conversions used to create the pinned file; it
is not called by a normal simulation.

## Outputs

Generated files are written to the ignored `outputs/phase1/` directory:

- `daily_truth.csv`: one unique `(zone_id, date)` row with zone metadata,
  `crop_active`, crop development and organ biomass, soil moisture, available
  nitrogen, nitrogen nutrition index, organ nitrogen amounts, and cumulative
  crop N uptake;
- `zone_summary.csv`: crop-calendar dates, maximum LAI, terminal storage-organ
  biomass (`TWSO`), total above-ground biomass, and total N uptake by zone;
- `lai_trajectories.png`;
- `soil_moisture_trajectories.png`;
- `nitrogen_trajectories.png`; and
- `final_yield_heatmap.png`.

`TWSO` is a dry storage-organ biomass proxy. Phase 1 does not convert it to
commercial grain yield, revenue, or avoided loss.

PCSE 6.0.13's WOFOST 8.1 nitrogen module does not publish `NNI` directly. The
export derives it from simulated leaf/stem biomass and N amounts using the same
critical-versus-residual concentration formula and 0.001-to-1 bounds used by
PCSE's standard NPK-stress implementation.

## Tests

Run the standard-library test suite with:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe run -n py3_pcse python -m unittest discover -s tests -v
```

The tests cover input validation, field coverage, a reference-zone model run,
25-zone invariants, controlled input comparisons, deterministic reruns, and
CSV round trips.

## Interpretation boundary

Phase 1 outputs are plausible, deterministic synthetic spatial trajectories.
They are a ground-truth test bed, not field-calibrated agronomic predictions
and not proof that a monitoring drone creates value. Stronger agronomic or
economic claims require field observations and expert review in later phases.

## Phase 2 controlled crop problems

Phase 2 adds deterministic stress treatments to the Phase 1 truth farm. It
runs every zone twice: once with the unchanged Phase 1 inputs (`baseline`) and
once with the configured treatments (`stressed`). The paired results isolate
the effect of each event from the farm's existing spatial differences.

The initial event file is `data/phase2/stress_events.csv`. It uses one row per
event-zone combination, so every footprint remains explicit and inspectable.
Zones may have only one event in this first controlled experiment. Multiple
rows can share an event ID only when their type, timing, duration, and severity
are identical.

| Stress type | Current synthetic implementation |
|---|---|
| `water_deficit` | Each active day, soil moisture is capped by moving the configured severity fraction from field capacity toward the wilting point before the next WOFOST daily step. Rain and other daily fluxes can make the exported end-of-day value slightly exceed that cap. |
| `nitrogen_deficit` | The severity fraction is removed from initial available nitrogen at planting and the condition lasts for the full campaign. |
| `stand_loss` | The severity fraction is treated as bare area. WOFOST simulates the surviving crop area, then LAI, biomass, storage-organ mass, and crop-N amounts are converted to whole-zone averages. |

`start_day` is zero-based relative to each zone's planting date. The onset and
inclusive end date are resolved and written to the impact output. Severity is
a fraction greater than zero and less than one. Nitrogen deficit and stand
loss currently start at planting and last all 200 campaign days; this keeps
their simple initial-condition representation honest.

Run Phase 2 from the repository root with:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe run -n py3_pcse python -m src.simulation.run_phase2 --config config/phase2.yaml
```

Existing results remain protected unless replacement is explicit:

```powershell
C:\Users\layde\anaconda3\Scripts\conda.exe run -n py3_pcse python -m src.simulation.run_phase2 --config config/phase2.yaml --overwrite
```

Generated files are written to the ignored `outputs/phase2/` directory:

- `daily_truth.csv`: baseline and stressed daily truth, including assigned and
  currently active event IDs;
- `zone_summary.csv`: paired terminal results for all 25 zones;
- `stress_impacts.csv`: configured onset, footprint and severity plus LAI and
  `TWSO` differences for every affected zone;
- `stress_trajectories.png`: baseline-versus-stressed LAI by event type; and
- `yield_impact_heatmap.png`: the zone-level percentage loss in the dry
  storage-organ biomass proxy.

These are deliberately strong synthetic perturbations for pipeline testing,
not calibrated estimates of real drought, nitrogen loss, or plant mortality.
Phase 2 does not simulate sensors, drone missions, human scouting, detection
logic, diagnosis, interventions, or economics.

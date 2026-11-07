# Changelog

All notable changes to RAINS are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `config.yaml` for the root folder, the NFM layers to include and the
  population size; `rains_main.py --config <file>` selects another file.
- Docstrings for every helper function.
- Automated tests (`tests/`) and a GitHub Actions workflow.
- `requirements.txt`, `environment.yaml`, `pyproject.toml`, `CITATION.cff`,
  `.gitignore` and this changelog.

### Changed
- `rains_main.py` imports `rains_all_functions.py` and
  `SHETRAN_Post_Simulation_Functions.py` from its own folder, so the scripts no
  longer need to be copied to `C:\z_nfm_opt\01_codes`.
- README rewritten with installation, input data, configuration, usage and
  output details.

The optimisation itself is unchanged: with the default `config.yaml`, results
are identical to version 0.1.0.

## [0.1.0] - 2025-05-28

- First public release of `rains_main.py`, `rains_all_functions.py` and
  `SHETRAN_Post_Simulation_Functions.py`.

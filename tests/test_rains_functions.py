"""Tests for the file, raster and plotting helpers in rains_all_functions.py.

All inputs are small synthetic files created in a temporary folder; SHETRAN
and the Irwell data are not needed.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.stats import genextreme
from shapely.geometry import box

import rains_all_functions as rf


# --- Helpers -------------------------------------------------------------------

def write_shetran_output(path, start, end, links, seed=0):
    """Write a small file in the layout read_shetran expects."""
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    season = 10 + 5 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    data = np.column_stack([season + rng.gamma(2, 3, len(dates))
                            for _ in range(len(links) + 1)])
    with open(path, "w") as f:
        f.write("Synthetic SHETRAN multi-point discharge output\n")
        f.write(f"{'Outlet and river link numbers:':<39}"
                + "".join(f"{link:>8d}" for link in links) + "\n")
        for row in data:
            f.write(" ".join(f"{v:10.4f}" for v in row) + "\n")
    return data


def write_grid(path, array, nodata=-9999):
    """Write a 1 m resolution ASCII grid with its top-left corner at (0, nrows)."""
    meta = dict(driver="AAIGrid", height=array.shape[0], width=array.shape[1],
                count=1, dtype=array.dtype,
                transform=from_origin(0, array.shape[0], 1, 1), nodata=nodata)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array, 1)


def write_catchments(path, ncols=4, nrows=4):
    """Two sub-catchments: opt_id 0 = left half, opt_id 1 = right half."""
    half = ncols / 2
    gdf = gpd.GeoDataFrame(
        {"opt_id": [0, 1]},
        geometry=[box(0, 0, half, nrows), box(half, 0, ncols, nrows)],
        crs="EPSG:27700")
    gdf.to_file(path)


# --- SHETRAN output and growth curves -------------------------------------------

def test_read_shetran(tmp_path):
    path = tmp_path / "discharge.txt"
    data = write_shetran_output(path, "2000-01-01", "2000-01-10", [101, 202])
    df = rf.read_shetran(str(path), "2000-01-01", "2000-01-10")
    assert list(df.columns) == ["outlet", 101, 202]
    assert df.index[0] == pd.Timestamp("2000-01-01")
    assert df.index[-1] == pd.Timestamp("2000-01-10")
    np.testing.assert_allclose(df.to_numpy(), data, atol=1e-4)


def test_generate_growth_curve_matches_gev_fit():
    amax = np.array([30.1, 42.5, 35.2, 50.3, 28.9, 61.0, 39.4, 44.8, 33.3, 47.7])
    flows = rf.generate_growth_curve(amax, 3.2)
    shape, loc, scale = genextreme.fit(amax)
    rps = np.array([3.2, 5, 20, 50, 100, 200])
    np.testing.assert_allclose(flows, genextreme.ppf(1 - 1 / rps, shape, loc, scale))
    assert np.all(np.diff(flows) > 0)


def test_get_growth_curves(tmp_path):
    path = tmp_path / "long.txt"
    write_shetran_output(path, "1990-01-01", "1999-12-31", [7, 8], seed=3)
    curves = rf.get_growth_curves(str(path), "1990-01-01", "1999-12-31", 3.2)
    assert list(curves.index) == [3.2, 5, 20, 50, 100, 200]
    assert list(curves.columns) == ["outlet", 7, 8]
    # Each column equals a GEV fit of that column's calendar-year maxima.
    df = rf.read_shetran(str(path), "1990-01-01", "1999-12-31")
    amax = df.groupby(df.index.year).max()
    for col in curves.columns:
        np.testing.assert_allclose(curves[col].to_numpy(),
                                   rf.generate_growth_curve(amax[col], 3.2))


# --- NFM rasters and soil map -----------------------------------------------------

def test_nfm_selection_zeroes_selected_subcatchments(tmp_path):
    shp, grid, out = tmp_path / "cats.shp", tmp_path / "nfm.asc", tmp_path / "out.asc"
    write_catchments(shp)
    write_grid(grid, np.full((4, 4), 5, dtype="int32"))

    rf.nfm_selection([0], str(shp), str(grid), str(out), True)
    with rasterio.open(out) as src:
        result = src.read(1)
    assert np.all(result[:, :2] == 0)   # opt_id 0 (left half) removed
    assert np.all(result[:, 2:] == 5)   # opt_id 1 (right half) kept


def test_nfm_selection_can_return_array(tmp_path):
    shp, grid = tmp_path / "cats.shp", tmp_path / "nfm.asc"
    write_catchments(shp)
    write_grid(grid, np.full((4, 4), 3, dtype="int32"))
    result = rf.nfm_selection([1], str(shp), str(grid), None, False)
    assert np.all(result[:, :2] == 3) and np.all(result[:, 2:] == 0)


def test_nfm_selection_requires_output_path(tmp_path):
    shp, grid = tmp_path / "cats.shp", tmp_path / "nfm.asc"
    write_catchments(shp)
    write_grid(grid, np.full((4, 4), 3, dtype="int32"))
    with pytest.raises(ValueError):
        rf.nfm_selection([1], str(shp), str(grid), None, True)


def test_soil_update_adds_new_soil_type(tmp_path):
    org, out = tmp_path / "soil.asc", tmp_path / "soil_out.asc"
    write_grid(org, np.array([[1, 2], [3, -9999]], dtype="int32"))
    improved = np.array([[0, 1], [1, 1]])
    rf.soil_update(str(org), improved, str(out))
    with rasterio.open(out) as src:
        result = src.read(1)
        assert src.nodata == -9999
    # Largest valid soil type is 3, so improved cells become 4;
    # the nodata cell stays nodata.
    assert result.tolist() == [[1, 4], [4, -9999]]


# --- SHETRAN set-up files -----------------------------------------------------------

VSD = (":VS01 header\n"
       "      1      2\n"
       ":VS03 NLYR NCAT INITYP\n"
       "      3      5      1\n"
       ":VS04 other\n"
       "      9\n")


def test_initial_conditions_setting(tmp_path):
    (tmp_path / "input_Irwell_to_Bury_Bridge_200m_vsd.txt").write_text(VSD)
    (tmp_path / "input_Irwell_to_Bury_Bridge_200m_vsi.txt").write_text("ic\n")
    rf.initial_conditions_setting(str(tmp_path))
    lines = (tmp_path / "input_Irwell_to_Bury_Bridge_200m_vsd.txt").read_text().splitlines()
    assert lines[3].split() == ["3", "5", "2"]
    assert lines[1] == "      1      2" and lines[5] == "      9"


def test_initial_conditions_setting_exits_without_vsi_file(tmp_path):
    (tmp_path / "input_Irwell_to_Bury_Bridge_200m_vsd.txt").write_text(VSD)
    with pytest.raises(SystemExit):
        rf.initial_conditions_setting(str(tmp_path))


# --- Contribution map and figure ------------------------------------------------------

def make_subcatchments():
    return gpd.GeoDataFrame(
        {"opt_id": [0, 1, 2]},
        geometry=[box(373000 + 2000 * i, 409000, 375000 + 2000 * i, 411000)
                  for i in range(3)],
        crs="EPSG:27700")


def test_catch_contribution():
    population = np.array([[1, 0, 1], [1, 1, 0]])
    joined, column = rf.catch_contribution(make_subcatchments(), "opt_id", 3,
                                           population, 2)
    assert column == "per_count"
    assert joined["count"].tolist() == [2, 1, 1]
    assert joined["per_count"].tolist() == [100, 50, 50]


def test_scatter_plot_map_plot_writes_png(tmp_path):
    population = np.array([[1, 0, 0], [1, 1, 0]])
    joined, column = rf.catch_contribution(make_subcatchments(), "opt_id", 3,
                                           population, 2)
    out = tmp_path / "Generation_No_0"
    rf.scatter_plot_map_plot(
        "default", "Test", np.array([1.0, 2.0]), np.array([[3.0], [4.0]]),
        "Evolving solution", [-1.5, 50], [-2, 60], "Total NFM area (km²)",
        "Damages avoided (£ million)", "Catchment contribution", "cividis_r",
        np.array([10, 30, 50, 70, 90]), "Contribution (%)", joined, column,
        str(out))
    assert os.path.getsize(str(out) + ".png") > 0

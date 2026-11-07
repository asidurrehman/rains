"""Tests for SHETRAN_Post_Simulation_Functions.py using small synthetic files."""
import os
import stat
import sys

import numpy as np
import pandas as pd
import pytest

import SHETRAN_Post_Simulation_Functions as sf


def test_get_date_components():
    assert sf.get_date_components("1983-10-09") == (1983, 10, 9)
    assert sf.get_date_components("09/10/1983", "%d/%m/%Y") == (1983, 10, 9)


def test_get_lib_from_flow():
    path = "C:/model/output_Irwell_discharge_sim_regulartimestep.txt"
    assert sf.get_lib_from_flow(path) == "C:/model/Irwell_LibraryFile.xml"


def test_get_library_date(tmp_path):
    lib = tmp_path / "Model_LibraryFile.xml"
    lib.write_text("<Root>\n<StartDay>3</StartDay>\n<StartMonth>10</StartMonth>\n"
                   "<StartYear>1983</StartYear>\n</Root>\n")
    assert sf.get_library_date(str(lib)) == "3/10/1983"


def write_regular_discharge(path, values):
    with open(path, "w") as f:
        f.write("Discharge at outlet with regular timestep 24 hours\n")
        for v in values:
            f.write(f"{v}\n")


def test_load_shetran_discharge_with_given_start_date(tmp_path):
    path = tmp_path / "output_Model_discharge_sim_regulartimestep.txt"
    write_regular_discharge(path, [1.5, 2.5, 3.5])
    flow = sf.load_shetran_discharge(str(path), "01/10/1983")
    assert list(flow.columns) == ["Flow"]
    assert flow.index.name == "Date"
    assert list(flow.index) == list(pd.date_range("1983-10-01", periods=3, freq="D"))
    assert flow["Flow"].tolist() == [1.5, 2.5, 3.5]


def test_load_shetran_discharge_reads_start_date_from_library(tmp_path):
    folder = tmp_path.as_posix()
    with open(f"{folder}/Model_LibraryFile.xml", "w") as f:
        f.write("<StartDay>2</StartDay>\n<StartMonth>1</StartMonth>\n"
                "<StartYear>1990</StartYear>\n")
    path = f"{folder}/output_Model_discharge_sim_regulartimestep.txt"
    write_regular_discharge(path, [1.0, 2.0])
    flow = sf.load_shetran_discharge(path)
    assert flow.index[0] == pd.Timestamp("1990-01-02")


def test_load_shetran_discharge_falls_back_to_default_date(tmp_path):
    path = tmp_path / "output_Model_discharge_sim_regulartimestep.txt"
    write_regular_discharge(path, [1.0, 2.0])
    # No library file next to the output, so the default start date is used.
    flow = sf.load_shetran_discharge(str(path))
    assert flow.index[0] == pd.Timestamp("1980-01-01")


def test_load_discharge_sets_negative_flows_to_nan(tmp_path):
    path = tmp_path / "gauge.csv"
    path.write_text("Date,Flow\n01/02/2000,1.5\n02/02/2000,-1\n03/02/2000,2.25\n")
    flow = sf.load_discharge(str(path))
    assert flow.index[0] == pd.Timestamp("2000-02-01")
    assert flow["Flow"].iloc[0] == 1.5
    assert np.isnan(flow["Flow"].iloc[1])


def test_load_discharge_files_dispatches_and_rounds(tmp_path):
    shetran = tmp_path / "output_Model_discharge_sim_regulartimestep.txt"
    write_regular_discharge(shetran, [1.234, 2.345])
    assert sf.load_discharge_files(str(shetran))["Flow"].tolist() == [1.23, 2.35]

    gauge = tmp_path / "gauge.csv"
    gauge.write_text("Date,Flow\n01/02/2000,1.239\n")
    assert sf.load_discharge_files(str(gauge))["Flow"].tolist() == [1.24]


def test_shetran_obj_functions_perfect_fit(tmp_path):
    values = [5.0, 6.0, 9.0, 7.0, 4.0, 3.0]
    sim = tmp_path / "sim.csv"
    sim.write_text("flow\n" + "\n".join(str(v) for v in values) + "\n")
    # The first simulated value precedes the simulation, so day i matches value i+1.
    rec = tmp_path / "rec.csv"
    dates = pd.date_range("2000-01-01", periods=len(values) - 1, freq="D")
    pd.DataFrame({"date": dates, "discharge_vol": values[1:]}).to_csv(rec, index=False)

    result = sf.shetran_obj_functions(str(sim), str(rec), "2000-01-01",
                                      return_flows=True)
    assert result["NSE"] == pytest.approx(1.0)
    assert result["KGE"] == pytest.approx(1.0)
    assert result["RMSE"] == pytest.approx(0.0)
    assert result["PBias"] == pytest.approx(0.0)
    assert len(result["flows"]) == len(values) - 1


def test_edit_rundata_line_inserts_entry_under_heading(tmp_path):
    rundata = tmp_path / "rundata.txt"
    rundata.write_text("29: initial conditions file\n\n47: discharge points file\n\n")
    sf.edit_RunData_line(str(rundata), 47, "points.txt")
    sf.edit_RunData_line(str(rundata), "29", "ic.txt")
    assert rundata.read_text() == ("29: initial conditions file\nic.txt\n"
                                   "47: discharge points file\npoints.txt\n")


def test_visualisation_plan_swap_line(tmp_path):
    plan = tmp_path / "vis.txt"
    plan.write_text("A : B\nC : D   \n")
    assert sf.visualisation_plan_swap_line("C : D", "C : E", str(plan)) is None
    assert plan.read_text() == "A : B\nC : E\n"
    assert (sf.visualisation_plan_swap_line("X", "Y", str(plan))
            == "WARNING: No changes made")


def test_visualisation_plan_remove_item(tmp_path):
    plan = tmp_path / "vis.txt"
    plan.write_text(
        "item \n"
        "NUMBER^1 : NAME^ph_depth : BASIS^grid_as_grid\n"
        "GRID_OR_LIST_NO^7 : TIMES^8 : ENDITEM\n"
        "item \n"
        "NUMBER^2 : NAME^ovr_flow : BASIS^grid_as_list\n"
        "GRID_OR_LIST_NO^7 : TIMES^8 : ENDITEM\n")
    assert sf.visualisation_plan_remove_item(1, str(plan)) is None
    assert plan.read_text() == (
        "item \n"
        "NUMBER^1 : NAME^ovr_flow : BASIS^grid_as_list\n"
        "GRID_OR_LIST_NO^7 : TIMES^8 : ENDITEM\n")
    assert (sf.visualisation_plan_remove_item(9, str(plan))
            == "WARNING: No lines were edited")


def test_folder_copy(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "output_flow.txt").write_text("1")
    (src / "input_dem.asc").write_text("2")
    source, destination = str(src) + os.sep, str(dst) + os.sep

    assert sorted(sf.folder_copy(source, destination, outputs_only=True)) == ["output_flow.txt"]
    assert sorted(os.listdir(dst)) == ["output_flow.txt"]
    # Existing files are skipped unless overwrite=True
    assert sf.folder_copy(source, destination) == ["input_dem.asc"]
    assert sf.folder_copy(source, destination) == "No files to copy..."
    assert sorted(sf.folder_copy(source, destination, overwrite=True)) == [
        "input_dem.asc", "output_flow.txt"]


def test_folder_copy_skips_incomplete_simulation(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "output_flow.txt").write_text("1")
    message = sf.folder_copy(str(src) + os.sep, str(tmp_path / "dst") + os.sep,
                             complete_only=True)
    assert message.endswith("was not copied as it is incomplete.")


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script")
@pytest.mark.parametrize("last_line, expected", [
    ("Normal completion of SHETRAN run", True),
    ("Fortran Pause - Enter command", False),
    ("forrtl: severe (24): end-of-file", False),
    ("ABNORMAL END", False),
])
def test_run_shetran_skip_pause(tmp_path, last_line, expected):
    exe = tmp_path / "fake_shetran"
    exe.write_text(f'#!/bin/sh\necho "  timestep 1"\necho "{last_line}"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    assert sf.run_SHETRAN_skip_pause(str(exe), "rundata.txt", print_timestep=False,
                                     force_continue=True) is expected


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script")
def test_run_shetran_without_force_continue(tmp_path):
    exe = tmp_path / "fake_shetran"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    assert sf.run_SHETRAN_skip_pause(str(exe), "rundata.txt") is True

# -------------------------------------------------------------
# SHETRAN Post Simulation Functions
# -------------------------------------------------------------
# Ben Smith
# 08/08/2022
# -------------------------------------------------------------
# This code is written to provide users with a range of
# functions that are generally useful for manipulating and
# analysing SHETRAN simulations. These include data extraction,
# analysis and visualisation.
#
# You can add these functions to other projects / scripts using the following code:
#   import sys
#   sys.path.append("I:/SHETRAN_GB_2021/01_Scripts/Other/OFFLINE Generic Catchment Setup Script/")
#   import SHETRAN_Post_Simulation_Functions as sf
# -------------------------------------------------------------


# --- Load in Packages ----------------------------------------
import os
import shutil
import pandas as pd
import numpy as np
import datetime
import hydroeval as he  # https://pypi.org/project/hydroeval/ - open conda prompt: pip install hydroeval
import plotly.graph_objects as go
import subprocess
from subprocess import Popen, PIPE, STDOUT


# --- Calculate Objective Functions for Flows -----------------
def shetran_obj_functions(regular_simulation_discharge_path: str, recorded_discharge_path: str,
                          start_date: str, period: list = None, recorded_date_discharge_columns: list = None,
                          return_flows=False, return_period=False):
    """
    Notes:
    - Assumes daily flow data, can be altered within function.
    - Assumes that recorded flows have dates and are regularly spaced, with no gaps.
    - NAs will be skipped from the analysis. NA count will be returned.
    - A more updated version of this functions exists in the SHETRAN Optimise functions:
        https://github.com/DrBenSmith/SHETRAN-python-optimiser

    TODO - consider whether you can add code that allows you to take other columns
            from the record so that they can be visualised at the end.

    regular_simulation_discharge_path:  Path to the txt file
    recorded_discharge_path:            Path to the csv file
    start_date:                         The start date of the simulated flows: "DD-MM-YYYY"
    period:                             The period to use (i.e. calibration/validation) as a list of dates:
                                        ["YYY-MM-DD", "YYY-MM-DD"].
                                        Leave blank if you want to use the whole thing.
                                        Leave as single item in list if you want to use until the end of the data.
    recorded_date_discharge_columns:    The columns (as a list) that contain the date and then flow data.
    RETURNS:                            The NSE value as an array.
    """

    # --- Read in the flows for Sim and Rec:
    if recorded_date_discharge_columns is None:
        recorded_date_discharge_columns = ["date", "discharge_vol"]

    flow_rec = pd.read_csv(recorded_discharge_path,
                           usecols=recorded_date_discharge_columns,
                           parse_dates=[recorded_date_discharge_columns[0]])

    # Set the columns to the following so that they are always correctly referenced:
    # (Do not use recorded_date_discharge_columns!)
    flow_rec.columns = ["date", "discharge_vol"]
    flow_rec = flow_rec.set_index('date')

    # Read in the simulated flows:
    flow_sim = pd.read_csv(regular_simulation_discharge_path)
    flow_sim.columns = ["flow"]

    # --- Give the simulation dates:
    flow_sim['date'] = pd.date_range(start=start_date, periods=len(flow_sim), freq='D')
    flow_sim = flow_sim.set_index('date').shift(-1)
    # ^^ The -1 removes the 1st flow, which is the flow before the simulation.

    # --- Resize them to match
    flows = flow_sim.merge(flow_rec, on="date")
    # ^^ Merge removes the dates that don't coincide. Beware missing record data!

    # Select the period for analysis (if given):
    if period is not None:
        if len(period) == 1:
            flows = flows[flows.index >= period[0]]
        if len(period) == 2:
            flows = flows[(flows.index >= period[0]) & (flows.index <= period[1])]

    # --- Do the comparison
    flow_NAs = np.isnan(flows["discharge_vol"])  # The NAs are actually automatically removed

    # Calculate the objective function:
    obj_funs = {"NSE": np.round(he.evaluator(he.nse, flows["flow"], flows["discharge_vol"]), 2),
                "KGE": np.round(he.evaluator(he.kge, flows["flow"], flows["discharge_vol"]), 2),
                "RMSE": np.round(he.evaluator(he.rmse, flows["flow"], flows["discharge_vol"]), 2),
                "PBias": np.round(he.evaluator(he.pbias, flows["flow"], flows["discharge_vol"]), 2)}

    # Print out the % of data that are NA:
    print(str(round(len(np.arange(len(flow_NAs))[flow_NAs]) / len(flows) * 100, 3)) + "% of comparison data are NA")

    if (period is not None) & (return_period):
        obj_funs["period"] = period

    if return_flows:
        obj_funs["flows"] = flows

    return obj_funs


# --- Sweep Files from Blades to Folder -----------------------
def folder_copy(source_folder, destination_folder, overwrite=False, outputs_only=False, complete_only=False):
    """
    I:/SHETRAN_GB_2021/scripts/Blade_Sweeper.py" will execute this function for the Blades and CONVEX.

    :param source_folder: E.g. "C:/BenSmith/Blade_SHETRANGB_OpenCLIM_UKCP18rcm_220708_APM/Temp_simulations/"
    :param destination_folder: E.g. "I:/SHETRAN_GB_2021/UKCP18rcm_220708_APM_GB/"
    :param overwrite: For if you want to overwrite the destination folder (False/True)
    :param outputs_only: For if you only want to copy "outputs_..." files (False/True)
    :param complete_only: For if you only want to copy completed files, based on PRI file (False/True)
    :return: A list of copied files
    """

    # Check whether the destination folder exists (make it if not):
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    # Get a list of the folders to copy:
    files_2_copy = os.listdir(source_folder)

    # Set conditions to skip incomplete simulations if desired:
    if complete_only:
        pri_file = [i for i in files_2_copy if i.endswith("pri.txt")]

        # If there isn't a PRI file, skip the copy:
        if len(pri_file) == 0:
            return source_folder + " was not copied as it is incomplete."

        # If there is, then check completeness:
        with open(source_folder + pri_file[0], 'r') as f:
            lines = f.read().split("\n")
            comp_line = lines[-24]
            if not comp_line.startswith("Normal completion of SHETRAN run:"):
                # If incomplete, skip the copy, else continue:
                return source_folder + " was not copied as it is incomplete."

    # If NOT overwriting files, remove duplicates from source list:
    if not overwrite:
        destination_files = os.listdir(destination_folder)
        files_2_copy = [i for i in files_2_copy if i not in destination_files]

    # If you only want to copy outputs, only include these in the copy list:
    if outputs_only:
        files_2_copy = [i for i in files_2_copy if "output" in i]

    # Copy each of the remaining files across:
    if len(files_2_copy) > 0:
        for file in files_2_copy:
            shutil.copy2(source_folder + file, destination_folder + file)
        return files_2_copy
    else:
        return "No files to copy..."


# --- Get Date Components from a data string ------------------
def get_date_components(date_string, fmt='%Y-%m-%d'):
    # "1980/01/01"
    date = datetime.datetime.strptime(date_string, fmt)
    return date.year, date.month, date.day


# --- Load SHETRAN Regular Timestep Data ----------------------
def get_lib_from_flow(discharge_filepath: str):
    """
    Function for extracting the model name from the discharge file path. Used in load_shetran_LibraryDate.
    :param discharge_filepath:  String of the file path to the SHETRAN output discharge_sim_regulartimestep.
    :return: The filepath to the model library file (assuming typical naming convention.
    """
    fp_split = discharge_filepath.split('/')
    fname = fp_split[-1].split('discharge_sim_regulartimestep')[0]
    fname_ext = fp_split[-1].split('discharge_sim_regulartimestep')[1]
    fname_ext = fname_ext[0:-4] if len(fname_ext) > 0 else ''
    lib_path = '/'.join(fp_split[0:-1]) + f'/{fname[7:]}LibraryFile{fname_ext}.xml'
    return lib_path


def get_library_date(filepath: str):
    """
    Function for extracting the model start date from a Library file.
    :param filepath: String of the file path to the SHETRAN simulation library file.
    :return: A string of the date in format dd/mm/yyyy.
    """
    # Read file in read mode 'r'
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("<StartDay>"):
                sd = line.split('>')[1].split('<')[0]
            if line.startswith("<StartMonth>"):
                sm = line.split('>')[1].split('<')[0]
            if line.startswith("<StartYear>"):
                sy = line.split('>')[1].split('<')[0]
        return f'{sd}/{sm}/{sy}'


def load_shetran_discharge(flow_path: str, simulation_start_date=None):
    """
    This will take the simulation dates from the library file and the output timestep from the regular discharge file
    and load in the SHETRAN discharge at the outlet.
    :param flow_path: String of the file path to the SHETRAN output discharge_sim_regulartimestep.
    :param simulation_start_date: string 'dd/mm/yyyy'
    :return: Pandas dataframe of flows.
    """
    # Load in the flow:
    flow = pd.read_csv(flow_path, skiprows=1, header=None)

    # If a simulation start date was not given, extract one from the library file or use a default.
    if simulation_start_date is None:
        try:
            # See whether there is a library file with a date in it and use that:
            lib_name = get_lib_from_flow(flow_path)
            simulation_start_date = get_library_date(lib_name)
        except:
            # If not Library file, use a default date. December 1980 for long UKCP18 scenarios, else Jan 1980.
            simulation_start_date = "12/1/1980" if len(flow) >= 30000 else "1/1/1980"
            print(f'Warning >load_shetran_LibraryDate< No date given or library file found.'
                  f'Using default simulation start date: {simulation_start_date}')

    # Get the timestep of the data from the discharge file:
    timestep = pd.read_csv(flow_path, nrows=1, header=None)
    timestep = timestep[0][0].replace(' ', '')
    timestep = timestep.split('timestep')[-1][:-5]

    # Build the flow dataframe
    flow.index = pd.date_range(start=pd.to_datetime(simulation_start_date, dayfirst=True),
                               periods=len(flow), freq=f'{timestep}H')
    flow.index.name = "Date"
    flow.columns = ["Flow"]
    return flow


# --- Load and prepare non-SHETRAN flow data ------------------
def load_discharge(flow_path: str):
    """
    Load in non-SHETRAN flow data. Assumes two columns: Date and Flow. Column names not important. Date format is.
    :param flow_path: String of path to the csv file.
    :return:
    """
    flow = pd.read_csv(flow_path, sep='\t|,', parse_dates=[0], dayfirst=True, engine='python')
    # Rename the columns if needed (assuming the first column contains dates and the second column contains values)
    flow.columns = ['Date', 'Flow']
    # Set the 'Date' column as the index
    flow.set_index('Date', inplace=True)
    flow['Flow'][flow['Flow'] < 0] = np.nan
    return flow


# --- Load and prepare SHETRAN and recorded flow data ---------
def load_discharge_files(discharge_path):
    """
    Load in discharge data from SHETRAN or other sources
    :param discharge_path:
    :return:
    """
    if '_discharge_sim_regulartimestep' in discharge_path:
        print('Loading SHETRAN regular timestep flow output')
        flow = load_shetran_discharge(discharge_path)
    else:
        print('Loading non-SHETRAN data')
        flow = load_discharge(discharge_path)
    # else:
    #     print('Unsure which format to use...')

    flow = round(flow, 2)
    return flow


# --- Plot Flow Data Interactively -----------------------------
def plot_flow_datasets(flow_path_list: dict):
    """
    This will plot an interactive line plot of flow data from different (correctly formatted) sources.
    Example:
        paths = {
        'NRFA 69035': 'myfolder/NRFA Gauged Daily Flow 69035.csv',
        'SHETRAN 69035': 'myfolder/output_69035_discharge_sim_regulartimestep.txt
        }
        plot_flow_datasets(flow_path_list=paths)
    :param flow_path_list: Dictionary of labels and paths to SHETRAN _discharge_sim_regulartimestep files
            or other flow datasets.
    :return:
    """
    # Set up figure:
    fig = go.Figure()
    # Run through each dictionary item.
    for sim_name, sim_data in flow_path_list.items():
        # Load the flow.
        flow = load_discharge_files(sim_data)
        # Build the plot.
        fig.add_trace(go.Scatter(x=flow.index, y=flow.Flow, name=sim_name, opacity=0.8))
    # Update layout properties:
    fig.layout['xaxis'] = dict(title=dict(text='Date'))
    fig.layout['yaxis'] = dict(title=dict(text='Flow (cumecs)'))
    fig.update_layout(title_text="Catchment Discharge")
    # Show the plot:
    fig.show()


# --- Edit the Visualisation Plan Prior to Simulation ----------
# There are occasions where you may not want the default visualisation plan setup. This code is useful for
# making clean edits of the file.

def visualisation_plan_swap_line(old_line, new_line, file_in, file_out=None, strip_ws=True):
    """
    Take an existing line from the visualisation plan and replace it with a new one.
    old_line & new_line  - Strings of the full lines in the visualisation plan (without white space,
                            see strip_ws).
    file_out             - Do not specify  if you want to overwrite.
    strip_ws             - True/False depending on whether you want trailing white space to be matched and included
                            in output. - Default True, so do not include white space in old line (and probably not new
                             line, just for consistency).
    ALSO, consider whether there are multiple matches.

    TODO Make a replacement method based on line number instead of string matching.
    """

    if file_out is None:
        file_out = file_in

    with open(file_in, 'r') as vis:

        replacement = ""
        change_checker = 0

        for line in vis:

            if strip_ws:
                line = line.rstrip()

            changes = line.replace(old_line, new_line)

            if line != changes:
                change_checker += 1
            replacement = replacement + changes + "\n"

    with open(file_out, "w") as new_vis:
        new_vis.write(replacement)

    if change_checker == 0:
        return "WARNING: No changes made"


def visualisation_plan_remove_item(item_number, vis_file_in=str, vis_file_out=None):
    """
    Strip outputs from the visualisation plan based on their number.
    If you use this in combination with the number altering, that you need to match the altered number.
    If you are removing multiple items, remove the higher numbers first. E.g.
        for n in [6, 5, 4, 3, 1]:
            visualisation_plan_remove_item(item_number=n, vis_file_in=vis_plan_filepath)
    :param item_number: string or integer relating to item number (e.g. NUMBER^1)
    :param vis_file_in: string filepath to the visualisation plan.
    :param vis_file_out: string filepath to an output if you want this to differ from the original.
    :return: a new visualisation plan, overwriting the original unless specified.
    """

    if vis_file_out is None:
        vis_file_out = vis_file_in

    with open(vis_file_in, 'r') as vis:
        updated_text = ""
        number_corrector = 0
        changes_made = False

        for line in vis:
            line = line.strip().split(" : ")

            # # IF the line starts with item then skip ('item' will be written later)
            if line[0].startswith("item"):
                continue

            # IF the line starts with NUMBER, decide whether to read or write:
            # if line[0][0:len(line[0]) - 2] == "NUMBER":
            if line[0].startswith('NUMBER'):

                # # IF it is the number of interest read the next line too, not writing either
                # # and add one to the index corrector:
                # if line[0][-1] == str(item_number):
                if line[0].startswith(f'NUMBER^{str(item_number)}'):
                    next(vis)
                    number_corrector += 1
                    changes_made = True

                # IF a different number:
                else:
                    new_number = int(line[0][-1]) - number_corrector
                    line[0] = str(line[0][0:len(line[0]) - 1] + str(new_number))
                    updated_text = updated_text + 'item \n' + " : ".join(line) + "\n" + next(vis)

            # If neither, just copy the line:
            else:
                updated_text = updated_text + " : ".join(line) + "\n"

    with open(vis_file_out, "w") as new_vis:
        new_vis.write(updated_text)

    if not changes_made:
        return "WARNING: No lines were edited"


def clean_visualisation_plan(vis_plan_filepath):  # , clear_level=False

    # if clear_level:
    #     for n in [6, 5, 4, 3, 2, 1]:
    #         visualisation_plan_remove_item(item_number=n, vis_file_in=vis_plan_filepath)
    # else:
    for n in [6, 5, 4, 3, 1]:
        visualisation_plan_remove_item(item_number=n, vis_file_in=vis_plan_filepath)

    visualisation_plan_swap_line(old_line="GRID_OR_LIST_NO^7 : TIMES^8 : ENDITEM",
                                 new_line="GRID_OR_LIST_NO^7 : TIMES^9 : ENDITEM",
                                 file_in=vis_plan_filepath)

    visualisation_plan_swap_line(
        old_line="NUMBER^1 : NAME^ph_depth : BASIS^grid_as_grid : SCOPE^squares :  EXTRA_DIMENSIONS^none",
        new_line="NUMBER^1 : NAME^ph_depth : BASIS^grid_as_list : SCOPE^squares :  EXTRA_DIMENSIONS^none",
        file_in=vis_plan_filepath)


# --- Edit RunData File ----------------------------------------
def edit_RunData_line(rundata_filepath, element_number, entry):
    """
    Sometimes we may want to add or change the content of the rundata file (between preparing and running the simulation
    For example if we want to add a baseflow file or additional discharge outputs.
    :param rundata_filepath: string filepath to the rundata file.
    :param element_number: Str or int of the line number that will be edited. E.g.
                            35 for >> 35: column base flow boundary condition (BFB).
    :param entry: String for the new entry.
    :return: will overwrite the rundata file with the new entry.
    """

    # Read file in read mode 'r'
    with open(rundata_filepath, 'r') as file:
        lines = file.readlines()

    # Run through the contents and overwrite the relevant line with the new content, else write the existing line:
    with open(rundata_filepath, 'w') as file:
        for line in lines:
            if line.startswith(str(element_number) + ":"):
                file.writelines(line)
                file.writelines(entry)  # This must not have an empty line beneath it.
            else:
                file.writelines(line)


# --- Run SHETRAN Through Python Skipping Errors ---------------
def run_SHETRAN_skip_pause(exe_filepath, rundata_filepath, print_timestep=True, force_continue=False):
    """
    The following code can be used to run SHETRAN through python in such away that any issues with the model will be
    skipped over. This is useful when trying to run the model lots of times, such as during an optimisation when a
    failed model is not important but a paused script is.
    :param exe_filepath: file path of the SHETRAN.exe executable
    :param rundata_filepath: file path to the run data file
    :param print_timestep: TRUE/FALSE - this is modely useful for troubleshooting, else it will fill the console.
    :param force_continue: TRUE/FALSE - use according to whether you wish to skip any FORTRAN Pauses or Errors.
    :return:
    """
    if not force_continue:
        subprocess.call([exe_filepath, '-f ', rundata_filepath])
        return True
    else:
        successful_simulation = True
        # Run the exe, passing outputs and errors to process so that they can be monitored
        with Popen([exe_filepath, '-f ', rundata_filepath], stdout=PIPE, stderr=STDOUT, text=True) as process:  # stdin=PIPE
            while successful_simulation:
                out = process.stdout.readline().lstrip()  # .decode('utf-8')

                # If user wants the timestep to be printed, print the output:
                if print_timestep:
                    print(out)

                # Catch SHETRAN Pauses and skip to the next simulation:
                if out.startswith("Fortran Pause"):
                    print('\n"FORTRAN PAUSE" detected. Exiting simulation.', flush=True, end="\n")
                    successful_simulation = False
                    process.kill()
                    break  # process.terminate() # process.communicate('\n') # process.****()

                # Catch Fortran Errors and skip to the next simulation:
                if out.startswith("forrtl"):  # or out.startswith('forrtl')
                    print(f'\n Error detected. Exiting simulation. Error: \n {out}', flush=True, end="\n")
                    successful_simulation = False
                    process.kill()
                    break

                # Once finished, move on to the analysis:
                if out.startswith('ABNORMAL END'):
                    successful_simulation = False
                    process.kill()
                    break

                # Once finished, move on to the analysis:
                if out.startswith('Normal completion'):
                    break

        return successful_simulation




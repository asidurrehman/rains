# -*- coding: utf-8 -*-
"""
 ██▀███   ▄▄▄       ██▓ ███▄    █   ██████ 
▓██ ▒ ██▒▒████▄    ▓██▒ ██ ▀█   █ ▒██    ▒ 
▓██ ░▄█ ▒▒██  ▀█▄  ▒██▒▓██  ▀█ ██▒░ ▓██▄   
▒██▀▀█▄  ░██▄▄▄▄██ ░██░▓██▒  ▐▌██▒  ▒   ██▒
░██▓ ▒██▒ ▓█   ▓██▒░██░▒██░   ▓██░▒██████▒▒
░ ▒▓ ░▒▓░ ▒▒   ▓▒█░░▓  ░ ▒░   ▒ ▒ ▒ ▒▓▒ ▒ ░
  ░▒ ░ ▒░  ▒   ▒▒ ░ ▒ ░░ ░░   ░ ▒░░ ░▒  ░ ░
  ░░   ░   ░   ▒    ▒ ░   ░   ░ ░ ░  ░  ░  
   ░           ░  ░ ░           ░       ░  
                                           
Resilient & Affordable Implementation of Natural flood management using SHETRAN
-------------------------------------------------------------------------------
File name: rains_main.py

@author: Asid Ur Rehman, Elizabeth Lewis, Ben Smith
Organisations: Manchester University, Newcastle Univeristy

About
------
This script file contains main part of RAINS - a framework which integrates
evolutionary multi-objective genetic algorithm (NSGA-II) with SHETRAN to find
optimal locations and sizes of NFM features for their cost-effective design.

Usage
-----
Set the root folder and run settings in config.yaml, then run this script
from Spyder or from a terminal:

    python rains_main.py                      # uses config.yaml next to this file
    python rains_main.py --config my.yaml     # uses another configuration file

"""
import os
import sys
import subprocess
import geopandas as gpd
import numpy as np
import shutil
import pandas as pd
import copy
from scipy.stats import genextreme
import pickle
import argparse
from pathlib import Path
import yaml

# =============================================================================
# HOW THIS SCRIPT WORKS
# =============================================================================
# RAINS couples the NSGA-II genetic algorithm with the SHETRAN hydrological
# model to decide in which SHETRAN sub-catchments natural flood management
# (NFM) should be implemented. Two objectives are optimised:
#   - total NFM area in km² (minimised), and
#   - flood damages avoided for the communities at risk, in £ million
#     (maximised; the code minimises the negative value instead).
#
# The script runs from top to bottom in these stages (search for "STEP n"):
#   STEP 1   Configuration: read config.yaml, move to the root folder and
#            load the two helper modules.
#   STEP 2   Read the NFM opportunity data and the communities at risk (CaR).
#   STEP 3   Design-hydrograph volumes of every community at risk.
#   STEP 4   Maximum NFM storage available to every community at risk.
#   STEP 5   Return period of the modelled event, and flow growth curves from
#            the baseline and maximum-NFM SHETRAN simulations.
#   STEP 6   File paths: SHETRAN model, NFM grids and outputs.
#   STEP 7   Random initial population.
#   STEP 8   shetran_sim(): the function that evaluates solutions with SHETRAN.
#   STEP 9   Simulate and plot the initial population.
#   STEP 10  Set up the records used to export results.
#   STEP 11  Generation loop (100 generations):
#              11a rank the parents (non-dominated sorting, crowding distance)
#              11b create new offspring (tournament selection, crossover,
#                  mutation, removal of duplicates and of solutions that
#                  were already simulated)
#              11c simulate the offspring with SHETRAN
#              11d export every simulated solution (simulated_data.csv)
#              11e combine parents and offspring and keep the best pop_size
#              11f plot the new population
#              11g the selected solutions become the parents of the next
#                  generation
#              11h export the population of every generation
#                  (generation_data.csv)
#   STEP 12  Pareto-optimal solutions of the last generation: plot and export.
#   STEP 13  How often each sub-catchment appears in the optimal solutions.
#   STEP 14  Re-simulate the optimal solutions to keep their SHETRAN flows.
#
# Lines marked "USER INPUT:" are the places to edit when adapting RAINS to
# another catchment, event or set-up. The root folder, the NFM layers and the
# population size are set in config.yaml instead of in this script.
#
# =============================================================================
# NAMING CONVENTIONS
# =============================================================================
# chromosome  One candidate solution: a 1-D array of 0/1 genes with one gene
#             per SHETRAN sub-catchment (gene i is the sub-catchment with
#             opt_id == i in ui_shetran_sub_catchments.shp). 1 = NFM is
#             implemented in that sub-catchment, 0 = no NFM there.
# chrom_len   Number of genes = number of sub-catchments.
# population  2-D array of chromosomes, shape (number of solutions, chrom_len).
# *_chrom_cost    NFM area of each solution in km², shape (n, 4). Columns:
#                 0 RAFs, 1 woodland, 2 improved grassland/soil, 3 total.
#                 Only the total (column 3) is used as an objective. "Cost"
#                 here means NFM area, not money.
# *_chrom_damage  Damages avoided by each solution in £ million, shape (n, 1).
#                 It is negated when the two objectives are paired, because
#                 the NSGA-II functions minimise both objectives.
# *_obj_record    The paired objectives, shape (n, 2):
#                 [total NFM area, -damages avoided].
# Prefixes:
#   i_          initial population (generation 0)
#   p_          parents: the population at the start of a generation
#   o_          offspring created and simulated in the current generation
#   comb_       parents and offspring combined
#   f_          the fittest pop_size solutions selected from comb_
#   gen_        populations of all generations stacked together (for export)
#   simulated_  every solution simulated so far (initial + all offspring)
#   opt_        Pareto-optimal solutions of the final generation
#   org_        original (backup) input files that are copied or modified
#   she_        files written into the SHETRAN model folder for one run
#   in_         arguments of shetran_sim()
# Abbreviations: NFM natural flood management; RAF runoff attenuation
#   feature; CaR community at risk; RP return period (years); WFD Water
#   Framework Directive catchment; FEH Flood Estimation Handbook; GEV
#   generalised extreme value distribution.

#-----------------------------------------------------------------------------#
# STEP 1
# Configuration: machine-specific paths and run settings live in config.yaml
#-----------------------------------------------------------------------------#
# USER INPUT: edit config.yaml (root folder, NFM layers, population size), or
# pass another file with:  python rains_main.py --config my.yaml

# Folder containing this script; config.yaml and the helper modules are
# looked up here
SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(
    description="Run RAINS: NSGA-II optimisation of NFM features with SHETRAN.")
parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"),
                    help="Path to the YAML configuration file")
args, _ = parser.parse_known_args()   # parse_known_args keeps Spyder/IPython working
with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Root folder holding the SHETRAN model, input data and output folders.
# normpath gives the same path string as before (e.g. C:\z_nfm_opt on Windows)
root = os.path.normpath(config["root"])
# From here on the working directory is the root folder, so relative file
# names (e.g. the SHETRAN library file given to the prepare program in
# shetran_sim, and the output file renamed there) are resolved from it.
os.chdir(root)

# Load the function modules that sit next to this script
#   sf = general SHETRAN helpers (edit the rundata file, run SHETRAN)
#   rf = RAINS functions (NSGA-II operators, NFM grid editing, plots)
sys.path.insert(0, str(SCRIPT_DIR))
import SHETRAN_Post_Simulation_Functions as sf
import rains_all_functions as rf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#-----------------------------------------------------------------------------#
# STEP 2: NFM opportunity data and communities at risk (CaR)
#-----------------------------------------------------------------------------#
# USER INPUT: choose the NFM layers in config.yaml (NFM_features_to_include).
# All input files in this step are read from <root>/04_irwell_notebook_data.

# Select which NFM features to consider (set in config.yaml).
# Available layers: ['Extra_Pond_Capacity_RP100_10', 'WfW', 'Losses', 'ImpGrass']
NFM_features_to_include = config["NFM_features_to_include"]

# Read in the NFM spatial data
# gdf: NFM opportunity polygons (the WFD catchments used for wfd_area in
# STEP 4), with one column per NFM opportunity layer and the polygon area
# (Area_km2).
gdf = gpd.read_file(os.path.join(root,'04_irwell_notebook_data',
                                 "a0000002b.gdbtable"))
# Three new columns, named like the layers in NFM_features_to_include: the
# workshop NFM areas (km², according to the column names) multiplied by
# 10 000.
gdf['WfW'] = gdf['Workshop_WfW_Area_km2']*10000
gdf['Losses'] = gdf['Workshop_CombLoss_Area_km2']*10000
gdf['ImpGrass'] = gdf['Workshop_ImpGrass_Area_km2']*10000
# Sum of the chosen layers = maximum NFM storage of each polygon (STEP 4)
gdf['total_NFM_storage'] = gdf[NFM_features_to_include].sum(axis=1)

# Find unique communities at risk in the Jacobs storage tool
# NFM_storage: design hydrographs from the Jacobs storage tool (a pickled
# pandas DataFrame). Columns used here: Unique_ID (location), RP ('RP2',
# 'RP5', ... 'RP200'), timestep and QT (flow).
file = open(os.path.join(root,'04_irwell_notebook_data',
                         'Calculated_Hydrographs.p'),'rb')
NFM_storage = pickle.load(file)
file.close()
# COR: one row per location (only RP2 and timestep 0.1 are kept, so each
# location appears once). The spatial join adds the attributes of the NFM
# polygon each location lies within (total_NFM_storage, Area_km2, ...).
COR = NFM_storage.loc[(NFM_storage.RP == 'RP2')&(NFM_storage.timestep == 0.1)]
joined_gdf = gpd.sjoin(COR, gdf, how="inner", predicate="within")

# Read in the FEH data to find community at risk catchment area
# (columns X, Y and AREA are used in STEP 4)
feh_data = pd.read_excel(os.path.join(root,'04_irwell_notebook_data',
                                      'FEH_Data.xlsx'))

# Read in the communities at risk file names and associated Jacobs storage tool unique IDs and SHETRAN river links
# CaR_name_ID_Final.csv has one row per community at risk:
#   property_report_name    name of the community (used as the row index)
#   jacobs_storage_tool_ID  one storage-tool Unique_ID, or several "a/b/c"
#   SHETRAN_id              the SHETRAN river link that represents it
CaR_ID_df = pd.read_csv(os.path.join(root,'04_irwell_notebook_data',
                                     'CaR_name_ID_Final.csv'))
CaR_ID_df.head()   # (no effect when run as a script)
# Collect every storage-tool ID as an integer, splitting entries like "a/b".
# (CaR_IDs is not used further below.)
a = CaR_ID_df.jacobs_storage_tool_ID.values
CaR_IDs = []
for i in a:
  try:
    CaR_IDs.append(int(i))
  except:
    j = [int(k) for k in i.split("/")]
    CaR_IDs.extend(j)

CaR_IDs = list(set(CaR_IDs))
CaR_names = CaR_ID_df.property_report_name.values
CaR_ID_df.set_index("property_report_name", inplace=True)

# Read in the volume attenuates/ damages avoided curves
# damages_df: one column per community at risk with 100 rows of damages
# avoided. The rows are labelled 0.01, 0.02, ..., 1.00: the attenuated
# volume as a fraction of the community's maximum NFM storage (STEP 4). A
# row for 0.00 (no damages avoided) is added, so that a fraction rounded to
# two decimals can be looked up directly in shetran_sim (STEP 8). There the
# summed values are divided by 1 000 000 to give £ million.

damages_df = pd.read_pickle(os.path.join(root,'04_irwell_notebook_data',
                                         'vol_damage_curves.pkl'))
damages_df_index = np.around(np.linspace(0.01, 1, 100), 2) 
damages_df.set_index(damages_df_index, inplace = True)
damages_df.loc[0] = [0 for CaR in CaR_names]
damages_df = damages_df.sort_index()

# Create new datasets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-----------------------------------------------------------------------------#
# STEP 3: Design-hydrograph volumes of every community at risk
#-----------------------------------------------------------------------------#
# vols_df: rows = return periods 5, 20, 50, 100 and 200 years; one column per
# community, named by its SHETRAN river link (SHETRAN_id). Each value is the
# volume of the community's design hydrograph: the flows (QT) of all its
# storage-tool IDs are added at each timestep, summed over time and
# multiplied by the timestep length in seconds (0.1 h = 0.1*3600 s).

# Calculate volumes associated with the design hydrographs

vols_df = pd.DataFrame({"RP":[5,20,50,100,200]})

for community_name in CaR_names:
    
    # storage-tool IDs belonging to this community ("a/b" -> [a, b])
    community_ids = [int(a) for a in CaR_ID_df.loc[community_name].jacobs_storage_tool_ID.split("/")]
    
    RP5_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP5')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP20_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP20')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP50_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP50')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP100_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP100')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP200_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP200')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    # one column per community, named by its SHETRAN river link
    vols_df[CaR_ID_df.loc[community_name].SHETRAN_id] = [RP5_hydrograph, RP20_hydrograph, RP50_hydrograph, RP100_hydrograph, RP200_hydrograph]

vols_df.set_index("RP", inplace=True)

#-----------------------------------------------------------------------------#
# STEP 4: Maximum NFM storage available to every community at risk
#-----------------------------------------------------------------------------#
# max_vol_df: one column per community, four rows:
#   community_area         catchment area at the community (FEH AREA, read at
#                          the snapped X/Y of its first storage-tool ID)
#   wfd_area               sum of Area_km2 over its storage-tool IDs (each ID
#                          adds the area of the NFM polygon it lies in)
#   wfd_max_storage        sum of total_NFM_storage over the same rows (STEP 2)
#   community_max_storage  wfd_max_storage scaled by community_area / wfd_area
# community_max_storage is the divisor of the "fraction attenuated" that is
# used to read the damage curves in shetran_sim (STEP 8).

# get reference information about the max NFM storage

max_vol_df = pd.DataFrame(index=['community_area', 'wfd_area', 'wfd_max_storage', 'community_max_storage'] )

for community_name in CaR_names:
    community_ids = [int(a) for a in CaR_ID_df.loc[community_name].jacobs_storage_tool_ID.split("/")]
    # location (snapped X, Y) of the community's first storage-tool ID,
    # used to find its catchment area in the FEH table
    community_X = int(joined_gdf.loc[joined_gdf.Unique_ID == community_ids[0]].SnappedX.values[0])
    community_Y = int(joined_gdf.loc[joined_gdf.Unique_ID == community_ids[0]].SnappedY.values[0])
    community_area = feh_data.loc[(feh_data.X == community_X)&(feh_data.Y == community_Y)].AREA.values[0]
    wfd_max_storage = joined_gdf.loc[joined_gdf.Unique_ID.isin(community_ids)].total_NFM_storage.sum() # this will be found from the JBA map
    wfd_area = joined_gdf.loc[joined_gdf.Unique_ID.isin(community_ids)].Area_km2.sum() #km2
    # Find the maximum NFM storage available to the community at risk
    community_max_storage = (community_area/wfd_area) * wfd_max_storage
    max_vol_df[community_name] = [community_area, wfd_area, wfd_max_storage, community_max_storage]

#-----------------------------------------------------------------------------#
# STEP 5: Return period of the modelled event and flow growth curves
#-----------------------------------------------------------------------------#
# Four SHETRAN discharge files prepared beforehand are used (all daily, with
# the outlet plus several river links as columns, see rf.read_shetran):
#   *_baseline_long / *_max_nfm_long    long simulations (1980-2014) without
#                                       NFM and with maximum NFM everywhere
#   *_baseline_event / *_max_nfm_event  simulations of the event period only
# USER INPUT: event date, file names and simulation periods below (the event
# date and event period are repeated inside shetran_sim, STEP 8).

# Find the RP of the modelled event
event_date = '1983-10-09' # change as appropriate, this is the date of the peak flow of the event modelled in the optimisation. 
baseline_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_baseline_long.txt')
baseline_df = rf.read_shetran(baseline_path , "1980-01-01", "2014-12-30") # change path and dates as appropriate
event_flow = baseline_df.outlet.loc[event_date] # check that this is returning a single value
# Largest daily outlet flow of every calendar year, then a GEV distribution
# fitted to these annual maxima
annual_maxima = baseline_df.resample("YE").max().dropna()
shape, loc, scale = genextreme.fit(annual_maxima.outlet.values)
# probability = chance that a year's maximum is below the event flow;
# return period RP = 1 / (1 - probability) in years
probability = genextreme.cdf(event_flow, c=shape, loc=loc, scale=scale)
RP = 1/(1-probability) # e.g. RP = 3.2 at a flow of 42.8 cumecs
# print('Event RP = ', RP)

# Growth curves: for every river link, the flows for return periods
# [event RP, 5, 20, 50, 100, 200] years from a GEV fitted to that link's
# annual maxima (rows indexed by RP). Built for the baseline and for the
# maximum-NFM long simulations.
baseline_growth_curves = rf.get_growth_curves(baseline_path, "1980-01-01", "2014-12-30", RP) # change path and dates as appropriate
max_NFM_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_max_nfm_long.txt')
max_NFM_growth_curves = rf.get_growth_curves(max_NFM_path, "1980-01-01", "2014-12-30", RP) # change path and dates as appropriate

# Event-day flows (one row, one column per river link) without NFM and with
# maximum NFM. They bound the effect NFM can have on the event and are used
# to place each simulated solution between the two (STEP 8).
baseline_event_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_baseline_event.txt')
baseline_event = rf.read_shetran(baseline_event_path, "1983-10-03", "1983-10-12").loc[[event_date]] # change path and dates as appropriate
max_NFM_event_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_max_nfm_event.txt')
max_NFM_event = rf.read_shetran(max_NFM_event_path, "1983-10-03", "1983-10-12").loc[[event_date]] # change path and dates as appropriate


#-----------------------------------------------------------------------------#
# STEP 6: File paths (SHETRAN model, NFM grids, outputs)
#-----------------------------------------------------------------------------#
# USER INPUT: the folder and file names below belong to the Irwell to Bury
# Bridge 200 m SHETRAN model; change them for another model. The SHETRAN
# model itself (library file, rundata file, vsd/vsi files, ...) sits directly
# in the root folder; "she_" paths are the files written there before each
# simulation.

# Paths
# folder with shetran-prepare-snow-NFM.exe and shetran.exe
exe_path = os.path.join(root, '20_shetran_easy_setup_snow')
dmg_cal_path = os.path.join(root, '04_irwell_notebook_data')   # (not used below)

# Paths for shapefiles used for visulaisation
# (subcat_smooth_shp_path is used for the maps; cat_smooth_shp_path is not
# used below)
subcat_smooth_shp_path = os.path.join(root, '15_shapefiles',
                                      'ui_shetran_sub_catchments_smooth.shp')
cat_smooth_shp_path = os.path.join(root, '15_shapefiles',
                                      'ui_shetran_atchments_smooth.shp')

# Orginal PET csv file path
# (copied into the model folder before every simulation)
org_pet_csv_path = os.path.join(root, '05_inputs_backup',
                                'Irwell_to_Bury_Bridge_200m_PET.csv')
she_pet_csv_path = os.path.join(root, 'Irwell_to_Bury_Bridge_200m_PET.csv')

# NFM input file paths
# The *_asc grids are ASCII rasters of the SHETRAN model grid. The "maximum"
# grids hold NFM in every sub-catchment; shetran_sim keeps only the
# sub-catchments selected by a chromosome.
# shapefile for NFM location/area selection
cat_shp_path = os.path.join(root, '15_shapefiles',
                            'ui_shetran_sub_catchments.shp')
# Maximum available storage NFM
storage_asc_path = os.path.join(root, '10_nfm_backup',
                            'Irwell_to_Bury_Bridge_200m_NFM_storage.asc')
# Maximum available woodland NFM
woodland_asc_path = os.path.join(root, '10_nfm_backup',
                            'Irwell_to_Bury_Bridge_200m_NFM_woodland.asc')
# Maximum available improved grassland/soil NFM
imp_soil_asc_path = os.path.join(root, '10_nfm_backup',
                            'Irwell_to_Bury_Bridge_200m_Improved_Soil.asc')
org_soil_asc_path = os.path.join(root, '05_inputs_backup',
                                 'Irwell_to_Bury_Bridge_200m_Soil.asc')

# NFM output file paths
she_storage_asc_path = os.path.join(root,
                            'Irwell_to_Bury_Bridge_200m_NFM_storage.asc')
she_woodland_asc_path = os.path.join(root,
                            'Irwell_to_Bury_Bridge_200m_NFM_woodland.asc')
she_soil_asc_path = os.path.join(root, 'Irwell_to_Bury_Bridge_200m_Soil.asc')

# SHETRAN discharge output read after each simulation (outlet plus the river
# links listed in the discharge-points file, see shetran_sim)
she_output_path = os.path.join(root,
        'output_Irwell_to_Bury_Bridge_200m_discharge_sim_regulartimestep.txt')
opt_graph_path = os.path.join(root,'25_output_graphs')   # (not used below)
# Results (CSV files and figures) are written to <root>/25_opt_outputs,
# which must exist before the run.

# List of file paths

other_paths = [org_pet_csv_path, cat_shp_path, root, exe_path]   # (not used below)

#-----------------------------------------------------------------------------#
# STEP 7: Random initial population
#-----------------------------------------------------------------------------#
# Reads sub-catchment shapefile
# (one polygon per sub-catchment = one gene; fields used: opt_id, raf,
# woodland, imp_soil)
cat_shp = gpd.read_file(cat_shp_path)

nfm_types_count = 3   # RAFs, woodland and improved grassland/soil
pop_size = config["pop_size"]   # population size (set in config.yaml)
chrom_len = cat_shp.shape[0]   # one gene per sub-catchment

# pop_size unique random 0/1 chromosomes; the first has no NFM anywhere and
# the second has NFM in every sub-catchment (see rf.initial_pop)
i_population = rf.initial_pop(pop_size, chrom_len)
# i_population = i_population[5,:][np.newaxis,:]

#-----------------------------------------------------------------------------#
# STEP 8: shetran_sim() - evaluate solutions with SHETRAN
#-----------------------------------------------------------------------------#
# For every chromosome: write the NFM grids and soil map for the selected
# sub-catchments, prepare and run SHETRAN, then calculate both objectives.
# The function also uses tables created above at module level (vols_df,
# CaR_names, CaR_ID_df, max_vol_df, damages_df and root).

def shetran_sim (in_gen_num, in_population, in_nfm_types_count,
                 in_org_pet_csv_path,
                 in_she_pet_csv_path, in_storage_asc_path,
                 in_she_storage_asc_path, in_woodland_asc_path,
                  in_she_woodland_asc_path,in_imp_soil_asc_path,
                  in_org_soil_asc_path, in_she_soil_asc_path,
                  in_cat_shp_path, in_exe_path, in_root, in_she_output_path,
                  in_baseline_growth_curves, in_max_NFM_growth_curves,
                  in_baseline_event, in_max_NFM_event):
    """Simulate every chromosome with SHETRAN and evaluate both objectives.

    For each chromosome the maximum-NFM rasters (storage/RAFs and woodland)
    are set to zero in sub-catchments without NFM, the soil map is updated
    for improved grassland/soil, SHETRAN is prepared and run, and the
    damages avoided are estimated from the simulated event flows (see
    README.md). Uses the module-level tables ``vols_df``, ``CaR_names``,
    ``CaR_ID_df``, ``max_vol_df``, ``damages_df`` and ``root``.

    Parameters
    ----------
    in_gen_num : int
        Generation number (used for printing). The value 999 also saves the
        SHETRAN discharge file of every chromosome to
        ``25_opt_outputs/opt_flows`` as ``Chromosome_XXX.txt``.
    in_population : numpy.ndarray
        Chromosomes to simulate, shape (n, chrom_len).
    in_nfm_types_count : int
        Number of NFM types (3: RAFs, woodland, improved soil).
    in_org_pet_csv_path, in_she_pet_csv_path : str
        Original PET file and the copy placed in the SHETRAN folder.
    in_storage_asc_path, in_she_storage_asc_path : str
        Maximum NFM storage grid and the grid written for SHETRAN.
    in_woodland_asc_path, in_she_woodland_asc_path : str
        Maximum NFM woodland grid and the grid written for SHETRAN.
    in_imp_soil_asc_path : str
        Maximum improved-soil NFM grid.
    in_org_soil_asc_path, in_she_soil_asc_path : str
        Original soil grid and the soil grid written for SHETRAN.
    in_cat_shp_path : str
        Sub-catchment shapefile (fields ``opt_id``, ``raf``, ``woodland``,
        ``imp_soil``).
    in_exe_path : str
        Folder with ``shetran-prepare-snow-NFM.exe`` and ``shetran.exe``.
    in_root : str
        SHETRAN model folder (the RAINS root folder).
    in_she_output_path : str
        SHETRAN multi-point discharge output file.
    in_baseline_growth_curves, in_max_NFM_growth_curves : pandas.DataFrame
        Growth curves of the baseline and maximum-NFM long simulations.
    in_baseline_event, in_max_NFM_event : pandas.DataFrame
        Event-day flows of the baseline and maximum-NFM event simulations.

    Returns
    -------
    tuple of numpy.ndarray
        ``(nfm_area, damages_avoided_M)``: NFM area in km2, shape (n, 4),
        with columns RAFs, woodland, improved soil and total; and damages
        avoided in GBP million, shape (n, 1).
    """
    
    # To save NFM areas individually and total
    # nfm_area: one row per chromosome; columns RAFs, woodland, improved
    # soil and total (km²). damages_avoided_M: one row per chromosome (£ m).
    nfm_area = np.zeros([in_population.shape[0], in_nfm_types_count+1])
    # nfm_flow = np.zeros([in_population.shape[0], 1])
    damages_avoided_M = np.zeros([in_population.shape[0], 1])
    
    # To save all SHETRAN flow files for optimised generation
    # (in_gen_num == 999 is the final re-simulation in STEP 14)
    if in_gen_num == 999:
        opt_flows = os.path.join(root,'25_opt_outputs', 'opt_flows')
    
        if not os.path.exists(opt_flows):
            os.mkdir(opt_flows)
        else:
            # NOTE (known issue): os.rmdir only removes an EMPTY folder; if
            # opt_flows is left over from an earlier run with files in it, this fails.
            os.rmdir(opt_flows)
            os.mkdir(opt_flows)
    
    # Evaluate the chromosomes one at a time (one SHETRAN run each)
    for idx, chrom in enumerate(in_population):
        print('\nGeneration {}: solution {} of {}'.format(
            in_gen_num, idx+1, in_population.shape[0]))
        
        # --- 1. Start from clean SHETRAN inputs -------------------------------
        # Delete the PET, NFM and soil files written for the previous solution
        file_paths = [in_she_pet_csv_path, in_she_storage_asc_path,
                          in_she_woodland_asc_path,in_she_soil_asc_path]    
        # Loop through the list and remove each file if it exists
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Place original PET csv file
        shutil.copy(in_org_pet_csv_path, in_she_pet_csv_path)
        
        # --- 2. Write the NFM grids and soil map for this chromosome --------
        # Prepare NFM datasets for SHETRAN
        # Get catchments ID for no-NFM catchment
        # (genes equal to 0; gene index = opt_id of the sub-catchment)
        no_nfm_cats = np.where(chrom == 0)[0]
        cat_shp = gpd.read_file(in_cat_shp_path)
        # At least one sub-catchment has NFM: start from the maximum-NFM grids
        # and set the cells of every no-NFM sub-catchment to 0
        if len(no_nfm_cats) != cat_shp.shape[0]:
            # Set no-NFM catchment to zero in maximum NFM files
            # RAF storage grid -> written for SHETRAN
            (rf.nfm_selection (no_nfm_cats, in_cat_shp_path,
                           in_storage_asc_path, in_she_storage_asc_path, True))
            # woodland grid -> written for SHETRAN
            rf.nfm_selection (no_nfm_cats, in_cat_shp_path, in_woodland_asc_path,
                           in_she_woodland_asc_path, True)
            # improved-soil grid -> returned as an array (not written) ...
            imp_soil_sel = rf.nfm_selection (no_nfm_cats, in_cat_shp_path,
                                          in_imp_soil_asc_path, None, False)
            # ... and used to give those cells a new soil type in the soil map
            rf.soil_update (in_org_soil_asc_path, imp_soil_sel, in_she_soil_asc_path)
        else:
           # No NFM anywhere: only the original soil map is copied; the NFM
           # storage and woodland grids deleted above are not written
           shutil.copy(in_org_soil_asc_path, in_she_soil_asc_path)
        
        # --- 3. Prepare and run SHETRAN ---------------------------------------
        # Build the SHETRAN input files from the library file (the working
        # directory is the root folder). Windows executable.
        # USER INPUT: library file name for another model.
        subprocess.call([f'{in_exe_path}/shetran-prepare-snow-NFM.exe',
                         'Irwell_to_Bury_Bridge_200m_LibraryFile.xml'])
        
        # The rundata file lists the SHETRAN input/output files under numbered
        # headings; edit_RunData_line writes the file name on the line after
        # heading 47 (multi-point discharge) and heading 29 (initial conditions)
        # Set multi-point flow extraction file name on rundata file
        sf.edit_RunData_line(rundata_filepath=f'{in_root}/rundata_Irwell_to_Bury_Bridge_200m.txt',
                     element_number=47, entry='Irwell_to_Bury_Bridge_200m_discharge_points.txt')
        # Set the initial conditions file name on rundata file
        sf.edit_RunData_line(rundata_filepath=f'{in_root}/rundata_Irwell_to_Bury_Bridge_200m.txt',
                     element_number=29, entry='input_Irwell_to_Bury_Bridge_200m_vsi.txt')
        
        # Set initial conditions option in vsd file
        # (INITYP = 2: read the initial conditions from the vsi file)
        rf.initial_conditions_setting(in_root)
        
        # Run SHETRAN. force_continue=True stops the run (instead of waiting)
        # if SHETRAN hits a Fortran pause or error, so one failed simulation
        # does not halt the whole optimisation.
        sf.run_SHETRAN_skip_pause(exe_filepath=f'{in_exe_path}/shetran.exe', 
        rundata_filepath=f'{in_root}/rundata_Irwell_to_Bury_Bridge_200m.txt', 
        print_timestep=False, force_continue=True)
        
        ## Extract specific flow value at outlet (will not work for multi-point flows)
        # nfm_flow[idx] = (pd.read_csv(in_she_output_path, sep='\s+', skiprows=1,
        #                        header=None, index_col=False).to_numpy()[26])
                 
        # --- 4. Objective 2: flood damages avoided ---------------------------
        # calculate damages avoided
        # get_event diff
        # USER INPUT: event date and event simulation period (same as STEP 5)
        event_date = '1983-10-09' # Whatever the date is of the event you are simulating
        # event_RP = RP 
        # Simulated event-day flow of this solution at every river link
        scenario_event = rf.read_shetran(in_she_output_path, "1983-10-03", "1983-10-12").loc[[event_date]] # change these to match your simulation
        # Where this solution lies between maximum NFM and no NFM, per link:
        # 0 = same flow as with maximum NFM, 1 = same flow as the baseline
        scenario_adjust = (scenario_event-in_max_NFM_event)/(in_baseline_event-in_max_NFM_event) 
        # Growth curves of this solution, interpolated per link between the
        # maximum-NFM and the baseline growth curves with that fraction
        scenario_growth_curves = (in_baseline_growth_curves - in_max_NFM_growth_curves).mul(scenario_adjust.iloc[0], axis=1) + in_max_NFM_growth_curves
        # Fractional flow reduction compared with the baseline, for every
        # return period and river link
        reduction_factors = 1-(scenario_growth_curves/in_baseline_growth_curves)
        # Attenuated volume per community column: design-hydrograph volume x
        # reduction factor, then the largest value over the return periods.
        # The two tables are aligned by RP and river link; the event-RP row
        # and links not in vols_df give NaN, which .max() skips.
        vols_attenuated = vols_df.mul(reduction_factors).max()

        total_damages_avoided = 0

        # Look up the damages avoided of every community and add them up
        for CaR in CaR_names:
            shetran_id = CaR_ID_df.SHETRAN_id.loc[CaR]
            # vals = max_vol_df[CaR].values
            community_max_storage = max_vol_df[CaR].community_max_storage
            max_vol_attenuated = vols_attenuated[shetran_id]
            # attenuated volume as a fraction of the community's maximum NFM
            # storage, limited to 0-1 and rounded to match damages_df's index
            fraction_attenuated = round(max(min(max_vol_attenuated/
                                        community_max_storage, 1.0), 0), 2)
            # (the else branch is reached only if the fraction is not a
            # number, e.g. NaN)
            if fraction_attenuated >=0:
                damages_avoided = damages_df[CaR].loc[fraction_attenuated]
            else:
                damages_avoided = 0

            total_damages_avoided += damages_avoided
            
            total_damages_avoided   # (no effect)
        
        # convert to £ million, 3 decimals
        damages_avoided_M[idx] = round(total_damages_avoided/1000000,3)
                
        # --- 5. Objective 1: NFM area -----------------------------------------
        # Get catchment IDs for NFM-catchments and sum their area
        nfm_cats = np.where(chrom == 1)[0]
        # Filter the polygons whose opt_id is in the given list
        selected_cats = cat_shp[cat_shp['opt_id'].isin(nfm_cats)]
        # shapefile area fields divided by 1 000 000 (m² -> km²)
        nfm_area[idx, 0] = selected_cats["raf"].sum()/1000000
        nfm_area[idx, 1] = selected_cats["woodland"].sum()/1000000
        nfm_area[idx, 2] = selected_cats["imp_soil"].sum()/1000000
        
        # --- 6. Final re-simulation only: keep the SHETRAN flow file ---------
        # Move SHETRAN flow file
        # (renamed Chromosome_000.txt, Chromosome_001.txt, ... and moved to
        # 25_opt_outputs/opt_flows)
        if in_gen_num == 999:
            if idx < 10:
                new_name = 'Chromosome_00' + str(idx) + '.txt'
            elif idx > 9 and idx < 100:
                new_name = 'Chromosome_0' + str(idx) + '.txt'
            else:
                new_name = 'Chromosome_' + str(idx) + '.txt'
            
            # file name only (split on Windows "\" separators); found because
            # the working directory is the root folder
            old_name = in_she_output_path.split('\\')[-1]
            
            os.rename(old_name, new_name)
            
            shutil.move(new_name, opt_flows)
        
    # column 3 = total NFM area of each solution
    nfm_area[:,3] = np.sum(nfm_area, axis = 1)
    # nfm_area = nfm_area[:,3][:, np.newaxis] # convert a column vector to 2D

    return nfm_area, damages_avoided_M

#-----------------------------------------------------------------------------#
# STEP 9: Simulate and plot the initial population (generation 0)
#-----------------------------------------------------------------------------#
# i_chrom_cost: NFM areas (pop_size, 4); i_chrom_damage: damages avoided
# (pop_size, 1). This runs SHETRAN pop_size times.
i_chrom_cost, i_chrom_damage = shetran_sim(0,
    i_population, nfm_types_count, org_pet_csv_path,
    she_pet_csv_path, storage_asc_path,
    she_storage_asc_path, woodland_asc_path,
    she_woodland_asc_path,imp_soil_asc_path,
    org_soil_asc_path, she_soil_asc_path,
    cat_shp_path, exe_path, root, she_output_path,
    baseline_growth_curves, max_NFM_growth_curves,
    baseline_event, max_NFM_event)

# read map shapefile for map
subcat_smooth = gpd.read_file(subcat_smooth_shp_path)

# field of the map polygons that matches the gene number
join_column = 'opt_id'

# For the map: percentage of solutions that put NFM in each sub-catchment,
# joined to the sub-catchment polygons
[shape_subcat_joined, shape_subcat_contribution_column] = (
        rf.catch_contribution(subcat_smooth,join_column, chrom_len,
                          i_population, i_population.shape[0]))

# Figure settings used for every generation's figure
# USER INPUT: style, titles, colours and especially the axis limits
# (plot_x_limit, plot_y_limit) to suit your results
# fig_background = 'dark_background'
fig_background = 'default'
map_title = "Catchment contribution"
map_colour = "cividis_r"
map_legend_ticks = np.array([10,30,50,70,90])
map_legend_label = 'Contribution (%)'
plot_title = "Initial popluation (random)"
plot_legend_series = "Evolving solution"
plot_legend_series_1 = "Good solution"
plot_legend_series_2 = "Evolving solution"
plot_x_limit = [-1.5, 50]
plot_y_limit = [-2, 60]
plot_x_axis_label = "Total NFM area (km²)"
plot_y_axis_label = "Damages avoided (£ million)"
save_file = os.path.join(root,'25_opt_outputs', 'Generation_No_0')

# Left panel: total NFM area vs damages avoided; right panel: the map.
# Saved as 25_opt_outputs/Generation_No_0.png
rf.scatter_plot_map_plot(fig_background, plot_title, i_chrom_cost[:,3],
        i_chrom_damage,
        plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_subcat_joined, shape_subcat_contribution_column,
        save_file)

#-----------------------------------------------------------------------------#
# STEP 10: Records used to export results
#-----------------------------------------------------------------------------#
## These variables will be used to keep record of all unique simulated chroms
# g_counter: generation number of every simulated chromosome (column
# vector); starts with pop_size zeros for the initial population.
# (s_g_counter is not used further below.)
onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
onetime_counter[:,0] = 0
g_counter = copy.deepcopy(onetime_counter)
s_g_counter = copy.deepcopy(onetime_counter)
# simulated_*: every chromosome simulated so far and its objectives. Offspring
# are checked against simulated_population so that no solution is simulated
# twice.
simulated_population = copy.deepcopy(i_population)
simulated_chrom_cost = copy.deepcopy(i_chrom_cost)
simulated_chrom_damage = copy.deepcopy(i_chrom_damage)

## Labels to export data
# Columns of the exported CSV files: Generation, Catchment_0 ...
# Catchment_<chrom_len-1> (the genes), the four NFM areas (km²) and
# Damage_avoided (£ million)
exp_labels = [None]*(chrom_len+6)
for i in range(len((exp_labels))):
    if i > 0 and i <chrom_len+1:
        exp_labels[i] = "Catchment_" + str(i-1)
    elif i == 0:
        exp_labels[i] = "Generation"
    elif i == chrom_len+1:
        exp_labels[i] = "RAFs_total_area"
    elif i == chrom_len+2:
        exp_labels[i] = "Woodland_total_area"
    elif i == chrom_len+3:
        exp_labels[i] = "Imp_grassland_total_area"
    elif i == chrom_len+4:
        exp_labels[i] = "All_total_area"
    elif i == chrom_len+5:
        exp_labels[i] = "Damage_avoided"
        
## These variables will be used to keep each generation objectives records
## Store in list
gen_population = copy.deepcopy(i_population)
gen_chrom_cost = copy.deepcopy(i_chrom_cost)
gen_chrom_damage = copy.deepcopy(i_chrom_damage)

# Per-generation statistics, keyed by generation number (kept in memory
# only; not exported): time each generation finished, number of new
# offspring, number of offspring that survived selection, size of the first
# front and number of offspring-creation attempts (gen_dup_count)
gen_time = {}
gen_offspring_count = {}
gen_offspring_sustained_count = {}
gen_front1_count = {}
gen_dup_count = {}

## parent population, chrom_cost, expo_high and expo_medium
## will change in each iteration
# (here: the population, its NFM areas and its damages avoided)
p_population = copy.deepcopy(i_population)
p_chrom_cost = copy.deepcopy(i_chrom_cost)
p_chrom_damage = copy.deepcopy(i_chrom_damage)

#-----------------------------------------------------------------------------#
# STEP 11: Generation loop
#-----------------------------------------------------------------------------#
# USER INPUT: number of generations, range(1, 101) = generations 1 to 100
#-------------------------------------#
""" Generation loop starts from here"""
#-------------------------------------#
# generation = 2
for generation in range(1,101):

    # --- 11a. Rank the parents ------------------------------------------------
    print('\n''Gen.{0}: Generating offspring population'.format(generation))
    ## Making pair of objectives
    ## Obj1 size = pop_size x 1, obj2 size = pop_size x 1
    # Objective 1 = total NFM area (column 3), objective 2 = minus damages
    # avoided, so that both are minimised
    p_chroms_obj_record = np.concatenate((p_chrom_cost[:,3][:,np.newaxis]
                                          ,-p_chrom_damage), axis=1)
    
    ## To rank the individual (Method: Dominance Depth)
    # p_front: {front number: [row indices]}; front 0 = non-dominated
    p_front = rf.non_dominated_sorting(pop_size,p_chroms_obj_record)
    
    ## To keep diversity (Method: Normalised Manhattan Distance)
    # p_distance: {front number: {row index: crowding distance}}
    p_distance = rf.calculate_crowding_distance(p_front,p_chroms_obj_record)
    
    ## Sorting population based on fitness (front rank & crowding distance)
    # sorted_fitness: row indices of p_population, fittest first
    sorted_fitness = np.array(rf.fitness_sort(p_distance, pop_size))
    
    # --- 11b. Create new offspring ------------------------------------------
    ## Generating offsprings
    offspring = np.empty((0,chrom_len)).astype(int)
    
    # Keep creating pairs of children until pop_size new, not yet simulated
    # offspring exist, or until 5000 attempts (safety limit for when few new
    # solutions can be found)
    print('Offspring creation loop counter')
    c = 0 # This is to keep while loop definate
    while len(offspring) < pop_size and c < 5000:        
        ## Creating parents
        # binary tournament: each call returns the fitter of two random rows

        parent_1 = rf.fitter_parent(sorted_fitness, pop_size)
        parent_2 = rf.fitter_parent(sorted_fitness, pop_size)
        ## Checking if duplication
        while parent_1 == parent_2:
            parent_2 = rf.fitter_parent(sorted_fitness, pop_size)
        
        ## creating offspring using crossover operator
        # single-point crossover; cut points from 1 to chrom_len-2, so each
        # child always gets at least one gene from each parent
        min_idx = 1     # These will provide random index for cross over
        max_idx = chrom_len-1
        # np.random.randint(1,79) will exclude 79 (upper limit) and will 
        # give values from 1 to 78
        [child_1, child_2] = rf.crossover_random_single_point_swap(
                                        p_population, parent_1, parent_2,
                                        min_idx, max_idx)
        
        ## Introducing diversity in offspring using mutation operator
        # USER INPUT: mutation probability (chance that one random gene of
        # each child is flipped)
        p = 0.4 # probability for mutation
        m_idx_range = chrom_len    # index for mutation
        # np.random.randint(80) will exclude 80 (upper bound) and will give 
        # values from 0 to 79
        [offspring_1_c, offspring_2_c] = rf.mutation_random_bitflip(
                                            child_1, child_2, chrom_len,
                                            p, m_idx_range)
        # 1-D children -> rows of shape (1, chrom_len)
        offspring_1 = offspring_1_c.reshape(1,len(offspring_1_c))
        offspring_2 = offspring_2_c.reshape(1,len(offspring_2_c))
        
        # Add the two children to the offspring list. Earlier offspring that
        # are identical to either new child are removed first, so the list
        # holds no duplicates.
        if len(offspring) > 0:
            a = []
            dup_count = 0
            for i in range(len(offspring)):
                if ((np.all(offspring[i] == offspring_1) == True) or
                    (np.all(offspring[i] == offspring_2) == True)):
                    a.append(i)
                    dup_count += 1
            offspring = np.delete(offspring, a, 0)
            offspring = np.concatenate((offspring, 
                                    offspring_1, offspring_2), axis = 0) 
        else:
             offspring = np.concatenate((offspring, 
                                    offspring_1, offspring_2), axis = 0)
        
        # Drop offspring that have already been simulated in any generation
        offspring = rf.remove_duplicate_different_population(
                         offspring, simulated_population)
        
        print(c+1)
        c = c + 1
    gen_dup_count[generation] = c
    
    # Each pass adds two children, so there can be pop_size + 1; keep at most
    # pop_size. If no new offspring could be found, the run stops here.
    if len(offspring) > pop_size:
        offspring = offspring[0:pop_size]
        print ('\n''Gen.{0}: {1} new offspring found'
               .format(generation, len(offspring)))
    elif len(offspring) > 0 and len(offspring) < pop_size:
        print ('\n''Gen.{0}: Only {1} new offspring found'
               .format(generation, len(offspring)))
    elif len(offspring) == 0 :
        print ('\n''Gen.{0}: Could not find new offspring'
               .format(generation))
        sys.exit(0)
    else:
        print ('\n''Gen.{0}: {1} new offspring found'
               .format(generation, len(offspring)))        
    
    gen_offspring_count[generation] = len(offspring)       
    # --- 11c. Simulate the offspring ----------------------------------------
    ## CityCAT simulation (flood modelling) and buildings exposure calcuation
    ## for offspring population
    # (in RAINS: SHETRAN simulation and damages avoided, see shetran_sim)
    print('\n''Gen.{0}: Simulating offspring population'.format(generation))
    
    o_chrom_cost, o_chrom_damage = shetran_sim (generation,
        offspring, nfm_types_count, org_pet_csv_path,
        she_pet_csv_path, storage_asc_path,
        she_storage_asc_path, woodland_asc_path,
        she_woodland_asc_path,imp_soil_asc_path,
        org_soil_asc_path, she_soil_asc_path,
        cat_shp_path, exe_path, root, she_output_path,
        baseline_growth_curves, max_NFM_growth_curves,
        baseline_event, max_NFM_event)
        
    # --- 11d. Record and export every simulated solution --------------------
    ## Saving unique chroms(individuals) created in each generation
    simulated_population = np.concatenate((simulated_population, 
                                           offspring), axis=0)
    simulated_chrom_cost = np.concatenate((simulated_chrom_cost, 
                                           o_chrom_cost), axis=0)
    simulated_chrom_damage = np.concatenate((simulated_chrom_damage, 
                                           o_chrom_damage), axis=0)    
    
    ## Important Note: simulated population and its objectives only represent
    ## offspring created in every generation. Don't mix it with generation-
    ## wise best population.
    
    ## Export simulated data
    # Add one generation number per new offspring to g_counter, then write
    # every simulated solution so far (one row each) to simulated_data.csv;
    # the file is rewritten every generation.
    onetime_counter = np.zeros(len(offspring)).astype(int)[:,np.newaxis]
    onetime_counter[:,0] = generation
    g_counter = np.concatenate((g_counter, onetime_counter), axis=0)
    simulated_output = np.empty((0,chrom_len+6))
    simulated_output = np.concatenate((g_counter,
                                       simulated_population, 
                                       simulated_chrom_cost, 
                                       simulated_chrom_damage),
                                       axis=1)
 
    simulated_df = pd.DataFrame(simulated_output, columns = exp_labels)
    simulated_df.to_csv(os.path.join(root,'25_opt_outputs',
                                     'simulated_data.csv'), index_label='SN')
    
    # --- 11e. Combine parents and offspring and keep the best pop_size ----
    ## Making pair of offspring objectives   
    o_chroms_obj_record = np.concatenate((o_chrom_cost[:,3][:,np.newaxis],
                                          -o_chrom_damage), axis=1)
    
    ## Combining parents objective & offspring objectives
    # rows: parents first, then offspring
    comb_chroms_obj_record = np.concatenate(
                            (p_chroms_obj_record,o_chroms_obj_record), axis=0)
    
    ## For code check point
    ## Checking duplicate records in combined objective list
    # Solutions with exactly the same pair of objective values are reduced to
    # one (the first); dup_idx_obj lists the removed rows. The same rows are
    # removed from the chromosomes and objective arrays below, so all stay
    # aligned.
    [comb_chroms_obj_record_uniq, dup_idx_obj] = rf.remove_duplicate_list(
                                            comb_chroms_obj_record)
    
    
    ## Joining parents and offspring individuals (chromosomes)
    comb_population = np.concatenate((p_population, offspring), axis=0)
  
    ## Takes indices of duplicate objectives and remove chromosomes of those
    ## indices
    comb_population_uniq_obj = rf.remove_same_objectives_population(
                                            comb_population, dup_idx_obj)
    
    # number of unique combined solutions to rank
    comb_pop_size = len(comb_population_uniq_obj)
    ## Ranking the individuals from combined population
    comb_front = rf.non_dominated_sorting(comb_pop_size, 
                                   comb_chroms_obj_record_uniq)
 
    
    ## Calculating crowding distance for individuals from combined population
    comb_distance = rf.calculate_crowding_distance(comb_front, 
                                            comb_chroms_obj_record_uniq)
    
    ## Sorting combined population based on fitness (ranking and
    ## crowding distance)
    comb_population_fitness_sort = rf.fitness_sort(comb_distance, comb_pop_size)
    
    ## Selecting pop_size number of fittest individuals. As individuals are
    ## already sorted so selecting first pop_size number of individuals        
    # select_fittest: row indices (into the de-duplicated combined arrays)
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])
    
    # Full NFM-area and damage arrays of the combined population, with the
    # same duplicate rows removed as above
    ## Joined cost objective of parents and offspring population
    comb_chrom_cost = np.concatenate((p_chrom_cost, o_chrom_cost), axis=0)
    comb_chrom_cost_uniq_obj = np.delete(comb_chrom_cost,dup_idx_obj, 0)
       
    comb_chrom_damage = np.concatenate((p_chrom_damage, o_chrom_damage), axis=0)
    comb_chrom_damage_uniq_obj = np.delete(comb_chrom_damage,dup_idx_obj, 0)
    
    ## Selecting objectives for fittest individuals (chromosomes)
    f_chrom_cost = copy.deepcopy(comb_chrom_cost_uniq_obj[select_fittest])
    f_chrom_damage = copy.deepcopy(comb_chrom_damage_uniq_obj[select_fittest])
    
   
    ## Selecting the fittest individuals to create new population
    f_population = copy.deepcopy(comb_population_uniq_obj[select_fittest])
    
    # --- 11f. Plot the new population ---------------------------------------
    # Saved as 25_opt_outputs/Generation_No_<generation>.png
    [shape_subcat_joined, shape_subcat_contribution_column] = (
            rf.catch_contribution(subcat_smooth,join_column, chrom_len,
                              f_population, f_population.shape[0]))

    plot_title = "Generation no " + str(generation)
    save_file = os.path.join(root,'25_opt_outputs',
                             'Generation_No_' + str(generation))

    rf.scatter_plot_map_plot(fig_background, plot_title, f_chrom_cost[:,3],
            f_chrom_damage,
            plot_legend_series, plot_x_limit, plot_y_limit,
            plot_x_axis_label, plot_y_axis_label, map_title,
            map_colour, map_legend_ticks, map_legend_label,
            shape_subcat_joined, shape_subcat_contribution_column,
            save_file)
    # --- 11g. The selected solutions become the next parents ---------------
    ## Making a copy of previous population
    old_population = copy.deepcopy(p_population)

    ## Separating new created chromosomes (individuals) and 
    ## old repeated chromosomes by comparing new_population 
    ## with old_population
    # (used only to report how many parents survived and how many offspring
    # entered the population)
    # NOTE (known issue): separate_new_old fails (UnboundLocalError) in a
    # generation where no parent survives into the new population.
    [new_chroms, old_chroms, old_chroms_index] = rf.separate_new_old(
                                            f_population,old_population)
    print('\n''Gen.{0}: New population contains {1} parents & {2} offspring'
          .format(generation, len(old_chroms), len(new_chroms)))
    
    ## Delete old population
    del p_population, p_chrom_cost, p_chrom_damage     
    
    ## New population
    p_population = copy.deepcopy(f_population)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_chrom_damage = copy.deepcopy(f_chrom_damage)
    
    # --- 11h. Record and export the population of every generation ---------
    # gen_*: the populations of generations 0, 1, 2, ... stacked row-wise
    gen_population = np.concatenate((gen_population, 
                                           p_population), axis=0)
    gen_chrom_cost = np.concatenate((gen_chrom_cost, 
                                           p_chrom_cost), axis=0)
    gen_chrom_damage = np.concatenate((gen_chrom_damage, 
                                           p_chrom_damage), axis=0)
    
    
    gen_offspring_sustained_count[generation] = len(new_chroms)
    gen_front1_count[generation] = len(comb_front[0])
    gen_time[generation] = pd.Timestamp.now()
    
    ## Export generation data
    ## g_counter is already populated when simulated population was exported
    # generation_data.csv: every generation's population (rewritten each
    # generation).
    # NOTE (known issue): g_counter has one row per SIMULATED chromosome,
    # while gen_population has pop_size rows per generation. If a generation
    # found fewer than pop_size new offspring, the row counts differ and this
    # concatenation fails.
    generation_output = np.empty((0,chrom_len+6))
    generation_output = np.concatenate((g_counter,
                                       gen_population, 
                                       gen_chrom_cost, 
                                       gen_chrom_damage),
                                       axis=1)
 
    generation_df = pd.DataFrame(generation_output, columns = exp_labels)
    generation_df.to_csv(os.path.join(root,'25_opt_outputs',
                                      'generation_data.csv'), index_label='SN')
    
    

#-----------------------------------------------------------------------------#
# STEP 12: Pareto-optimal solutions of the last generation
#-----------------------------------------------------------------------------#
# The final population is ranked once more and only its first front (the
# non-dominated, Pareto-optimal solutions) is kept as the result.

# to make pairs of offspring objectives   
# (here: the objectives of the final population)
opt_chroms_objs = np.concatenate((p_chrom_cost[:,3][:,np.newaxis],
                                  -p_chrom_damage), axis=1)

# to get non-dominated solutions (first front)
# opt_front: row indices of the Pareto-optimal solutions
opt_front = rf.non_dominated_sorting(pop_size, 
                               opt_chroms_objs)[0]

# popluation that provides optimal solutions
opt_population = p_population[opt_front]

# optimal cost
opt_chrom_cost = p_chrom_cost[opt_front]

# optimal expected annual damage
# (in RAINS: damages avoided, £ million)
opt_chrom_damage = p_chrom_damage[opt_front]


[shape_subcat_joined, shape_subcat_contribution_column] = (
        rf.catch_contribution(subcat_smooth,join_column, chrom_len,
                          opt_population, opt_population.shape[0]))
# plot title
opt_plot_title = 'Generation no ' + str(generation) + ' optimal'

# plot legend
opt_plot_legend_series = 'Optimal solution'

# to save file
opt_save_file = os.path.join(root,'25_opt_outputs',
                         'Generation_No_' + str(generation) + '_optimal')

# Saved as 25_opt_outputs/Generation_No_<last generation>_optimal.png
# NOTE (known issue): plot_title is passed here instead of opt_plot_title, so
# the title reads "Generation no <n>" without "optimal".
rf.scatter_plot_map_plot(fig_background, plot_title, opt_chrom_cost[:,3],
        opt_chrom_damage,
        opt_plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_subcat_joined, shape_subcat_contribution_column,
        opt_save_file)

## Export generation data
# final_generation_optimal_data.csv: the Pareto-optimal solutions, with the
# same columns as the other CSV files.
# NOTE (known issue): opt_onetime_counter has pop_size rows but
# opt_population only has the Pareto-front rows. If the final front holds
# fewer than pop_size solutions this concatenation fails, and STEPS 13-14
# are not reached.
opt_onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
opt_onetime_counter[:,0] = generation
generation_output = np.empty((0,chrom_len+6))
generation_output = np.concatenate((opt_onetime_counter,
                                   opt_population, 
                                   opt_chrom_cost, 
                                   opt_chrom_damage),
                                   axis=1)
 
opt_generation_df = pd.DataFrame(generation_output, columns = exp_labels)
opt_generation_df.to_csv(os.path.join(root,'25_opt_outputs',
                                  'final_generation_optimal_data.csv'), index_label='SN')



#-----------------------------------------------------------------------------#
# STEP 13: Sub-catchment contribution to the optimal solutions
#-----------------------------------------------------------------------------#
# For each sub-catchment (gene): how many Pareto-optimal solutions put NFM
# there, and what percentage that is (stored as a whole number). The "BGI"
# wording in the comments below is carried over from earlier code; here
# each row is a sub-catchment. Written to
# catchments_contribution_to_optimised_solutions.csv.

# Catchments contribution in Pareto optimal front
cat_contribution = np.zeros((chrom_len, 3), dtype=int)

for i in range(chrom_len):
    
    # BGI id
    cat_contribution[i,0] = i   
    
    # BGI contribution
    cat_contribution[i,1] = sum(opt_population[:,i]) # zone contribution
    
    # BGI contribution in percentage
    cat_contribution[i,2] = (100*cat_contribution[i,1])/(len(opt_population))

del i
    
# to create a data frame from an array
cont_df = pd.DataFrame(cat_contribution)

# to assign names to columns
cont_df.columns = ['cat_id', 'count', 'percent_count']

# to export BGI contribution data as a CSV file
cont_df.to_csv(os.path.join(root,'25_opt_outputs',
    'catchments_contribution_to_optimised_solutions.csv'), index=False)

#-----------------------------------------------------------------------------#
# STEP 14: Re-simulate the optimal solutions to keep their SHETRAN flows
#-----------------------------------------------------------------------------#
# Generation number 999 tells shetran_sim to save each solution's discharge
# file as 25_opt_outputs/opt_flows/Chromosome_XXX.txt, where XXX is the row
# of the solution in opt_population (and in final_generation_optimal_data.csv).
# The returned objectives (opt_chrom_cost2, opt_chrom_damage2) are not used
# further.

# Get all SHETRAN flow files for the optimised generation
opt_chrom_cost2, opt_chrom_damage2 = shetran_sim (999,
    opt_population, nfm_types_count, org_pet_csv_path,
    she_pet_csv_path, storage_asc_path,
    she_storage_asc_path, woodland_asc_path,
    she_woodland_asc_path,imp_soil_asc_path,
    org_soil_asc_path, she_soil_asc_path,
    cat_shp_path, exe_path, root, she_output_path,
    baseline_growth_curves, max_NFM_growth_curves,
    baseline_event, max_NFM_event)

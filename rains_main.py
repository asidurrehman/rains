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
import importlib.util

# Path to function files
root = os.path.join('C:', os.path.sep,'z_nfm_opt')
os.chdir(root)

shetran_func_path = os.path.join(root,'01_codes',
                                 'SHETRAN_Post_Simulation_Functions.py')
rains_func_path = os.path.join(root,'01_codes',
                               'rains_all_functions.py')
# Load shetran_func
spec1 = importlib.util.spec_from_file_location("shetran_func", shetran_func_path)
sf = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(sf)

# Load rains_func
spec2 = importlib.util.spec_from_file_location("rains_func", rains_func_path)
rf = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rf)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Select which NFM features to consider
# The line of code below can be changed. Delete layers as applicable from the list ['Extra_Pond_Capacity_RP100_10', 'WfW', 'Losses', 'ImpGrass'] below
NFM_features_to_include = ['Extra_Pond_Capacity_RP100_10', 'WfW', 'ImpGrass']

# Read in the NFM spatial data
gdf = gpd.read_file(os.path.join(root,'04_irwell_notebook_data',
                                 "a0000002b.gdbtable"))
gdf['WfW'] = gdf['Workshop_WfW_Area_km2']*10000
gdf['Losses'] = gdf['Workshop_CombLoss_Area_km2']*10000
gdf['ImpGrass'] = gdf['Workshop_ImpGrass_Area_km2']*10000
gdf['total_NFM_storage'] = gdf[NFM_features_to_include].sum(axis=1)

# Find unique communities at risk in the Jacobs storage tool
file = open(os.path.join(root,'04_irwell_notebook_data',
                         'Calculated_Hydrographs.p'),'rb')
NFM_storage = pickle.load(file)
file.close()
COR = NFM_storage.loc[(NFM_storage.RP == 'RP2')&(NFM_storage.timestep == 0.1)]
joined_gdf = gpd.sjoin(COR, gdf, how="inner", predicate="within")

# Read in the FEH data to find community at risk catchment area
feh_data = pd.read_excel(os.path.join(root,'04_irwell_notebook_data',
                                      'FEH_Data.xlsx'))

# Read in the communities at risk file names and associated Jacobs storage tool unique IDs and SHETRAN river links
CaR_ID_df = pd.read_csv(os.path.join(root,'04_irwell_notebook_data',
                                     'CaR_name_ID_Final.csv'))
CaR_ID_df.head()
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

damages_df = pd.read_pickle(os.path.join(root,'04_irwell_notebook_data',
                                         'vol_damage_curves.pkl'))
damages_df_index = np.around(np.linspace(0.01, 1, 100), 2) 
damages_df.set_index(damages_df_index, inplace = True)
damages_df.loc[0] = [0 for CaR in CaR_names]
damages_df = damages_df.sort_index()

# Create new datasets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Calculate volumes associated with the design hydrographs

vols_df = pd.DataFrame({"RP":[5,20,50,100,200]})

for community_name in CaR_names:
    
    community_ids = [int(a) for a in CaR_ID_df.loc[community_name].jacobs_storage_tool_ID.split("/")]
    
    RP5_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP5')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP20_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP20')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP50_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP50')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP100_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP100')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    RP200_hydrograph = NFM_storage.loc[(NFM_storage.Unique_ID.isin(community_ids))&(NFM_storage.RP == 'RP200')].groupby('timestep')['QT'].sum().to_numpy().sum()*(0.1*3600)
    vols_df[CaR_ID_df.loc[community_name].SHETRAN_id] = [RP5_hydrograph, RP20_hydrograph, RP50_hydrograph, RP100_hydrograph, RP200_hydrograph]

vols_df.set_index("RP", inplace=True)

# get reference information about the max NFM storage

max_vol_df = pd.DataFrame(index=['community_area', 'wfd_area', 'wfd_max_storage', 'community_max_storage'] )

for community_name in CaR_names:
    community_ids = [int(a) for a in CaR_ID_df.loc[community_name].jacobs_storage_tool_ID.split("/")]
    community_X = int(joined_gdf.loc[joined_gdf.Unique_ID == community_ids[0]].SnappedX.values[0])
    community_Y = int(joined_gdf.loc[joined_gdf.Unique_ID == community_ids[0]].SnappedY.values[0])
    community_area = feh_data.loc[(feh_data.X == community_X)&(feh_data.Y == community_Y)].AREA.values[0]
    wfd_max_storage = joined_gdf.loc[joined_gdf.Unique_ID.isin(community_ids)].total_NFM_storage.sum() # this will be found from the JBA map
    wfd_area = joined_gdf.loc[joined_gdf.Unique_ID.isin(community_ids)].Area_km2.sum() #km2
    # Find the maximum NFM storage available to the community at risk
    community_max_storage = (community_area/wfd_area) * wfd_max_storage
    max_vol_df[community_name] = [community_area, wfd_area, wfd_max_storage, community_max_storage]

# Find the RP of the modelled event
event_date = '1983-10-09' # change as appropriate, this is the date of the peak flow of the event modelled in the optimisation. 
baseline_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_baseline_long.txt')
baseline_df = rf.read_shetran(baseline_path , "1980-01-01", "2014-12-30") # change path and dates as appropriate
event_flow = baseline_df.outlet.loc[event_date] # check that this is returning a single value
annual_maxima = baseline_df.resample("YE").max().dropna()
shape, loc, scale = genextreme.fit(annual_maxima.outlet.values)
probability = genextreme.cdf(event_flow, c=shape, loc=loc, scale=scale)
RP = 1/(1-probability) # e.g. RP = 3.2 at a flow of 42.8 cumecs
# print('Event RP = ', RP)

baseline_growth_curves = rf.get_growth_curves(baseline_path, "1980-01-01", "2014-12-30", RP) # change path and dates as appropriate
max_NFM_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_max_nfm_long.txt')
max_NFM_growth_curves = rf.get_growth_curves(max_NFM_path, "1980-01-01", "2014-12-30", RP) # change path and dates as appropriate

baseline_event_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_baseline_event.txt')
baseline_event = rf.read_shetran(baseline_event_path, "1983-10-03", "1983-10-12").loc[[event_date]] # change path and dates as appropriate
max_NFM_event_path = os.path.join(root,'04_irwell_notebook_data',
                    'Irwell_to_Bury_Bridge_200m_discharge_max_nfm_event.txt')
max_NFM_event = rf.read_shetran(max_NFM_event_path, "1983-10-03", "1983-10-12").loc[[event_date]] # change path and dates as appropriate


# Paths
exe_path = os.path.join(root, '20_shetran_easy_setup_snow')
dmg_cal_path = os.path.join(root, '04_irwell_notebook_data')

# Paths for shapefiles used for visulaisation
subcat_smooth_shp_path = os.path.join(root, '15_shapefiles',
                                      'ui_shetran_sub_catchments_smooth.shp')
cat_smooth_shp_path = os.path.join(root, '15_shapefiles',
                                      'ui_shetran_atchments_smooth.shp')

# Orginal PET csv file path
org_pet_csv_path = os.path.join(root, '05_inputs_backup',
                                'Irwell_to_Bury_Bridge_200m_PET.csv')
she_pet_csv_path = os.path.join(root, 'Irwell_to_Bury_Bridge_200m_PET.csv')

# NFM input file paths
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

she_output_path = os.path.join(root,
        'output_Irwell_to_Bury_Bridge_200m_discharge_sim_regulartimestep.txt')
opt_graph_path = os.path.join(root,'25_output_graphs')

# List of file paths

other_paths = [org_pet_csv_path, cat_shp_path, root, exe_path]

# Reads sub-catchment shapefile
cat_shp = gpd.read_file(cat_shp_path)

nfm_types_count = 3
pop_size = 100
chrom_len = cat_shp.shape[0]

i_population = rf.initial_pop(pop_size, chrom_len)
# i_population = i_population[5,:][np.newaxis,:]

def shetran_sim (in_gen_num, in_population, in_nfm_types_count,
                 in_org_pet_csv_path,
                 in_she_pet_csv_path, in_storage_asc_path,
                 in_she_storage_asc_path, in_woodland_asc_path,
                  in_she_woodland_asc_path,in_imp_soil_asc_path,
                  in_org_soil_asc_path, in_she_soil_asc_path,
                  in_cat_shp_path, in_exe_path, in_root, in_she_output_path,
                  in_baseline_growth_curves, in_max_NFM_growth_curves,
                  in_baseline_event, in_max_NFM_event):
    
    # To save NFM areas individually and total
    nfm_area = np.zeros([in_population.shape[0], in_nfm_types_count+1])
    # nfm_flow = np.zeros([in_population.shape[0], 1])
    damages_avoided_M = np.zeros([in_population.shape[0], 1])
    
    # To save all SHETRAN flow files for optimised generation
    if in_gen_num == 999:
        opt_flows = os.path.join(root,'25_opt_outputs', 'opt_flows')
    
        if not os.path.exists(opt_flows):
            os.mkdir(opt_flows)
        else:
            os.rmdir(opt_flows)
            os.mkdir(opt_flows)
    
    for idx, chrom in enumerate(in_population):
        print('\nGeneration {}: solution {} of {}'.format(
            in_gen_num, idx+1, in_population.shape[0]))
        
        file_paths = [in_she_pet_csv_path, in_she_storage_asc_path,
                          in_she_woodland_asc_path,in_she_soil_asc_path]    
        # Loop through the list and remove each file if it exists
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Place original PET csv file
        shutil.copy(in_org_pet_csv_path, in_she_pet_csv_path)
        
        # Prepare NFM datasets for SHETRAN
        # Get catchments ID for no-NFM catchment
        no_nfm_cats = np.where(chrom == 0)[0]
        cat_shp = gpd.read_file(in_cat_shp_path)
        if len(no_nfm_cats) != cat_shp.shape[0]:
            # Set no-NFM catchment to zero in maximum NFM files
            (rf.nfm_selection (no_nfm_cats, in_cat_shp_path,
                           in_storage_asc_path, in_she_storage_asc_path, True))
            rf.nfm_selection (no_nfm_cats, in_cat_shp_path, in_woodland_asc_path,
                           in_she_woodland_asc_path, True)
            imp_soil_sel = rf.nfm_selection (no_nfm_cats, in_cat_shp_path,
                                          in_imp_soil_asc_path, None, False)
            rf.soil_update (in_org_soil_asc_path, imp_soil_sel, in_she_soil_asc_path)
        else:
           shutil.copy(in_org_soil_asc_path, in_she_soil_asc_path)
        
        subprocess.call([f'{in_exe_path}/shetran-prepare-snow-NFM.exe',
                         'Irwell_to_Bury_Bridge_200m_LibraryFile.xml'])
        
        # Set multi-point flow extraction file name on rundata file
        sf.edit_RunData_line(rundata_filepath=f'{in_root}/rundata_Irwell_to_Bury_Bridge_200m.txt',
                     element_number=47, entry='Irwell_to_Bury_Bridge_200m_discharge_points.txt')
        # Set the initial conditions file name on rundata file
        sf.edit_RunData_line(rundata_filepath=f'{in_root}/rundata_Irwell_to_Bury_Bridge_200m.txt',
                     element_number=29, entry='input_Irwell_to_Bury_Bridge_200m_vsi.txt')
        
        # Set initial conditions option in vsd file
        rf.initial_conditions_setting(in_root)
        
        sf.run_SHETRAN_skip_pause(exe_filepath=f'{in_exe_path}/shetran.exe', 
        rundata_filepath=f'{in_root}/rundata_Irwell_to_Bury_Bridge_200m.txt', 
        print_timestep=False, force_continue=True)
        
        ## Extract specific flow value at outlet (will not work for multi-point flows)
        # nfm_flow[idx] = (pd.read_csv(in_she_output_path, sep='\s+', skiprows=1,
        #                        header=None, index_col=False).to_numpy()[26])
                 
        # calculate damages avoided
        # get_event diff
        event_date = '1983-10-09' # Whatever the date is of the event you are simulating
        # event_RP = RP 
        scenario_event = rf.read_shetran(in_she_output_path, "1983-10-03", "1983-10-12").loc[[event_date]] # change these to match your simulation
        scenario_adjust = (scenario_event-in_max_NFM_event)/(in_baseline_event-in_max_NFM_event) 
        scenario_growth_curves = (in_baseline_growth_curves - in_max_NFM_growth_curves).mul(scenario_adjust.iloc[0], axis=1) + in_max_NFM_growth_curves
        reduction_factors = 1-(scenario_growth_curves/in_baseline_growth_curves)
        vols_attenuated = vols_df.mul(reduction_factors).max()

        total_damages_avoided = 0

        for CaR in CaR_names:
            shetran_id = CaR_ID_df.SHETRAN_id.loc[CaR]
            # vals = max_vol_df[CaR].values
            community_max_storage = max_vol_df[CaR].community_max_storage
            max_vol_attenuated = vols_attenuated[shetran_id]
            fraction_attenuated = round(max(min(max_vol_attenuated/
                                        community_max_storage, 1.0), 0), 2)
            if fraction_attenuated >=0:
                damages_avoided = damages_df[CaR].loc[fraction_attenuated]
            else:
                damages_avoided = 0

            total_damages_avoided += damages_avoided
            
            total_damages_avoided
        
        damages_avoided_M[idx] = round(total_damages_avoided/1000000,3)
                
        # Get catchment IDs for NFM-catchments and sum their area
        nfm_cats = np.where(chrom == 1)[0]
        # Filter the polygons whose opt_id is in the given list
        selected_cats = cat_shp[cat_shp['opt_id'].isin(nfm_cats)]
        nfm_area[idx, 0] = selected_cats["raf"].sum()/1000000
        nfm_area[idx, 1] = selected_cats["woodland"].sum()/1000000
        nfm_area[idx, 2] = selected_cats["imp_soil"].sum()/1000000
        
        # Move SHETRAN flow file
        if in_gen_num == 999:
            if idx < 10:
                new_name = 'Chromosome_00' + str(idx) + '.txt'
            elif idx > 9 and idx < 100:
                new_name = 'Chromosome_0' + str(idx) + '.txt'
            else:
                new_name = 'Chromosome_' + str(idx) + '.txt'
            
            old_name = in_she_output_path.split('\\')[-1]
            
            os.rename(old_name, new_name)
            
            shutil.move(new_name, opt_flows)
        
    nfm_area[:,3] = np.sum(nfm_area, axis = 1)
    # nfm_area = nfm_area[:,3][:, np.newaxis] # convert a column vector to 2D

    return nfm_area, damages_avoided_M

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

join_column = 'opt_id'

[shape_subcat_joined, shape_subcat_contribution_column] = (
        rf.catch_contribution(subcat_smooth,join_column, chrom_len,
                          i_population, i_population.shape[0]))

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

rf.scatter_plot_map_plot(fig_background, plot_title, i_chrom_cost[:,3],
        i_chrom_damage,
        plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_subcat_joined, shape_subcat_contribution_column,
        save_file)

## These variables will be used to keep record of all unique simulated chroms
onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
onetime_counter[:,0] = 0
g_counter = copy.deepcopy(onetime_counter)
s_g_counter = copy.deepcopy(onetime_counter)
simulated_population = copy.deepcopy(i_population)
simulated_chrom_cost = copy.deepcopy(i_chrom_cost)
simulated_chrom_damage = copy.deepcopy(i_chrom_damage)

## Labels to export data
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

gen_time = {}
gen_offspring_count = {}
gen_offspring_sustained_count = {}
gen_front1_count = {}
gen_dup_count = {}

## parent population, chrom_cost, expo_high and expo_medium
## will change in each iteration
p_population = copy.deepcopy(i_population)
p_chrom_cost = copy.deepcopy(i_chrom_cost)
p_chrom_damage = copy.deepcopy(i_chrom_damage)

#-------------------------------------#
""" Generation loop starts from here"""
#-------------------------------------#
# generation = 2
for generation in range(1,101):

    print('\n''Gen.{0}: Generating offspring population'.format(generation))
    ## Making pair of objectives
    ## Obj1 size = pop_size x 1, obj2 size = pop_size x 1
    p_chroms_obj_record = np.concatenate((p_chrom_cost[:,3][:,np.newaxis]
                                          ,-p_chrom_damage), axis=1)
    
    ## To rank the individual (Method: Dominance Depth)
    p_front = rf.non_dominated_sorting(pop_size,p_chroms_obj_record)
    
    ## To keep diversity (Method: Normalised Manhattan Distance)
    p_distance = rf.calculate_crowding_distance(p_front,p_chroms_obj_record)
    
    ## Sorting population based on fitness (front rank & crowding distance)
    sorted_fitness = np.array(rf.fitness_sort(p_distance, pop_size))
    
    ## Generating offsprings
    offspring = np.empty((0,chrom_len)).astype(int)
    
    print('Offspring creation loop counter')
    c = 0 # This is to keep while loop definate
    while len(offspring) < pop_size and c < 5000:        
        ## Creating parents

        parent_1 = rf.fitter_parent(sorted_fitness, pop_size)
        parent_2 = rf.fitter_parent(sorted_fitness, pop_size)
        ## Checking if duplication
        while parent_1 == parent_2:
            parent_2 = rf.fitter_parent(sorted_fitness, pop_size)
        
        ## creating offspring using crossover operator
        min_idx = 1     # These will provide random index for cross over
        max_idx = chrom_len-1
        # np.random.randint(1,79) will exclude 79 (upper limit) and will 
        # give values from 1 to 78
        [child_1, child_2] = rf.crossover_random_single_point_swap(
                                        p_population, parent_1, parent_2,
                                        min_idx, max_idx)
        
        ## Introducing diversity in offspring using mutation operator
        p = 0.4 # probability for mutation
        m_idx_range = chrom_len    # index for mutation
        # np.random.randint(80) will exclude 80 (upper bound) and will give 
        # values from 0 to 79
        [offspring_1_c, offspring_2_c] = rf.mutation_random_bitflip(
                                            child_1, child_2, chrom_len,
                                            p, m_idx_range)
        offspring_1 = offspring_1_c.reshape(1,len(offspring_1_c))
        offspring_2 = offspring_2_c.reshape(1,len(offspring_2_c))
        
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
        
        offspring = rf.remove_duplicate_different_population(
                         offspring, simulated_population)
        
        print(c+1)
        c = c + 1
    gen_dup_count[generation] = c
    
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
    ## CityCAT simulation (flood modelling) and buildings exposure calcuation
    ## for offspring population
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
    
    ## Making pair of offspring objectives   
    o_chroms_obj_record = np.concatenate((o_chrom_cost[:,3][:,np.newaxis],
                                          -o_chrom_damage), axis=1)
    
    ## Combining parents objective & offspring objectives
    comb_chroms_obj_record = np.concatenate(
                            (p_chroms_obj_record,o_chroms_obj_record), axis=0)
    
    ## For code check point
    ## Checking duplicate records in combined objective list
    [comb_chroms_obj_record_uniq, dup_idx_obj] = rf.remove_duplicate_list(
                                            comb_chroms_obj_record)
    
    
    ## Joining parents and offspring individuals (chromosomes)
    comb_population = np.concatenate((p_population, offspring), axis=0)
  
    ## Takes indices of duplicate objectives and remove chromosomes of those
    ## indices
    comb_population_uniq_obj = rf.remove_same_objectives_population(
                                            comb_population, dup_idx_obj)
    
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
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])
    
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
    ## Making a copy of previous population
    old_population = copy.deepcopy(p_population)

    ## Separating new created chromosomes (individuals) and 
    ## old repeated chromosomes by comparing new_population 
    ## with old_population
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
    generation_output = np.empty((0,chrom_len+6))
    generation_output = np.concatenate((g_counter,
                                       gen_population, 
                                       gen_chrom_cost, 
                                       gen_chrom_damage),
                                       axis=1)
 
    generation_df = pd.DataFrame(generation_output, columns = exp_labels)
    generation_df.to_csv(os.path.join(root,'25_opt_outputs',
                                      'generation_data.csv'), index_label='SN')
    
    

# to make pairs of offspring objectives   
opt_chroms_objs = np.concatenate((p_chrom_cost[:,3][:,np.newaxis],
                                  -p_chrom_damage), axis=1)

# to get non-dominated solutions (first front)
opt_front = rf.non_dominated_sorting(pop_size, 
                               opt_chroms_objs)[0]

# popluation that provides optimal solutions
opt_population = p_population[opt_front]

# optimal cost
opt_chrom_cost = p_chrom_cost[opt_front]

# optimal expected annual damage
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

rf.scatter_plot_map_plot(fig_background, plot_title, opt_chrom_cost[:,3],
        opt_chrom_damage,
        opt_plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_subcat_joined, shape_subcat_contribution_column,
        opt_save_file)

## Export generation data
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

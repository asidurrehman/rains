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
File name: rains_all_functions.py

@author: Asid Ur Rehman, Elizabeth Lewis, Ben Smith
Organisation: Newcastle Univeristy, Manchester University

About
------
This script file contains all function definations used to run
RAINS - a framework which integrates evolutionary multi-objective genetic
algorithm (NSGA-II) with SHETRAN to find optimal locations and sizes of NFM 
features for their cost-effective design.

"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import copy
from scipy.stats import genextreme


warnings.filterwarnings("ignore")

###############################################################################
""" Functions section starts from here """
###############################################################################

def read_shetran(filename, start_date, end_date):
    '''
    reads shetran output files for multiple river links

    Arguments:
    shetran file path, start date/end dat format "%Y-%m-%d"

    Returns:
    pandas dataframe of shetran simulated flows with riverlink names as column names and a datetime index
    '''
    
    with open(filename, "r") as f:
        f.readline()
        col_names = [int(i) for i in f.readline().rstrip()[39:].split()]
        col_names.insert(0, 'outlet')
    
    # Create a date range with daily frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Create an empty DataFrame with the date index
    
    df = pd.read_csv(filename, sep=r"\s+", skiprows=2, names=col_names)
    df.set_index(date_range, inplace=True)

    return(df)

def generate_growth_curve(amax, event_RP):
    rps = np.array([event_RP, 5, 20, 50, 100, 200]) # This is assuming that the modelled event has an RP of 3.2
    # Fit a Generalized Extreme Value (GEV) distribution to the data
    shape, loc, scale = genextreme.fit(amax)  # Negate values for maxima fitting
    probs = 1 - (1 / rps)  # Exceedance probability
    flow_values = genextreme.ppf(probs, shape, loc, scale)  # Invert negation

    return(flow_values)

def get_growth_curves(filename, start_date, end_date, event_RP):
    df = read_shetran(filename, start_date, end_date)
    annual_maxima = df.resample("YE").max().dropna()
    growth_curves = annual_maxima.apply(lambda col: generate_growth_curve(col, event_RP), axis=0)
    growth_curves["RP"] = [event_RP, 5, 20, 50, 100, 200] # This is assuming that the modelled event has an RP of 3.2
    growth_curves.set_index("RP", inplace=True)
    return(growth_curves)

def nfm_selection (in_cat_ids, in_cat_shp, in_nfm, out_nfm=None, save_nfm=True):
    """
    Selects the NFM features only for selected small sub-catchments
    
    Args:
        param1 (array of integers): Represents small catchment IDs which
        will be used to select the catchments
        param2 (shapefile): Polygon shapefile of catchments
        .
        .
        .
    
    Retruns:
        2D numpy array (int)
    
    Raises:
        ValueError: If out_nfm is not provided while save_nfm option is True
        
    """

    # Load the polygon shapefile
    gdf = gpd.read_file(in_cat_shp)
    
    # Open the ASCII raster and read its data and metadata
    with rasterio.open(in_nfm) as src:
        raster = src.read(1)
        transform = src.transform
        meta = src.meta.copy()
    
    # Assign the shapefile's coordinate system to the ASCII raster
    # (simple assignment)
    meta['crs'] = gdf.crs
    
    # Filter the polygons whose opt_id is in the given list
    selected_polygons = gdf[gdf['opt_id'].isin(in_cat_ids)]
    
    # Rasterize the selected polygons.
    # Each cell that falls within any of these polygons gets
    # a value of 1; others remain 0.
    mask = rasterize(
        [(geom, 1) for geom in selected_polygons.geometry],
        out_shape=raster.shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    # For cells covered by the polygons (mask==1), set the raster value to 0
    new_raster = np.where(mask == 1, 0, raster)
    
    # Update metadata for ASCII (AAIGrid) export; ensure the driver
    # and data type are set
    meta.update(driver='AAIGrid', dtype=raster.dtype, count=1)
    
    # SHETRAN does not need CRS
    meta['crs'] = None
    
    if save_nfm:
        if out_nfm is None:
            raise ValueError("Output filename 'out_nfm' must be " 
                             "provided when save_nfm is True.")
        else:
            # Write the modified raster to the output ASCII file
            with rasterio.open(out_nfm, 'w', **meta) as dst:
                dst.write(new_raster, 1)
    else:
           return new_raster
    
#-----------------------------------------------------------------------------
def soil_update (in_org_soil, in_imp_soil, out_soil):
    # Open the org_soil raster and read its data and metadata.
    with rasterio.open(in_org_soil) as soil_src:
        soil_data = soil_src.read(1)
        soil_meta = soil_src.meta.copy()
        soil_nodata = soil_src.nodata
    
    # Create a valid mask for the soil raster (ignore nodata)
    if soil_nodata is not None:
        valid_soil = soil_data != soil_nodata
        # Compute the maximum of valid soil values
        max_soil = soil_data[valid_soil].max()
    else:
        max_soil = soil_data.max()
    
    # Create nodata masks for each raster
    mask_soil = ((soil_data == soil_nodata) if soil_nodata is not None
                 else np.zeros_like(soil_data, dtype=bool))
    
    # For each pixel, if grass_data is nonzero, the new value is max_soil +1.
    combined = np.where(in_imp_soil != 0, max_soil +1, soil_data)
    
    combined[mask_soil] = soil_nodata
    
    soil_meta['crs'] = None
    
    # Update metadata to write as an ASCII grid (driver 'AAIGrid')
    soil_meta.update(driver="AAIGrid", dtype=combined.dtype, nodata=soil_nodata)
    
    # Write the combined raster to the output ASCII file.
    with rasterio.open(out_soil, "w", **soil_meta) as dst:
        dst.write(combined, 1)


# This function change the settings in vsd file for initial conditions
def initial_conditions_setting(root_path):
    # vsd is setting file
    vsd_file_path = os.path.join(root_path,
                                 'input_Irwell_to_Bury_Bridge_200m_vsd.txt')
    
    # vsi is file containing initial conditions
    vsi_file_path = os.path.join(root_path,
                                 'input_Irwell_to_Bury_Bridge_200m_vsi.txt')
    
    if not os.path.isfile(vsi_file_path):
        print(f"Error: {vsi_file_path} not found")
        sys.exit(1)
    
    # Read the file into a list of lines
    with open(vsd_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and modify the :VS03 block
    for i in range(len(lines)):
        if lines[i].startswith(':VS03'):
            # The next line contains the integer values
            values = lines[i + 1].split()
            # Change the last value (INITYP)
            values[-1] = '2'
            # Replace the line
            lines[i + 1] = '      '+'      '.join(values) + '\n'
            break
    
    # Write the modified lines to a new file
    with open(vsd_file_path, 'w') as f:
        f.writelines(lines)
#------------------------------------------------------------------------------


#-----------------------------------------------------------------------------#
# Creates initial for size chromosomes based on location chromosomes
#-----------------------------------------------------------------------------#
def initial_pop(in_pop_size, in_chrom_len):
   
    # Check if generating a unique population is feasible
    if in_pop_size > 2 ** in_chrom_len:
        raise ValueError(
    "Population size is too large for the given chromosome length to have "
    "all unique individuals."
    )
    
    population_set = set()
    population = []
    
    # Define the forced individuals
    individual_zero = tuple([0] * in_chrom_len)
    individual_one = tuple([1] * in_chrom_len)
    
    # Add the forced individuals
    population_set.add(individual_zero)
    population.append(np.array(individual_zero))
    
    population_set.add(individual_one)
    population.append(np.array(individual_one))
    
    # Generate the rest of the population until we have enough unique individuals
    while len(population) < in_pop_size:
        candidate = tuple(np.random.randint(2, size=in_chrom_len))
        if candidate not in population_set:
            population_set.add(candidate)
            population.append(np.array(candidate))
    
    return np.array(population)

#-----------------------------------------------------------------------------#
# Non-dominated sorting function
#-----------------------------------------------------------------------------#
def non_dominated_sorting(population_size,f_chroms_obj_record):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):
            
            if ((f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<=f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                f_chroms_obj_record[p][1]<=f_chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>=f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>=f_chroms_obj_record[q][1])):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                
    del front[len(front)-1]
    return front

#-----------------------------------------------------------------------------#
# calculate crowding distance function
#-----------------------------------------------------------------------------#
def calculate_crowding_distance(f_front,f_chroms_obj_record):
    distance = {}
    for i in range(len(f_front)):
        distance[i] = dict.fromkeys(f_front[i], 0)
        del i
    
    for o in range(len(f_front)):
            dt = dict.fromkeys(f_front[o], 0)
            dt_dis = dict.fromkeys(f_front[o], 0)
            de = dict.fromkeys(f_front[o], 0)
            de_dis = dict.fromkeys(f_front[o], 0)
            for k in f_front[o]:
                dt[k] = f_chroms_obj_record[k][0]
                de[k] = f_chroms_obj_record[k][1]
            del k
            dt_sort = {k: v for k, v in sorted(dt.items(), key=lambda 
                                               item: item[1])}
            de_sort = {k: v for k, v in sorted(de.items(), key=lambda 
                                               item: item[1])}
    
            ## now de_sort and dt_sort keys are not same, we need to find a
            ## way so we could calculate distance for element having same 
            ## key in dt_sort and de_sort    
            ## list(dictionary.values()) returns list of dictionary values
            key_lst = list(dt_sort.keys())    
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(dt_sort)-1:
                    dt_dis[key] = ((abs(dt_sort[key_lst[i+1]]-
                                        dt_sort[key_lst[i-1]]))/
                                   (dt_sort[key_lst[len(key_lst)-1]]-
                                    dt_sort[key_lst[0]]))
                else:
                    dt_dis[key] = 666666666
            del i,key, key_lst
            key_lst = list(de_sort.keys())  
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(de_sort)-1:
                    de_dis[key] = ((abs(de_sort[key_lst[i+1]]-
                                        de_sort[key_lst[i-1]]))/
                                   (de_sort[key_lst[len(key_lst)-1]]-
                                    de_sort[key_lst[0]]))
                else:
                    de_dis[key] = 333333333    
            
            t_dis = {}
            
            for i in key_lst:
                t_dis[i] = dt_dis[i]+de_dis[i]
            
            distance[o] = t_dis
    
    return distance

#-----------------------------------------------------------------------------#
# Sorting population based on their rank and crowding distance
#-----------------------------------------------------------------------------#
def fitness_sort(f_distance, f_pop_size):
    f_distance_sort = {}
    for i in range(len(f_distance)):
        f_distance_sort[i] = {k: v for k, v in sorted(f_distance[i].items(), 
                                                     key=lambda 
                                                     item: item[1], 
                                                     reverse = True)}
    parents_offspring = [None]*f_pop_size
    a = 0
    for i in range(len(f_distance_sort)):
        for j in f_distance_sort[i].keys():
            parents_offspring[a] = j
            a = a+1
    return parents_offspring

#-----------------------------------------------------------------------------#
# Parents selection using Binary Tournament
#-----------------------------------------------------------------------------#
def fitter_parent(f_sorted_fitness,f_pop_size):
    pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    while pairs_rand[0,0] == pairs_rand[0,1]:
        pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    if (np.where(f_sorted_fitness == pairs_rand[0,0]) < 
          np.where(f_sorted_fitness == pairs_rand[0,1])):
        return pairs_rand[0,0]
    else:
        return pairs_rand[0,1]
        

#-----------------------------------------------------------------------------#
# Random one point corssover
#-----------------------------------------------------------------------------#
def crossover_random_single_point_swap(f_pop, p1, p2, f_min_idx, f_max_idx):
    ## creating childrens
    ## random crossover index position for first child
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_1 = np.concatenate((f_pop[p1][0:c_index], 
                        f_pop[p2][c_index:len(f_pop[p1])]), axis=0)
    
    ## random crossover index position for second child
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
                        f_pop[p1][c_index:len(f_pop[p1])]), axis=0)
    
    # while np.all(f_child_1 == f_child_2) == True:
    #     c_index = np.random.randint(1,9)
    #     f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
    #                         f_pop[p1][c_index:len(f_pop[p1])]), axis=0)

    return f_child_1.transpose(), f_child_2.transpose() 

#-----------------------------------------------------------------------------#
# Random Bit Flip Mutation
#-----------------------------------------------------------------------------#
def mutation_random_bitflip(f_child_1, f_child_2, f_chrom_len, prob,
                            f_m_idx_range):
    ## Mutation of 1st child
    if prob > np.random.rand():
        m_index = np.random.randint(f_m_idx_range)
        if f_child_1[m_index] == 0:
            f_child_1[m_index] = 1
        else:
            f_child_1[m_index] = 0
    
    ## Mutation of 2nd child
    if prob > np.random.rand():    
        m_index = np.random.randint(f_m_idx_range)
        if f_child_2[m_index] == 0:
            f_child_2[m_index] = 1
        else:
            f_child_2[m_index] = 0
    
    ## Check if both children are the same
    while np.all(f_child_1 == f_child_2) == True:
        m_index = np.random.randint(f_m_idx_range)
        if f_child_2[m_index] == 0:
            f_child_2[m_index] = 1
        else:
            f_child_2[m_index] = 0
        
    return f_child_1, f_child_2
#-----------------------------------------------------------------------------#
# Remove duplicate from a list
#-----------------------------------------------------------------------------#
def remove_duplicate_list(record_list):
    # print('\n' 'Checking duplicates in the list:')
    m_pool = copy.deepcopy(record_list)
    idx = {}
    for i in range(0,len(m_pool)):
        for j in range(i+1,len(m_pool)):
            if np.all((m_pool[i] == m_pool[j]) == True):
                # print('Record no. {0} was equal to record no. {1}'
                #       .format(i,j))
                idx[j] = j 
    del i, j
    
    if idx!={}:
        m_pool = np.delete(m_pool, list(idx.values()),0)

    return m_pool, list(idx.values())


#-----------------------------------------------------------------------------#
# Removing duplicates from offspring
#-----------------------------------------------------------------------------#
def remove_duplicate_same_population(same_population):
    # print('\nChecking duplicate chroms in the same population:')
    pop_uniq = copy.deepcopy(same_population)
    a = {}
    for i in range(0, len(pop_uniq)):
        for j in range(i+1,len(pop_uniq)):
            if np.all((pop_uniq[i] == pop_uniq[j]) == True):
                # print('Chrom no. {0}  was equal to chrom no {1}'.format(i,j))
                a[j] = j
    if a!={}:
        pop_uniq = np.delete(pop_uniq, list(a.values()),0)
        # print('....\n {0} duplicate chroms deleted'.format(len(a)))
    return pop_uniq, list(a.values())

#-----------------------------------------------------------------------------#
# Removing offspring which are duplicates of parents
#-----------------------------------------------------------------------------#
def remove_duplicate_different_population(population1, population2):
    # print('\n' 'Checking duplicate chroms in different populations:')
    pop_1 = copy.deepcopy(population1)
    a = {}
    for i in range(0,len(population2)):
        for j in range(0,len(population1)):
            if np.all((population2[i] == population1[j]) == True):
                # print('Population 2 chrom no. ' + str(i) + 
                #       ' was equal to population 1 chrom no. ' + str(j))
                a[j] = j
    if a!={}:
        pop_1 = np.delete(pop_1, list(a.values()),0)
        # print('....\n {0} duplicate chroms from poulation 1 deleted'
        #       .format(len(a)))
    return pop_1


#-----------------------------------------------------------------------------#
#  Cost and Exposure of fittest population
#-----------------------------------------------------------------------------#
def final_exposure_cost(f_comb_total_c, f_comb_cat_high, 
                        f_select_fittest, population_size):
    total_c_f = [None]*population_size
    cat_high_f = [None]*population_size
    k = 0
    for i in f_select_fittest:
        total_c_f[k] = f_comb_total_c[i]
        cat_high_f[k] = f_comb_cat_high[i]
        k = k+1
    return total_c_f, cat_high_f

#-----------------------------------------------------------------------------#
#  Finds zone contribution to all solutions (population)
#-----------------------------------------------------------------------------#
def catch_contribution(f_shape_catch, f_join_column, f_chrom_len,
                      f_population, f_pop_size):    
    # calculating total contribution of each zone (chromosome)
    chrom_contribution = np.zeros((f_chrom_len, 3), dtype=int)
    for i in range(f_chrom_len):
        chrom_contribution[i,0] = i   # zone id
        chrom_contribution[i,1] = sum(f_population[:,i]) # zone contribution
        # zone percentage contribution
        chrom_contribution[i,2] = 100*chrom_contribution[i,1]/f_pop_size
    del i
    
    # creating dataframe from array
    df = pd.DataFrame(chrom_contribution)
    # assigning names to columns
    df.columns = ['catch_id', 'count', 'per_count']
    
    # # zone id in shapefile
    # shape_zone_id = np.array(f_shape_zones['cluster_id'])
    
    # joining fields from dataframe to shapefile (gpd)
    joined_shape_catch = pd.merge(
        left=f_shape_catch,
        right=df,
        left_on=f_join_column,
        right_on='catch_id',
        how='left'
        )
    
    contribution_column_shape_catch = 'per_count'
        
    return joined_shape_catch, contribution_column_shape_catch

#-----------------------------------------------------------------------------#
#  Creating scatter plot and a map on the same figure
#-----------------------------------------------------------------------------#
def scatter_plot_map_plot(f_fig_background, f_plot_title, f_cost, f_exposure,
         f_plot_legend_series, f_plot_x_limit, f_plot_y_limit,
         f_plot_x_axis_label, f_plot_y_axis_label, f_map_title, f_map_colour,
         f_map_legend_ticks, f_map_legend_label, f_shape_zones_joined,
         f_shape_zones_contribution_column, f_save_file):
    
    plt.style.use(f_fig_background)
    
    fig, ax = plt.subplots(1, 2, figsize =(18, 8), dpi = 300,
                           gridspec_kw={'width_ratios': [1, 1]})
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    '''----------------------------------------------------------------------
    This part creates scatter plot
    ----------------------------------------------------------------------'''    
    if f_fig_background == 'dark_background':
        edgecolors_evolving = '#2E75B6'
        
    else:
        edgecolors_evolving = '#2E75B6'
    # Create the scatter plot 
    ax[0].scatter(f_cost, f_exposure, s= 80, facecolors='#9BC2E6', 
               edgecolors=edgecolors_evolving, linewidth=1.5, 
               alpha=1, marker='o')
    
    # Add series legend
    ax[0].legend([f_plot_legend_series], 
               loc ="upper left", 
               prop={'weight': 'normal', "size": 14, 
                                           'stretch': 'normal'})
    ## Formate ticks
    #Set the current Axes to ax and the current Figure to the parent of ax
    plt.sca(ax[0])
    plt.xlim(f_plot_x_limit[0], f_plot_x_limit[1])
    plt.ylim(f_plot_y_limit[0], f_plot_y_limit[1])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tick_params(direction='out', length=6, width=1)
    
    # # Formate ticks labels
    # ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Formate grid
    plt.grid(color = '#A6A6A6', linestyle = '-', linewidth = 0.25)
     
    # Add labels and title
    plt.xlabel(f_plot_x_axis_label, fontsize = 14)
    plt.ylabel(f_plot_y_axis_label, fontsize = 14)
       
    plt.title(f_plot_title, fontsize = 18)
    
    # ax[0].spines['bottom'].set_color('white')
    # ax[0].spines['top'].set_color('white')
    # ax[0].spines['left'].set_color('white')
    # ax[0].spines['right'].set_color('white')
    
    '''----------------------------------------------------------------------
    This part creates map plot
    ----------------------------------------------------------------------'''
 
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cax.tick_params(labelsize='12')
    
    f_shape_zones_joined = copy.deepcopy(f_shape_zones_joined)
    f_shape_zones_joined[f_shape_zones_contribution_column] = (
    f_shape_zones_joined[f_shape_zones_contribution_column].replace(
                                                            {0:np.nan}))
    
    f_shape_zones_joined.plot(column= f_shape_zones_contribution_column,
                cmap=plt.get_cmap(f_map_colour,10), 
                ax=ax[1],
                vmin=0.0, vmax=100.0,
                legend=True ,cax = cax, edgecolor = 'lightgrey',
                linewidth=1, legend_kwds={'label': f_map_legend_label,
                                          'ticks': f_map_legend_ticks,
                                          'shrink': 0.5,
                                          'format': '%.0f%%'},
                missing_kwds={'color': 'white', 'edgecolor': 'lightgrey',
                              })
    
                # missing_kwds={'color': 'white', 'edgecolor': 'lightgrey',
                #               "hatch": "///"})
        
                
    if f_shape_zones_joined[f_shape_zones_contribution_column].isna().sum()!=0:
        ax[1].text(0.1, 0.03, 'No contribution ',
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=ax[1].transAxes, fontsize = 12)
        ax[1].add_patch(Rectangle((373300, 409660), 1000, 800,
                 edgecolor = 'lightgrey',
                 facecolor = 'white',
                 fill=True,
                 lw=1))
        
    # Set the current Axes to ax and the current Figure to the parent of ax
    plt.sca(ax[1])
    # plt.xlim(364000, 390000)
    # plt.ylim(563000, 567000)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14, rotation = 90, ha = 'right')
    # plt.xticks([], [])
    # plt.yticks([], [])
    plt.tick_params(direction='out', length=6, width=1)
    plt.xlabel('Eastings (m)', fontsize = 14)
    plt.ylabel('Northings (m)', fontsize = 14)
   
    ax[1].yaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1].set_title(f_map_title, size=18)
    # ax[1].get_xaxis().set_visible(False)
    # ax[1].get_yaxis().set_visible(False)
        
    # plt.show()            
    # Save figure to desktop
    plt.savefig(f_save_file, dpi='figure', transparent = False, 
                bbox_inches = 'tight', pad_inches = 0.25)
    plt.close()

#-----------------------------------------------------------------------------#
# Takes new population and separates new & old individuals
#-----------------------------------------------------------------------------#
def separate_new_old(f_new_population, f_old_population):
    # print('\n' 'Checking new and old chroms in new population')
    f_new_chroms = copy.deepcopy(f_new_population)
    f_old_chroms = copy.deepcopy(f_new_population)
    ## ^ for new and old we used f_new_population. For new_chroms, we
    ## will delete old chroms from new population. For old_chroms, we
    ## will select old chroms from new population
    a = {}
    for i in range(0,len(f_old_population)):
        for j in range(0,len(f_new_population)):
            if np.all((f_old_population[i] == f_new_population[j]) == True):
                # print('Chromosome no. {0} in new population was '.format(i) +  
                #       'equal to chromosome no. {0} in old population'
                #       .format(j))
                a[j] = j
    if a!={}:
        f_new_chroms = np.delete(f_new_chroms, list(a.values()),0)
        f_old_chroms = f_old_chroms[list(a.values())]
        f_old_chroms_index = list(a.values())
    return f_new_chroms, f_old_chroms, f_old_chroms_index

#-----------------------------------------------------------------------------#
# Deletes selected variables
#-----------------------------------------------------------------------------#
def remove_variables(f_dir, f_variables_list):
    for var in f_variables_list:
        for obj in f_dir:
            if var == obj:
                del globals()[var]
                
#-----------------------------------------------------------------------------#
# Deletes those chromosomes which have same objective functions
#-----------------------------------------------------------------------------#
def remove_same_objectives_population(f_comb_population, f_dup_idx_obj):
    # print('\n' 'Checking duplicate chroms in different populations:')
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)

    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
        # print('....\n {0} duplicate chroms from poulation 1 deleted'
        #       .format(len(a)))
    return comb_pop
###############################################################################
""" Functions section ends here """
###############################################################################
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
Organisations: Manchester University, Newcastle Univeristy

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


# =============================================================================
# WHAT IS IN THIS FILE
# =============================================================================
# Functions called by rains_main.py (imported there as "rf"):
#   SHETRAN output and flow statistics
#     read_shetran, generate_growth_curve, get_growth_curves
#   Preparing SHETRAN inputs for one solution
#     nfm_selection (keep NFM only in selected sub-catchments of a grid),
#     soil_update (new soil type where improved grassland/soil NFM is applied),
#     initial_conditions_setting (read initial conditions from the vsi file)
#   NSGA-II genetic algorithm
#     initial_pop, non_dominated_sorting, calculate_crowding_distance,
#     fitness_sort, fitter_parent, crossover_random_single_point_swap,
#     mutation_random_bitflip
#   Duplicate handling
#     remove_duplicate_list, remove_duplicate_same_population,
#     remove_duplicate_different_population,
#     remove_same_objectives_population, separate_new_old
#   Results and figures
#     catch_contribution, scatter_plot_map_plot
#   Not used by rains_main.py: final_exposure_cost, remove_variables
#
# In the NSGA-II functions a "chromosome" is a 1-D 0/1 array with one gene per
# sub-catchment, a "population" is a 2-D array of chromosomes (one per row),
# and solutions are referred to by their row index. Both objectives are
# minimised (rains_main.py passes [total NFM area, -damages avoided]).
# Arguments starting with "f_" or "in_" are simply function inputs.

# Hides all warnings from here on (this changes Python's global warning
# filter, so it also applies to rains_main.py once this module is imported)
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
    
    # Column names: the first line of the file is skipped; on the second line
    # the river link numbers start at character 39 (the text before that
    # describes the first column, the outlet).
    with open(filename, "r") as f:
        f.readline()
        col_names = [int(i) for i in f.readline().rstrip()[39:].split()]
        col_names.insert(0, 'outlet')
    
    # Create a date range with daily frequency
    # (one date per data row, so start_date and end_date must match the
    # simulation exactly; otherwise set_index below fails)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Create an empty DataFrame with the date index
    
    # The flows start on line 3, separated by spaces; one row per day and one
    # column per discharge point
    df = pd.read_csv(filename, sep=r"\s+", skiprows=2, names=col_names)
    df.set_index(date_range, inplace=True)

    return(df)

def generate_growth_curve(amax, event_RP):
    """Fit a GEV distribution to annual maxima and return design flows.

    A Generalised Extreme Value (GEV) distribution is fitted to the annual
    maximum flows and evaluated at the return periods
    [event_RP, 5, 20, 50, 100, 200] years.

    Parameters
    ----------
    amax : array-like
        Annual maximum flows for one river link.
    event_RP : float
        Return period (years) of the modelled event.

    Returns
    -------
    numpy.ndarray
        Six flow values (same units as ``amax``) for the return periods
        listed above, in that order.
    """
    # Return periods (years) at which to evaluate the curve. event_RP is the
    # return period calculated in rains_main.py (about 3.2 for the 1983 event).
    rps = np.array([event_RP, 5, 20, 50, 100, 200]) # This is assuming that the modelled event has an RP of 3.2
    # Note: despite the "negate" comments on the right, no values are negated;
    # the GEV is fitted to the annual maxima as they are.
    # Fit a Generalized Extreme Value (GEV) distribution to the data
    shape, loc, scale = genextreme.fit(amax)  # Negate values for maxima fitting
    # 1 - 1/RP is the probability that a year's maximum is BELOW the flow
    # (non-exceedance), which is what genextreme.ppf expects
    probs = 1 - (1 / rps)  # Exceedance probability
    # Flow with that probability = design flow for each return period
    flow_values = genextreme.ppf(probs, shape, loc, scale)  # Invert negation

    return(flow_values)

def get_growth_curves(filename, start_date, end_date, event_RP):
    """Build flow growth curves for every river link in a SHETRAN output file.

    The file is read with :func:`read_shetran`, annual (calendar-year) maxima
    are taken for each column and :func:`generate_growth_curve` is applied to
    each column.

    Parameters
    ----------
    filename : str
        Path to a SHETRAN multi-point discharge output file (daily values).
    start_date, end_date : str
        First and last dates of the simulation, format "%Y-%m-%d".
    event_RP : float
        Return period (years) of the modelled event.

    Returns
    -------
    pandas.DataFrame
        Flows indexed by return period ("RP": event_RP, 5, 20, 50, 100, 200),
        with one column per river link ("outlet" plus the link numbers).
    """
    df = read_shetran(filename, start_date, end_date)
    # Largest daily flow of every calendar year ("YE" = year end), per column
    annual_maxima = df.resample("YE").max().dropna()
    # Fit a GEV to each column (river link) separately; each column becomes
    # six design flows
    growth_curves = annual_maxima.apply(lambda col: generate_growth_curve(col, event_RP), axis=0)
    # Label the six rows with their return periods (same order as in
    # generate_growth_curve)
    growth_curves["RP"] = [event_RP, 5, 20, 50, 100, 200] # This is assuming that the modelled event has an RP of 3.2
    growth_curves.set_index("RP", inplace=True)
    return(growth_curves)

def nfm_selection (in_cat_ids, in_cat_shp, in_nfm, out_nfm=None, save_nfm=True):
    """
    Remove NFM from the given sub-catchments of a maximum-NFM raster.

    Cells that fall inside the polygons whose ``opt_id`` is in
    ``in_cat_ids`` are set to 0; all other cells keep their values.
    ``rains_main.py`` passes the sub-catchments *without* NFM, so NFM is
    kept only in the selected sub-catchments.

    Parameters
    ----------
    in_cat_ids : array-like of int
        Sub-catchment IDs (``opt_id`` values) whose cells are set to 0.
    in_cat_shp : str
        Path to the sub-catchment polygon shapefile (field ``opt_id``).
    in_nfm : str
        Path to the input ASCII grid (maximum NFM extent).
    out_nfm : str, optional
        Path of the ASCII grid to write; required when ``save_nfm`` is True.
    save_nfm : bool, optional
        If True (default) write the grid to ``out_nfm`` and return None;
        if False return the grid as an array instead.

    Returns
    -------
    numpy.ndarray or None
        The modified 2-D grid when ``save_nfm`` is False, otherwise None.

    Raises
    ------
    ValueError
        If ``out_nfm`` is not provided while ``save_nfm`` is True.
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
    # (meta is not used by rasterize, and the CRS is set back to None before
    # writing, so this does not affect the result)
    meta['crs'] = gdf.crs
    
    # Filter the polygons whose opt_id is in the given list
    selected_polygons = gdf[gdf['opt_id'].isin(in_cat_ids)]
    
    # Rasterize the selected polygons.
    # Each cell that falls within any of these polygons gets
    # a value of 1; others remain 0.
    # (transform links row/column numbers to map coordinates, so the polygons
    # line up with the raster cells)
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
    """Write a soil map in which improved-soil NFM cells get a new soil type.

    Cells where ``in_imp_soil`` is non-zero are given a new soil category
    equal to the largest existing soil category plus one. All other cells,
    including nodata cells, keep their original values.

    Parameters
    ----------
    in_org_soil : str
        Path to the original SHETRAN soil-category ASCII grid.
    in_imp_soil : numpy.ndarray
        2-D array with the same shape as the soil grid; non-zero values mark
        cells with improved grassland/soil NFM.
    out_soil : str
        Path of the ASCII grid to write (overwritten if it exists).

    Returns
    -------
    None
        The new soil map is written to ``out_soil``.
    """
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
    # (True where the soil grid has no data)
    mask_soil = ((soil_data == soil_nodata) if soil_nodata is not None
                 else np.zeros_like(soil_data, dtype=bool))
    
    # For each pixel, if grass_data is nonzero, the new value is max_soil +1.
    # (grass_data = in_imp_soil, the improved-soil grid of the selected
    # sub-catchments; all other cells keep their soil category)
    combined = np.where(in_imp_soil != 0, max_soil +1, soil_data)
    
    # nodata cells stay nodata
    combined[mask_soil] = soil_nodata
    
    # SHETRAN does not need a CRS
    soil_meta['crs'] = None
    
    # Update metadata to write as an ASCII grid (driver 'AAIGrid')
    soil_meta.update(driver="AAIGrid", dtype=combined.dtype, nodata=soil_nodata)
    
    # Write the combined raster to the output ASCII file.
    with rasterio.open(out_soil, "w", **soil_meta) as dst:
        dst.write(combined, 1)


# This function change the settings in vsd file for initial conditions
def initial_conditions_setting(root_path):
    """Tell SHETRAN to read subsurface initial conditions from a file.

    Edits ``input_Irwell_to_Bury_Bridge_200m_vsd.txt`` in ``root_path`` in
    place: on the line after ``:VS03`` the last value (INITYP) is set to 2.
    The initial-conditions file ``input_Irwell_to_Bury_Bridge_200m_vsi.txt``
    must exist in the same folder.

    Parameters
    ----------
    root_path : str
        Folder containing the SHETRAN vsd and vsi input files.

    Returns
    -------
    None
        The vsd file is overwritten. If the vsi file is missing, an error is
        printed and the program exits with ``sys.exit(1)``.
    """
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
    # (":VS03" is a heading in the vsd file; its values are on the next line.
    # Only the first :VS03 heading is changed.)
    for i in range(len(lines)):
        if lines[i].startswith(':VS03'):
            # The next line contains the integer values
            values = lines[i + 1].split()
            # Change the last value (INITYP)
            # (2 = read the initial conditions from the vsi file)
            values[-1] = '2'
            # Replace the line
            # (each value preceded by six spaces)
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
    """Create a random initial population of unique binary chromosomes.

    Each gene represents one sub-catchment (1 = NFM implemented, 0 = no NFM).
    The first individual has no NFM anywhere (all zeros) and the second has
    NFM in every sub-catchment (all ones); the rest are random and unique.

    Parameters
    ----------
    in_pop_size : int
        Number of chromosomes (population size).
    in_chrom_len : int
        Number of genes per chromosome (number of sub-catchments).

    Returns
    -------
    numpy.ndarray
        Integer array of shape (in_pop_size, in_chrom_len) with values 0/1.

    Raises
    ------
    ValueError
        If ``in_pop_size`` is larger than ``2 ** in_chrom_len``, so that a
        population of unique chromosomes cannot exist.
    """
   
    # Check if generating a unique population is feasible
    if in_pop_size > 2 ** in_chrom_len:
        raise ValueError(
    "Population size is too large for the given chromosome length to have "
    "all unique individuals."
    )
    
    # population_set holds the chromosomes as tuples so that duplicates can be
    # detected quickly; population holds the same chromosomes as arrays, in
    # the order they were created
    population_set = set()
    population = []
    
    # Define the forced individuals
    # (no NFM anywhere, and NFM in every sub-catchment: the two extremes)
    individual_zero = tuple([0] * in_chrom_len)
    individual_one = tuple([1] * in_chrom_len)
    
    # Add the forced individuals
    population_set.add(individual_zero)
    population.append(np.array(individual_zero))
    
    population_set.add(individual_one)
    population.append(np.array(individual_one))
    
    # Generate the rest of the population until we have enough unique individuals
    while len(population) < in_pop_size:
        # random 0/1 genes; kept only if not already in the population
        candidate = tuple(np.random.randint(2, size=in_chrom_len))
        if candidate not in population_set:
            population_set.add(candidate)
            population.append(np.array(candidate))
    
    return np.array(population)

#-----------------------------------------------------------------------------#
# Non-dominated sorting function
#-----------------------------------------------------------------------------#
def non_dominated_sorting(population_size,f_chroms_obj_record):
    """Rank solutions into Pareto fronts (fast non-dominated sorting, NSGA-II).

    Both objectives are minimised. A solution dominates another if it is no
    worse in both objectives and strictly better in at least one.

    Parameters
    ----------
    population_size : int
        Number of solutions (rows of ``f_chroms_obj_record``) to rank.
    f_chroms_obj_record : array-like
        Objective values, shape (population_size, 2).

    Returns
    -------
    dict
        ``{front_number: [solution indices]}``; front 0 is the
        non-dominated (Pareto-optimal) front.
    """
    # s[p]: solutions that p dominates
    # n[p]: number of solutions that dominate p
    # front[i]: solutions in front i; rank[p]: front number of p
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    # Compare every solution p with every solution q
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):
            
            # p dominates q: p is lower (better) in both objectives, or equal
            # in one and lower in the other
            if ((f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<=f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                f_chroms_obj_record[p][1]<=f_chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            # q dominates p (the same test the other way round)
            elif ((f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>=f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>=f_chroms_obj_record[q][1])):
                n[p]=n[p]+1
        # Nobody dominates p, so p is in the first (Pareto-optimal) front
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    # Build the next fronts: remove the current front from the counts; the
    # solutions whose count drops to 0 are dominated only by solutions in
    # earlier fronts and form the next front
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
                
    # the loop always ends with an empty front; remove it (rank is not
    # returned)
    del front[len(front)-1]
    return front

#-----------------------------------------------------------------------------#
# calculate crowding distance function
#-----------------------------------------------------------------------------#
def calculate_crowding_distance(f_front,f_chroms_obj_record):
    """Compute the crowding distance of every solution within its front.

    For each objective the solutions of a front are sorted; the two boundary
    solutions receive a large constant (666666666 for the first objective,
    333333333 for the second) and every other solution receives the distance
    between its two neighbours divided by the range of that objective in the
    front. The two contributions are added together.

    Parameters
    ----------
    f_front : dict
        Fronts as returned by :func:`non_dominated_sorting`.
    f_chroms_obj_record : array-like
        Objective values, shape (n_solutions, 2).

    Returns
    -------
    dict
        ``{front_number: {solution_index: crowding_distance}}``.
    """
    # distance[front number] = {solution index: crowding distance}
    distance = {}
    for i in range(len(f_front)):
        distance[i] = dict.fromkeys(f_front[i], 0)
        del i
    
    # Work front by front (o = front number)
    for o in range(len(f_front)):
            # dt / de: first and second objective of each solution in the
            # front; dt_dis / de_dis: the distance from each objective
            dt = dict.fromkeys(f_front[o], 0)
            dt_dis = dict.fromkeys(f_front[o], 0)
            de = dict.fromkeys(f_front[o], 0)
            de_dis = dict.fromkeys(f_front[o], 0)
            for k in f_front[o]:
                dt[k] = f_chroms_obj_record[k][0]
                de[k] = f_chroms_obj_record[k][1]
            del k
            # the solutions of the front sorted by each objective (ascending)
            dt_sort = {k: v for k, v in sorted(dt.items(), key=lambda 
                                               item: item[1])}
            de_sort = {k: v for k, v in sorted(de.items(), key=lambda 
                                               item: item[1])}
    
            ## now de_sort and dt_sort keys are not same, we need to find a
            ## way so we could calculate distance for element having same 
            ## key in dt_sort and de_sort    
            ## list(dictionary.values()) returns list of dictionary values
            # First objective: a solution in the middle gets the gap between
            # its two neighbours divided by the objective's range in this
            # front (normalised). The two end solutions get a very large
            # value, so they are ranked first within their front and the
            # extremes of the front are favoured.
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
            # Second objective: the same, with a different large value for
            # the ends
            key_lst = list(de_sort.keys())  
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(de_sort)-1:
                    de_dis[key] = ((abs(de_sort[key_lst[i+1]]-
                                        de_sort[key_lst[i-1]]))/
                                   (de_sort[key_lst[len(key_lst)-1]]-
                                    de_sort[key_lst[0]]))
                else:
                    de_dis[key] = 333333333    
            
            # Crowding distance = sum of both contributions. A larger value
            # means the solution lies in a less crowded part of the front.
            t_dis = {}
            
            for i in key_lst:
                t_dis[i] = dt_dis[i]+de_dis[i]
            
            distance[o] = t_dis
    
    return distance

#-----------------------------------------------------------------------------#
# Sorting population based on their rank and crowding distance
#-----------------------------------------------------------------------------#
def fitness_sort(f_distance, f_pop_size):
    """Order solutions from fittest to least fit.

    Solutions are ordered by front (front 0 first) and, within each front,
    by crowding distance in descending order.

    Parameters
    ----------
    f_distance : dict
        Crowding distances as returned by :func:`calculate_crowding_distance`.
    f_pop_size : int
        Total number of solutions contained in ``f_distance``.

    Returns
    -------
    list of int
        Solution indices, fittest first.
    """
    # Within each front, sort the solutions by crowding distance, largest
    # first
    f_distance_sort = {}
    for i in range(len(f_distance)):
        f_distance_sort[i] = {k: v for k, v in sorted(f_distance[i].items(), 
                                                     key=lambda 
                                                     item: item[1], 
                                                     reverse = True)}
    # List the solutions front by front (front 0 first) in that order
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
    """Select a parent by binary tournament.

    Two different random indices are drawn from ``range(f_pop_size)``; the one
    that appears earlier in ``f_sorted_fitness`` (the fitter one) is returned.

    Parameters
    ----------
    f_sorted_fitness : numpy.ndarray
        Solution indices sorted from fittest to least fit
        (see :func:`fitness_sort`).
    f_pop_size : int
        Population size.

    Returns
    -------
    int
        Index of the selected parent.
    """
    # Two different random solutions (row indices 0 .. f_pop_size-1)
    pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    while pairs_rand[0,0] == pairs_rand[0,1]:
        pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    # Compare their positions in the fitness ranking: the one nearer the
    # front of f_sorted_fitness is fitter and wins the tournament
    if (np.where(f_sorted_fitness == pairs_rand[0,0]) < 
          np.where(f_sorted_fitness == pairs_rand[0,1])):
        return pairs_rand[0,0]
    else:
        return pairs_rand[0,1]
        

#-----------------------------------------------------------------------------#
# Random one point corssover
#-----------------------------------------------------------------------------#
def crossover_random_single_point_swap(f_pop, p1, p2, f_min_idx, f_max_idx):
    """Create two children by single-point crossover.

    Each child uses its own random cut point drawn from
    ``[f_min_idx, f_max_idx)``: child 1 takes the genes of parent 1 before
    the cut and of parent 2 after it; child 2 does the opposite.

    Parameters
    ----------
    f_pop : numpy.ndarray
        Current population, shape (pop_size, chrom_len).
    p1, p2 : int
        Row indices of the two parents in ``f_pop``.
    f_min_idx, f_max_idx : int
        Lower (inclusive) and upper (exclusive) bounds of the cut point.

    Returns
    -------
    tuple of numpy.ndarray
        The two children (1-D arrays of length chrom_len).
    """
    ## creating childrens
    ## random crossover index position for first child
    # child 1 = parent 1 genes [0, cut) + parent 2 genes [cut, end)
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_1 = np.concatenate((f_pop[p1][0:c_index], 
                        f_pop[p2][c_index:len(f_pop[p1])]), axis=0)
    
    ## random crossover index position for second child
    # child 2 = parent 2 genes [0, cut) + parent 1 genes [cut, end), with its
    # own random cut point
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
                        f_pop[p1][c_index:len(f_pop[p1])]), axis=0)
    
    # while np.all(f_child_1 == f_child_2) == True:
    #     c_index = np.random.randint(1,9)
    #     f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
    #                         f_pop[p1][c_index:len(f_pop[p1])]), axis=0)

    # (transposing a 1-D array leaves it unchanged)
    return f_child_1.transpose(), f_child_2.transpose() 

#-----------------------------------------------------------------------------#
# Random Bit Flip Mutation
#-----------------------------------------------------------------------------#
def mutation_random_bitflip(f_child_1, f_child_2, f_chrom_len, prob,
                            f_m_idx_range):
    """Apply random bit-flip mutation to two children.

    With probability ``prob`` one randomly chosen gene of each child is
    flipped (0 to 1 or 1 to 0). If the two children are then identical,
    random genes of child 2 are flipped until they differ. The input arrays
    are modified in place and also returned.

    Parameters
    ----------
    f_child_1, f_child_2 : numpy.ndarray
        The two children (1-D binary arrays).
    f_chrom_len : int
        Chromosome length (not used inside the function).
    prob : float
        Mutation probability for each child.
    f_m_idx_range : int
        Genes are chosen from ``range(f_m_idx_range)``.

    Returns
    -------
    tuple of numpy.ndarray
        The mutated children.
    """
    # The children are changed in place (the arrays passed in are modified)
    ## Mutation of 1st child
    # with probability prob, flip one randomly chosen gene
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
    # keep flipping random genes of child 2 until the two children differ
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
    """Remove repeated rows from an array of records.

    The first occurrence of each row is kept.

    Parameters
    ----------
    record_list : numpy.ndarray
        2-D array of records (e.g. objective-value pairs). As written, it
        needs at least two rows.

    Returns
    -------
    tuple
        ``(unique_records, removed_indices)`` where ``removed_indices`` lists
        the row numbers that were removed.
    """
    # print('\n' 'Checking duplicates in the list:')
    m_pool = copy.deepcopy(record_list)
    # Compare every row i with every later row j; a later row equal to an
    # earlier one is marked for removal (dict keys avoid marking a row twice)
    idx = {}
    for i in range(0,len(m_pool)):
        for j in range(i+1,len(m_pool)):
            if np.all((m_pool[i] == m_pool[j]) == True):
                # print('Record no. {0} was equal to record no. {1}'
                #       .format(i,j))
                idx[j] = j 
    # NOTE (known issue): with fewer than two rows, j (or i) is never
    # assigned and this "del" fails. rains_main.py always passes more rows.
    del i, j
    
    # remove the marked rows (the first copy of each record is kept)
    if idx!={}:
        m_pool = np.delete(m_pool, list(idx.values()),0)

    return m_pool, list(idx.values())


#-----------------------------------------------------------------------------#
# Removing duplicates from offspring
#-----------------------------------------------------------------------------#
def remove_duplicate_same_population(same_population):
    """Remove duplicate chromosomes from a population.

    The first occurrence of each chromosome is kept.

    Parameters
    ----------
    same_population : numpy.ndarray
        Population, shape (n, chrom_len).

    Returns
    -------
    tuple
        ``(unique_population, removed_indices)``.
    """
    # print('\nChecking duplicate chroms in the same population:')
    pop_uniq = copy.deepcopy(same_population)
    # Compare every chromosome with every later one; later copies are marked
    # for removal (a: {row index: row index})
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
    """Remove from one population any chromosome that exists in another.

    Parameters
    ----------
    population1 : numpy.ndarray
        Population to filter (e.g. new offspring).
    population2 : numpy.ndarray
        Reference population (e.g. all chromosomes simulated so far).

    Returns
    -------
    numpy.ndarray
        ``population1`` without the chromosomes that also appear in
        ``population2``.
    """
    # print('\n' 'Checking duplicate chroms in different populations:')
    pop_1 = copy.deepcopy(population1)
    # Mark every chromosome of population1 that also occurs anywhere in
    # population2 (a: {row index in population1: row index}), then delete
    # the marked rows. In rains_main.py this removes offspring that have
    # already been simulated.
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
    """Collect the cost and exposure values of the selected solutions.

    Not used by ``rains_main.py``; kept for compatibility.

    Parameters
    ----------
    f_comb_total_c : sequence
        Cost of every solution.
    f_comb_cat_high : sequence
        Exposure (or damage) of every solution.
    f_select_fittest : sequence of int
        Indices of the selected solutions.
    population_size : int
        Number of selected solutions.

    Returns
    -------
    tuple of list
        ``(costs, exposures)`` of the selected solutions, in selection order.
    """
    # (not used by rains_main.py)
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
    """Count how often each sub-catchment is selected and join it to a map.

    Parameters
    ----------
    f_shape_catch : geopandas.GeoDataFrame
        Sub-catchment polygons.
    f_join_column : str
        Column in ``f_shape_catch`` holding the sub-catchment (gene) number.
    f_chrom_len : int
        Number of sub-catchments (genes).
    f_population : numpy.ndarray
        Population, shape (f_pop_size, f_chrom_len).
    f_pop_size : int
        Number of chromosomes in ``f_population``.

    Returns
    -------
    tuple
        ``(joined, column_name)``: the polygons merged with the columns
        ``catch_id``, ``count`` and ``per_count`` (percentage of chromosomes
        that select the sub-catchment, stored as an integer), and the name of
        the percentage column (``'per_count'``).
    """
    # calculating total contribution of each zone (chromosome)
    # ("zone" = sub-catchment = gene). One row per sub-catchment: its number,
    # how many chromosomes have a 1 for it, and that count as a percentage of
    # the population. The array is integer, so the percentage is cut to a
    # whole number.
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
    # (left join on the polygons: every polygon is kept; polygons whose ID
    # does not match a gene get empty values)
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
    """Save a two-panel figure: objective scatter plot and contribution map.

    The left panel plots the two objectives of every solution; the right
    panel maps the percentage contribution of each sub-catchment. The figure
    is saved at 300 dpi (PNG when ``f_save_file`` has no extension).

    Parameters
    ----------
    f_fig_background : str
        Matplotlib style name, e.g. ``'default'`` or ``'dark_background'``.
    f_plot_title : str
        Title of the scatter plot.
    f_cost, f_exposure : array-like
        x and y values of the scatter plot (total NFM area and damages
        avoided in ``rains_main.py``).
    f_plot_legend_series : str
        Legend label of the scatter series.
    f_plot_x_limit, f_plot_y_limit : sequence of two floats
        Axis limits of the scatter plot.
    f_plot_x_axis_label, f_plot_y_axis_label : str
        Axis labels of the scatter plot.
    f_map_title : str
        Title of the map.
    f_map_colour : str
        Matplotlib colormap name for the map.
    f_map_legend_ticks : array-like
        Tick positions of the map colour bar (percent).
    f_map_legend_label : str
        Label of the map colour bar.
    f_shape_zones_joined : geopandas.GeoDataFrame
        Polygons with a contribution column (see :func:`catch_contribution`).
    f_shape_zones_contribution_column : str
        Name of the contribution column.
    f_save_file : str
        Output file path.

    Returns
    -------
    None
    """
    
    plt.style.use(f_fig_background)
    
    # One figure, two panels side by side: ax[0] scatter plot, ax[1] map
    fig, ax = plt.subplots(1, 2, figsize =(18, 8), dpi = 300,
                           gridspec_kw={'width_ratios': [1, 1]})
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    '''----------------------------------------------------------------------
    This part creates scatter plot
    ----------------------------------------------------------------------'''    
    # (both branches currently use the same edge colour)
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
 
    # Narrow axis to the right of the map for the colour bar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cax.tick_params(labelsize='12')
    
    # Work on a copy; 0 % contribution is turned into "missing" so those
    # sub-catchments are drawn white (missing_kwds) instead of coloured
    f_shape_zones_joined = copy.deepcopy(f_shape_zones_joined)
    f_shape_zones_joined[f_shape_zones_contribution_column] = (
    f_shape_zones_joined[f_shape_zones_contribution_column].replace(
                                                            {0:np.nan}))
    
    # Colour each sub-catchment by its contribution (0-100 %, colour map
    # split into 10 classes)
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
        
                
    # If any sub-catchment has no contribution, add a small legend entry: the
    # text is placed relative to the panel, the white box at fixed map
    # coordinates (easting 373300 m, northing 409660 m).
    # USER INPUT: move the box for a map of another area.
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
    # (saved to f_save_file at the figure's 300 dpi; the figure is then
    # closed to free memory)
    plt.savefig(f_save_file, dpi='figure', transparent = False, 
                bbox_inches = 'tight', pad_inches = 0.25)
    plt.close()

#-----------------------------------------------------------------------------#
# Takes new population and separates new & old individuals
#-----------------------------------------------------------------------------#
def separate_new_old(f_new_population, f_old_population):
    """Split a new population into newly created and surviving chromosomes.

    Parameters
    ----------
    f_new_population : numpy.ndarray
        Population after selection.
    f_old_population : numpy.ndarray
        Population of the previous generation.

    Returns
    -------
    tuple
        ``(new_chroms, old_chroms, old_chroms_index)``: chromosomes not found
        in the old population, chromosomes also found in the old population,
        and the row indices (in ``f_new_population``) of the latter.

    Notes
    -----
    As written, at least one chromosome must be shared by the two
    populations; otherwise ``old_chroms_index`` is never assigned.
    """
    # print('\n' 'Checking new and old chroms in new population')
    f_new_chroms = copy.deepcopy(f_new_population)
    f_old_chroms = copy.deepcopy(f_new_population)
    ## ^ for new and old we used f_new_population. For new_chroms, we
    ## will delete old chroms from new population. For old_chroms, we
    ## will select old chroms from new population
    # a: rows of the new population that also occur in the old population
    a = {}
    for i in range(0,len(f_old_population)):
        for j in range(0,len(f_new_population)):
            if np.all((f_old_population[i] == f_new_population[j]) == True):
                # print('Chromosome no. {0} in new population was '.format(i) +  
                #       'equal to chromosome no. {0} in old population'
                #       .format(j))
                a[j] = j
    # new = rows not found in the old population; old = rows found there
    if a!={}:
        f_new_chroms = np.delete(f_new_chroms, list(a.values()),0)
        f_old_chroms = f_old_chroms[list(a.values())]
        f_old_chroms_index = list(a.values())
    # NOTE (known issue): if no chromosome is shared (a is empty),
    # f_old_chroms_index is never assigned and this line raises
    # UnboundLocalError.
    return f_new_chroms, f_old_chroms, f_old_chroms_index

#-----------------------------------------------------------------------------#
# Deletes selected variables
#-----------------------------------------------------------------------------#
def remove_variables(f_dir, f_variables_list):
    """Delete named global variables.

    Not used by ``rains_main.py``. Note that ``globals()`` here refers to the
    namespace of this module, not to the caller's namespace.

    Parameters
    ----------
    f_dir : iterable of str
        Names that currently exist (e.g. the output of ``dir()``).
    f_variables_list : iterable of str
        Names to delete if they appear in ``f_dir``.

    Returns
    -------
    None
    """
    # (not used by rains_main.py)
    for var in f_variables_list:
        for obj in f_dir:
            if var == obj:
                del globals()[var]
                
#-----------------------------------------------------------------------------#
# Deletes those chromosomes which have same objective functions
#-----------------------------------------------------------------------------#
def remove_same_objectives_population(f_comb_population, f_dup_idx_obj):
    """Remove chromosomes at the given row indices.

    Used to drop chromosomes whose objective values duplicate those of
    another chromosome (indices from :func:`remove_duplicate_list`).

    Parameters
    ----------
    f_comb_population : numpy.ndarray
        Combined parent and offspring population.
    f_dup_idx_obj : list of int
        Row indices to remove.

    Returns
    -------
    numpy.ndarray
        The population without those rows.
    """
    # print('\n' 'Checking duplicate chroms in different populations:')
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)

    # delete those rows, so the chromosomes stay aligned with the
    # de-duplicated objective records
    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
        # print('....\n {0} duplicate chroms from poulation 1 deleted'
        #       .format(len(a)))
    return comb_pop
###############################################################################
""" Functions section ends here """
###############################################################################

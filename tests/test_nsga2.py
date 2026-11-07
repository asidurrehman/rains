"""Tests for the NSGA-II building blocks in rains_all_functions.py."""
import numpy as np
import pytest

import rains_all_functions as rf


# --- Initial population -----------------------------------------------------

def test_initial_pop_shape_values_and_uniqueness():
    np.random.seed(1)
    pop = rf.initial_pop(20, 8)
    assert pop.shape == (20, 8)
    assert set(np.unique(pop)) <= {0, 1}
    assert len({tuple(row) for row in pop}) == 20


def test_initial_pop_starts_with_no_nfm_and_full_nfm():
    np.random.seed(2)
    pop = rf.initial_pop(5, 6)
    assert np.array_equal(pop[0], np.zeros(6))
    assert np.array_equal(pop[1], np.ones(6))


def test_initial_pop_can_fill_every_combination():
    np.random.seed(3)
    pop = rf.initial_pop(4, 2)
    assert {tuple(row) for row in pop} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_initial_pop_rejects_impossible_size():
    with pytest.raises(ValueError):
        rf.initial_pop(5, 2)


def test_initial_pop_is_reproducible_with_seed():
    np.random.seed(42)
    a = rf.initial_pop(10, 12)
    np.random.seed(42)
    b = rf.initial_pop(10, 12)
    assert np.array_equal(a, b)


# --- Non-dominated sorting and crowding distance ------------------------------

# Two objectives, both minimised.
OBJS = np.array([
    [1, 5],   # 0  front 0
    [2, 3],   # 1  front 0
    [3, 1],   # 2  front 0
    [2, 5],   # 3  front 1 (dominated by 0 and 1)
    [4, 4],   # 4  front 1 (dominated by 1 and 2)
    [5, 6],   # 5  front 2 (dominated by 3 and 4)
])


def test_non_dominated_sorting_fronts():
    front = rf.non_dominated_sorting(len(OBJS), OBJS)
    assert {k: sorted(v) for k, v in front.items()} == {
        0: [0, 1, 2], 1: [3, 4], 2: [5]}


def test_non_dominated_sorting_keeps_identical_points_together():
    objs = np.array([[1, 1], [1, 1], [2, 2]])
    front = rf.non_dominated_sorting(3, objs)
    assert sorted(front[0]) == [0, 1]
    assert front[1] == [2]


def test_crowding_distance_boundary_and_interior_values():
    front = {0: [0, 1, 2]}
    distance = rf.calculate_crowding_distance(front, OBJS)
    # Boundary solutions get both large constants; the middle one gets
    # (3 - 1) / (3 - 1) + (5 - 1) / (5 - 1) = 2.
    assert distance[0][0] == 666666666 + 333333333
    assert distance[0][2] == 666666666 + 333333333
    assert distance[0][1] == pytest.approx(2.0)


def test_crowding_distance_has_one_entry_per_solution():
    front = rf.non_dominated_sorting(len(OBJS), OBJS)
    distance = rf.calculate_crowding_distance(front, OBJS)
    assert sorted(i for f in distance.values() for i in f) == list(range(6))


def test_fitness_sort_orders_by_front_then_distance():
    distance = {0: {0: 999.0, 1: 2.0, 2: 999.0}, 1: {3: 5.0, 4: 7.0}, 2: {5: 1.0}}
    assert rf.fitness_sort(distance, 6) == [0, 2, 1, 4, 3, 5]


# --- Selection, crossover and mutation ---------------------------------------

def test_fitter_parent_returns_fitter_of_the_pair():
    # With two solutions the pair is always (0, 1); solution 1 ranks first.
    sorted_fitness = np.array([1, 0])
    for seed in range(20):
        np.random.seed(seed)
        assert rf.fitter_parent(sorted_fitness, 2) == 1


def test_fitter_parent_returns_valid_index():
    sorted_fitness = np.array([3, 1, 0, 2, 4])
    np.random.seed(0)
    for _ in range(50):
        assert 0 <= rf.fitter_parent(sorted_fitness, 5) < 5


def test_crossover_combines_parent_genes():
    pop = np.array([[0] * 6, [1] * 6])
    for seed in range(30):
        np.random.seed(seed)
        c1, c2 = rf.crossover_random_single_point_swap(pop, 0, 1, 1, 5)
        assert c1.shape == c2.shape == (6,)
        # child 1 = zeros then ones, child 2 = ones then zeros, cut in [1, 5)
        cut1, cut2 = int((c1 == 0).sum()), int((c2 == 1).sum())
        assert 1 <= cut1 < 5 and 1 <= cut2 < 5
        assert np.array_equal(c1, np.r_[np.zeros(cut1), np.ones(6 - cut1)])
        assert np.array_equal(c2, np.r_[np.ones(cut2), np.zeros(6 - cut2)])


def test_mutation_keeps_binary_values_and_distinct_children():
    np.random.seed(5)
    for _ in range(30):
        c1 = np.random.randint(2, size=8)
        c2 = np.random.randint(2, size=8)
        m1, m2 = rf.mutation_random_bitflip(c1, c2, 8, 0.4, 8)
        assert set(np.unique(m1)) <= {0, 1} and set(np.unique(m2)) <= {0, 1}
        assert not np.array_equal(m1, m2)


def test_mutation_separates_identical_children_even_without_mutation():
    np.random.seed(6)
    c1 = np.array([0, 1, 0, 1])
    c2 = c1.copy()
    m1, m2 = rf.mutation_random_bitflip(c1, c2, 4, 0.0, 4)
    assert np.array_equal(m1, [0, 1, 0, 1])
    assert not np.array_equal(m1, m2)


def test_mutation_with_probability_one_flips_one_gene_each():
    np.random.seed(7)
    c1 = np.zeros(10, dtype=int)
    c2 = np.ones(10, dtype=int)
    m1, m2 = rf.mutation_random_bitflip(c1.copy(), c2.copy(), 10, 1.0, 10)
    assert (m1 != c1).sum() == 1
    assert (m2 != c2).sum() == 1


# --- Duplicate handling --------------------------------------------------------

def test_remove_duplicate_list():
    records = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]])
    uniq, removed = rf.remove_duplicate_list(records)
    assert np.array_equal(uniq, [[1, 2], [3, 4], [5, 6]])
    assert sorted(removed) == [2, 4]


def test_remove_duplicate_same_population():
    pop = np.array([[0, 1], [1, 1], [0, 1], [0, 0]])
    uniq, removed = rf.remove_duplicate_same_population(pop)
    assert np.array_equal(uniq, [[0, 1], [1, 1], [0, 0]])
    assert removed == [2]


def test_remove_duplicate_same_population_without_duplicates():
    pop = np.array([[0, 1], [1, 1]])
    uniq, removed = rf.remove_duplicate_same_population(pop)
    assert np.array_equal(uniq, pop) and removed == []


def test_remove_duplicate_different_population():
    pop1 = np.array([[0, 0], [0, 1], [1, 1]])
    pop2 = np.array([[0, 1], [1, 0]])
    assert np.array_equal(rf.remove_duplicate_different_population(pop1, pop2),
                          [[0, 0], [1, 1]])


def test_remove_same_objectives_population():
    pop = np.array([[0, 0], [0, 1], [1, 1]])
    assert np.array_equal(rf.remove_same_objectives_population(pop, [1]),
                          [[0, 0], [1, 1]])
    assert np.array_equal(rf.remove_same_objectives_population(pop, []), pop)


def test_separate_new_old():
    new = np.array([[0, 0], [0, 1], [1, 1]])
    old = np.array([[0, 1], [1, 0]])
    new_chroms, old_chroms, old_idx = rf.separate_new_old(new, old)
    assert np.array_equal(new_chroms, [[0, 0], [1, 1]])
    assert np.array_equal(old_chroms, [[0, 1]])
    assert old_idx == [1]


def test_final_exposure_cost():
    costs, exposures = rf.final_exposure_cost([10, 20, 30], [1, 2, 3], [2, 0], 2)
    assert costs == [30, 10] and exposures == [3, 1]

from typing import Tuple

import numpy as np
import pandas as pd
from brainio.assemblies import walk_coords, NeuroidAssembly
from matplotlib import pyplot
from numpy.random import RandomState
from scipy.stats import pearsonr
from tqdm import tqdm

REPETITION_COORDS = ['repetition', 'eye_h_degrees', 'eye_v_degrees', 'eye_time_ms', 'samp_on_us', 'photodiode_on_us',
                     'session_date_str', 'session_time_str', 'session_datetime', 'intan_session_identifier',
                     'mwk_session_identifier', 'presentation_id', 'stimulus_order_in_trial',
                     'normalizer_assembly_identifier']
""" all the coordinates that are not shared across repeated presentations of the same stimulus """


def average_repetitions(assembly: NeuroidAssembly) -> NeuroidAssembly:
    repetition_dims = assembly['presentation'].dims
    nonrepetition_coords = [coord for coord, dims, values in walk_coords(assembly)
                            if dims == repetition_dims and coord not in REPETITION_COORDS]
    average = assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)
    return average


def internal_consistency(assembly: NeuroidAssembly, num_splits: int = 10) -> pd.DataFrame:
    consistencies = []
    random_state = RandomState(0)
    repetitions = list(sorted(set(assembly['repetition'].values)))
    for split in range(num_splits):
        repetitions_half1 = random_state.choice(repetitions, size=len(repetitions) // 2, replace=False)
        half1 = assembly[{'presentation': [repetition in repetitions_half1
                                           for repetition in assembly['repetition'].values]}]
        half2 = assembly[{'presentation': [repetition not in repetitions_half1
                                           for repetition in assembly['repetition'].values]}]
        half1 = average_repetitions(half1)
        half2 = average_repetitions(half2)
        # align halves
        half1 = half1.sortby('stimulus_id')
        half2 = half2.sortby('stimulus_id')
        assert (half1['stimulus_id'].values == half2['stimulus_id'].values).all()
        # compute correlation per neuroid
        for neuroid_id in assembly['neuroid_id'].values:
            neuroid_correlation, _ = pearsonr(half1.sel(neuroid_id=neuroid_id).squeeze('neuroid'),
                                              half2.sel(neuroid_id=neuroid_id).squeeze('neuroid'))
            consistencies.append({'split': split, 'neuroid_id': neuroid_id, 'correlation': neuroid_correlation})
    consistencies = pd.DataFrame(consistencies)
    return consistencies


def internal_consistency_over_time(assembly: NeuroidAssembly) -> pd.DataFrame:
    results = []
    for time_bin_start in tqdm(assembly['time_bin_start_ms'].values, desc='time_bin'):
        bin_assembly = assembly.sel(time_bin_start_ms=time_bin_start).squeeze('time_bin')
        correlations = internal_consistency(bin_assembly)
        correlations['time_bin_start'] = time_bin_start
        results.append(correlations)
    results = pd.concat(results)
    return results


def time_averaged_consistencies(assembly: NeuroidAssembly, time_range: Tuple[int, int] = (70, 170)) -> pd.DataFrame:
    time_start, time_end = time_range
    assembly = assembly[{'time_bin': [time_start <= start <= time_end
                                      for start in assembly['time_bin_start_ms'].values]}]
    assembly = assembly.mean('time_bin')
    consistencies = internal_consistency(assembly)
    consistencies = consistencies.groupby('neuroid_id').mean().reset_index()  # average over splits
    return consistencies


def filter_assembly(assembly: NeuroidAssembly, consistency_threshold: float = 0.7,
                    time_range: Tuple[int, int] = (70, 170)) -> NeuroidAssembly:
    consistencies = time_averaged_consistencies(assembly, time_range=time_range)
    good_neuroids = consistencies['neuroid_id'][consistencies['correlation'] > consistency_threshold]
    good_neuroids = good_neuroids.values
    filtered_assembly = assembly[{'neuroid': [neuroid_id in good_neuroids
                                              for neuroid_id in assembly['neuroid_id'].values]}]
    return filtered_assembly


def plot_consistencies(assembly: NeuroidAssembly, time_range: Tuple[int, int] = (70, 170)):
    consistencies = time_averaged_consistencies(assembly, time_range=time_range)
    fig, ax = pyplot.subplots()
    ax.hist(consistencies['correlation'], bins=np.arange(0, 1.1, .1), width=0.08, color='darkgreen')
    ax.set_xlabel('internal consistency [r]')
    ax.set_ylabel('count')


def plot_consistencies_survival_count(assembly: NeuroidAssembly, time_range: Tuple[int, int] = (70, 170)):
    consistencies = time_averaged_consistencies(assembly, time_range=time_range)
    # compute histogram
    histogram_base = np.arange(0, 1.1, .1)
    histogram_values, histogram_base = np.histogram(consistencies['correlation'], bins=histogram_base)
    cumulative = np.cumsum(histogram_values[::-1])[::-1]
    # plot the cumulative function
    fig, ax = pyplot.subplots()
    x, y = histogram_base[:-1], cumulative
    ax.bar(x, y, width=0.08, color='darkgreen')
    for _x, _y in zip(x, y):
        ax.text(_x, _y, _y, ha='center', va='bottom')
    ax.set_xlabel('internal consistency [r]')
    ax.set_ylabel('survival count')
    ax.set_xticks(histogram_base[:-1])


def plot_internal_consistency_over_time(assembly: NeuroidAssembly):
    results = internal_consistency_over_time(assembly)

    fig, ax = pyplot.subplots()
    for neuroid_id in set(results['neuroid_id']):
        neuroid_results = results[results['neuroid_id'] == neuroid_id]
        neuroid_results = neuroid_results.groupby('time_bin_start').mean().reset_index()  # average over splits
        ax.plot(neuroid_results['time_bin_start'], neuroid_results['correlation'], color='gray', alpha=0.5)

    aggregate_results = results.groupby(['time_bin_start', 'neuroid_id']).mean().reset_index()  # average over splits
    aggregate_results = aggregate_results.groupby(['time_bin_start']).median('neuroid_id').reset_index()  # avg neuroids

    ax.plot(aggregate_results['time_bin_start'], aggregate_results['correlation'], color='darkgreen', linewidth=5)
    ax.set_xlabel('time bin start [ms]')
    ax.set_ylabel('split-half consistency [r]')

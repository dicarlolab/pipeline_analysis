from typing import Iterable

import numpy as np
import pandas as pd
from brainio.assemblies import NeuroidAssembly
from matplotlib import pyplot
from numpy.random import RandomState
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def plot_sites_vs_accuracy(assembly: NeuroidAssembly, sites: Iterable[int] = (1, 5, 10, 20, 40)):
    assembly = assembly.transpose('presentation', 'neuroid')
    results = []

    random_state = RandomState(0)
    for num_sites in tqdm(sites, desc='num sites'):
        for split in range(10):
            sites = random_state.choice(assembly['neuroid_id'].values, size=num_sites, replace=False)
            sites_assembly = assembly[
                {'neuroid': [neuroid_id in sites for neuroid_id in assembly['neuroid_id'].values]}]
            # train/test
            stimulus_ids = sites_assembly['stimulus_id'].values
            train_stimuli, test_stimuli = train_test_split(stimulus_ids, test_size=0.1, random_state=random_state)
            train_assembly = sites_assembly[{'presentation': [stimulus_id in train_stimuli
                                                              for stimulus_id in sites_assembly['stimulus_id'].values]}]
            test_assembly = sites_assembly[{'presentation': [stimulus_id in test_stimuli
                                                             for stimulus_id in sites_assembly['stimulus_id'].values]}]
            # run classifier
            classifier = RidgeClassifierCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], fit_intercept=True)
            classifier.fit(train_assembly, train_assembly['image_label'])
            test_score = classifier.score(test_assembly, test_assembly['image_label'])
            results.append({'num_sites': num_sites, 'split': split, 'test_score': test_score})
    results = pd.DataFrame(results)
    results = results.groupby('num_sites') \
        .agg(mean=('test_score', np.mean), std=('test_score', np.std)) \
        .reset_index()

    # plot
    fig, ax = pyplot.subplots()
    ax.errorbar(results['num_sites'], results['mean'], yerr=results['std'])
    ax.set_xlabel('# sites')
    ax.set_ylabel('accuracy')


if __name__ == '__main__':
    import brainio
    from brainio.assemblies import walk_coords
    from pipeline_analysis.internal_consistency import filter_assembly
    from pipeline_analysis import average_time_and_repetitions

    raw_assembly = brainio.get_assembly('dicarlo.MajajHong2015.temporal')
    raw_assembly = raw_assembly.sel(region='IT')
    # adjust to new format
    raw_assembly['region'] = 'neuroid', ['IT'] * len(raw_assembly['neuroid'])
    raw_assembly['time_bin_start_ms'] = raw_assembly['time_bin_start']
    raw_assembly['time_bin_stop_ms'] = raw_assembly['time_bin_end']
    raw_assembly['image_label'] = raw_assembly['category_name']
    # for some reason, there are nan object_names
    stimuli_filter = [isinstance(v, str) or not np.isnan(v) for v in raw_assembly['object_name'].values]
    raw_assembly = raw_assembly[{'presentation': stimuli_filter}]
    raw_assembly = type(raw_assembly)(raw_assembly.values, coords={
        coord: (dims, values) for coord, dims, values in walk_coords(raw_assembly)},
                                      dims=raw_assembly.dims)  # reindex

    filtered_assembly = filter_assembly(raw_assembly, consistency_threshold=0.7)

    assembly = average_time_and_repetitions(filtered_assembly)

    print(f"Assembly filtered: {assembly}")

    plot_sites_vs_accuracy(assembly)

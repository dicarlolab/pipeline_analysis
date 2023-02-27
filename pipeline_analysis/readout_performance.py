import matplotlib.figure
import numpy as np
import pandas as pd
from brainio.assemblies import NeuroidAssembly
from matplotlib import pyplot
from numpy.random import RandomState
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def plot_sites_vs_accuracy(assembly: NeuroidAssembly):
    results = []

    random_state = RandomState(0)
    for num_sites in tqdm([1, 5, 10, 20, 40], desc='num sites'):
        for split in range(10):
            sites = random_state.choice(assembly['neuroid_id'].values, size=num_sites, replace=False)
            sites_assembly = assembly[
                {'neuroid': [neuroid_id in sites for neuroid_id in assembly['neuroid_id'].values]}]
            # train/test
            stimulus_ids = assembly['stimulus_id'].values
            train_stimuli, test_stimuli = train_test_split(stimulus_ids, test_size=0.1, random_state=random_state)
            train_assembly = assembly[{'presentation': [stimulus_id in train_stimuli
                                                        for stimulus_id in assembly['stimulus_id'].values]}]
            test_assembly = assembly[{'presentation': [stimulus_id in test_stimuli
                                                       for stimulus_id in assembly['stimulus_id'].values]}]
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

import numpy as np
from brainio.assemblies import NeuroidAssembly

from pipeline_analysis.internal_consistency import average_repetitions


def print_assembly_info(assembly: NeuroidAssembly):
    print(f"unique images: {len(set(assembly['stimulus_id'].values))}")
    print("repetitions:", np.unique(assembly['repetition'].values, return_counts=True))
    print()
    print(f"unique neuroid_ids: {len(set(assembly['neuroid_id'].values))}")
    print(f"regions: {sorted(set(assembly['region'].values))}")
    print(f"subregions: {sorted(set(assembly['subregion'].values))}")
    print()
    print(f"time bins: {list(zip(assembly['time_bin_start_ms'].values, assembly['time_bin_stop_ms'].values))}")


def average_time_and_repetitions(assembly: NeuroidAssembly) -> NeuroidAssembly:
    assembly = average_repetitions(assembly)
    assembly = assembly[{'time_bin': [70 <= start <= 170 for start in assembly['time_bin_start_ms'].values]}]
    assembly = assembly.mean('time_bin')
    return assembly

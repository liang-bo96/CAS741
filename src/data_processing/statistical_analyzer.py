"""Statistical analysis functions for EEG data."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import stats
from scipy.stats import ttest_ind, f_oneway

def compute_statistics(
    data: Dict[str, Any],
    methods: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """Perform statistical analysis on EEG data.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing EEG data and metadata
    methods : list, optional
        List of statistical methods to apply. Available methods:
        - 'descriptive': Basic descriptive statistics
        - 'ttest': Independent samples t-test
        - 'anova': One-way ANOVA
        - 'correlation': Pearson correlation
        - 'spectral': Spectral analysis
        - 'permutation': Permutation testing
    **kwargs
        Additional parameters for specific methods:
        - group_labels: list, labels for group comparison
        - n_permutations: int, number of permutations for permutation test
        - alpha: float, significance level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing statistical analysis results
    """
    if methods is None:
        methods = ['descriptive', 'ttest']
    
    eeg_data = data['data']
    n_channels = eeg_data.shape[0]
    n_times = eeg_data.shape[1]
    
    result = {}
    
    # Descriptive statistics
    if 'descriptive' in methods:
        desc_stats = {
            'mean': np.mean(eeg_data, axis=1),
            'std': np.std(eeg_data, axis=1),
            'median': np.median(eeg_data, axis=1),
            'skewness': stats.skew(eeg_data, axis=1),
            'kurtosis': stats.kurtosis(eeg_data, axis=1)
        }
        result['descriptive'] = desc_stats
    
    # Independent samples t-test
    if 'ttest' in methods and 'group_labels' in kwargs:
        group_labels = kwargs['group_labels']
        unique_groups = np.unique(group_labels)
        if len(unique_groups) == 2:
            t_stats = np.zeros(n_channels)
            p_values = np.zeros(n_channels)
            for i in range(n_channels):
                group1 = eeg_data[i, group_labels == unique_groups[0]]
                group2 = eeg_data[i, group_labels == unique_groups[1]]
                t_stats[i], p_values[i] = ttest_ind(group1, group2)
            result['ttest'] = {
                't_statistics': t_stats,
                'p_values': p_values
            }
    
    # One-way ANOVA
    if 'anova' in methods and 'group_labels' in kwargs:
        group_labels = kwargs['group_labels']
        unique_groups = np.unique(group_labels)
        if len(unique_groups) > 2:
            f_stats = np.zeros(n_channels)
            p_values = np.zeros(n_channels)
            for i in range(n_channels):
                groups = [eeg_data[i, group_labels == g] for g in unique_groups]
                f_stats[i], p_values[i] = f_oneway(*groups)
            result['anova'] = {
                'f_statistics': f_stats,
                'p_values': p_values
            }
    
    # Pearson correlation
    if 'correlation' in methods:
        corr_matrix = np.corrcoef(eeg_data)
        result['correlation'] = {
            'matrix': corr_matrix,
            'p_values': np.zeros((n_channels, n_channels))
        }
        # Calculate p-values for correlations
        for i in range(n_channels):
            for j in range(n_channels):
                r = corr_matrix[i, j]
                df = n_times - 2
                t = r * np.sqrt(df / (1 - r**2))
                result['correlation']['p_values'][i, j] = 2 * (1 - stats.t.cdf(abs(t), df))
    
    # Permutation testing
    if 'permutation' in methods and 'group_labels' in kwargs:
        n_permutations = kwargs.get('n_permutations', 1000)
        alpha = kwargs.get('alpha', 0.05)
        group_labels = kwargs['group_labels']
        unique_groups = np.unique(group_labels)
        
        if len(unique_groups) == 2:
            perm_stats = np.zeros((n_channels, n_permutations))
            for i in range(n_permutations):
                # Permute labels
                perm_labels = np.random.permutation(group_labels)
                group1 = eeg_data[:, perm_labels == unique_groups[0]]
                group2 = eeg_data[:, perm_labels == unique_groups[1]]
                # Calculate t-statistics
                perm_stats[:, i] = np.mean(group1, axis=1) - np.mean(group2, axis=1)
            
            # Calculate p-values
            obs_diff = np.mean(eeg_data[:, group_labels == unique_groups[0]], axis=1) - \
                      np.mean(eeg_data[:, group_labels == unique_groups[1]], axis=1)
            p_values = np.mean(np.abs(perm_stats) >= np.abs(obs_diff)[:, np.newaxis], axis=1)
            
            result['permutation'] = {
                'p_values': p_values,
                'significant': p_values < alpha
            }
    
    return result 
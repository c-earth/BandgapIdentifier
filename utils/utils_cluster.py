import numpy as np

from sklearn.mixture import GaussianMixture

def fit_n_gmm(data, n_mixtures, n_features = None, return_model = False):
    '''
    Fit Gaussian Mixture Model with n_mixtures components
    data is expected as np.array of shape (n_data, n_features)
    If n_features is None, infer from data shape.
    Throws error if data shape and n_features disagree.
    Returns bic_score, result(, model)
    bic_score is the Bayesian Information Criterion score for model selection
    result is a list of list of band index i.e. [(group 1)[band1, band2, ...], (group 2)[band1, band2, ...], ...]
    optionally returns model (if return_model is True) as well
    '''
    if n_features is None:
        n_features = data.shape[1]
    if len(data.shape) != 2 or n_features != data.shape[1]:
        raise ValueError(f'incorrect input data dimension, expected (n_bands, n_features), got {data.shape}')

    model = GaussianMixture(n_components=n_mixtures).fit(data)
    bic_score = model.bic(data)
    labels = model.predict(data)
    band_groups = [np.where(labels == i) for i in range(n_mixtures)]
    if return_model:
        return bic_score, band_groups, model
    return bic_score, band_groups

def auto_fit_gmm(data, max_n_mixtures=10, n_features=None):
    '''
    Fit Gaussian Mixture Model and use Bayesian Information Criterion to select number of components (up to max_n_mixture components)
    data is expected as np.array of shape (n_data, n_features)
    If n_features is None, infer from data shape.
    Throws error if data shape and n_features disagree.
    Returns a list of list of band index i.e. [(group 1)[band1, band2, ...], (group 2)[band1, band2, ...], ...]
    '''
    if n_features is None:
        n_features = data.shape[1]
    if len(data.shape) != 2 or n_features != data.shape[1]:
        raise ValueError(f'incorrect input data dimension, expected (n_bands, n_features), got {data.shape}')

    bic_scores = list()
    results = list()
    for i in range(max_n_mixtures):
        bic_score, result = fit_n_gmm(data, i+1, n_features=n_features)
        bic_scores.append(bic_score)
        results.append(result)

    return results[np.argmin(bic_scores)], bic_scores


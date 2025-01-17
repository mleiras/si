from scipy.stats import stats

def f_classification(dataset):
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p


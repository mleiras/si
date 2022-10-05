from scipy import stats

def f_regression(dataset):
    graus = 2 # Corrigir depois
    coeficientes = stats.pearsonr(dataset.X)
    F = ((coeficientes**2)/(1-(coeficientes**2)))*graus    
    p = stats.f.sf(F,1,graus)
    return F, p



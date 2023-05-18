from sklearn.isotonic import IsotonicRegression


def isotonic_regressin(X, y):
    ir = IsotonicRegression()
    ir.fit(X, y)
    return ir

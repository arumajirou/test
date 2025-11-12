# Placeholder for custom metrics (not used in minimal pipeline)
def smape(y_true, y_pred):
    import numpy as np
    denom = (abs(y_true) + abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(abs(y_true - y_pred) / denom) * 100)

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

def calculate_metrics(val_targets, val_map_scores):
    fpr, tpr, thresholds = roc_curve(val_targets, val_map_scores)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    thresh = thresholds[ix]

    val_preds = [1 if x > thresh else 0 for x in val_map_scores]
    tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel()

    frr = fp / (fp + tn)
    print(f'frr = {frr} = {fp} / {fp + tn}')
    far = fn / (tp + fn)
    print(f'far = {far} = {fn} / {tp + fn}')
    print('thresh', thresh)

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(f'accuracy = {accuracy} = {tp + tn} / {tp + tn + fp + fn}')

    return accuracy, far, frr, thresh
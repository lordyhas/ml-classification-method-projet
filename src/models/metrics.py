from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

class Metrics:
    @staticmethod
    def roc_auc(predictions, target):
        """
        Returne :
        - le score AUC lorsque les prédictions et les étiquettes sont fournies.
        """
        fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc
    @staticmethod
    def f1_score():
        pass
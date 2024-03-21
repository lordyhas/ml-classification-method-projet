from sklearn.linear_model import Perceptron

class PerceptronModel(Perceptron):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def weights(self):
        """ Retourne les poids du modèle [PerceptronModel] """
        return self.coef_

    @property
    def bias(self):
        """ Retourne les biais du modèle [PerceptronModel] """
        return self.intercept_

    def training(self, x_train, y_train, coef_init=None, intercept_init=None, sample_weight=None):
        return self.fit(x_train, y_train, coef_init=coef_init, intercept_init=intercept_init, sample_weight=sample_weight)

    def error(self, y, y_pred) -> float:
        pass



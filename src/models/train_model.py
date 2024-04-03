from sklearn.model_selection import GridSearchCV


class CrossValidate(object):

    def __init__(self, params: dict, model, k_fold: int, metric="accuracy"):
        self.params = params
        self.grid_search = GridSearchCV(
            estimator=model, param_grid=self.params,
            cv=k_fold, scoring=metric)
    def train(self, x_train, y_train):
        return self.grid_search.fit(x_train, y_train)
    @property
    def best_params(self):
        return self.grid_search.best_params_
    @property
    def best_model(self):
        return self.grid_search.best_estimator_


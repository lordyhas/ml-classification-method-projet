from sklearn.model_selection import GridSearchCV


class CrossValidate(object):
    def __init__(self, params: dict, model, k_fold: int, metric="accuracy"):
        self.params = params
        self.metric = metric
        self.grid_search = GridSearchCV(
            estimator=model, param_grid=self.params,
            cv=k_fold, scoring=self.metric)

        self.is_trained = False

    def train(self, x_train, y_train):
        # Effectuer la recherche des hyperparamètres
        self.grid_search.fit(x_train, y_train)
        self.is_trained = True

    @property
    def best_params(self):
        # Afficher les meilleurs hyperparamètres trouvés

        return self.grid_search.best_params_ \
            if self.is_trained \
            else {}
    @property
    def best_model(self):
        return self.grid_search.best_estimator_ \
            if self.is_trained \
            else None


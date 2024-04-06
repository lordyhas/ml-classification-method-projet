from sklearn.model_selection import GridSearchCV

class TrainModel(object):
    def __init__(self, model):
        self.model = model
        self.is_trained = False

    def training(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.is_trained = True
        return self.model
    def history(self):
        pass
    def predict(self, x):
         return self.model.predict(x)
    def error(self, x, target):
        pass
    def train_error(self):
        pass
    def train_accuracy(self):
        pass
    def get_trained_model(self):
        return self.model \
            if self.is_trained \
            else None
class CrossValidate(object):
    """
    Cette classe utilise GridSearchCV pour effectuer une validation croisée et une recherche d'hyperparamètres sur un modèle donné.

    Attributs :
        - params (dict): Dictionnaire des hyperparamètres à tester.
        - metric (str): Métrique d'évaluation à utiliser pour la recherche.
        - grid_search (GridSearchCV): Instance de GridSearchCV configurée avec le modèle, les paramètres, le nombre de plis et la métrique.
        - is_trained (bool): Indicateur si la recherche des hyperparamètres a été effectuée.

    Méthodes :
        - train(self, x_train, y_train): Lance la recherche des hyperparamètres sur les données d'entraînement fournies.
        - best_params (property): Retourne les meilleurs hyperparamètres trouvés après l'entraînement.
        - best_model (property): Retourne le meilleur estimateur trouvé après l'entraînement.
    """
    def __init__(self, params: dict, model, k_fold: int, metric="accuracy"):
        """
        Initialise l'instance de CrossValidate avec les paramètres, le modèle, le nombre de plis pour la validation croisée et la métrique d'évaluation.

        Paramètres :
            - params (dict) : Dictionnaire des hyperparamètres à tester.
            - model : Modèle de machine learning à évaluer.
            - k_fold (int) : Nombre de plis pour la validation croisée.
            - metric (str) : Métrique d'évaluation à utiliser pour la recherche.
        """
        self._params = params
        self.metric = metric
        self.grid_search = GridSearchCV(
            estimator=model, param_grid=self._params,
            cv=k_fold, scoring=self.metric)

        self.is_trained = False
    def search_best_params(self, x_train, y_train):
        self.train(x_train, y_train)
    def train(self, x_train, y_train):
        """
        Effectue la recherche des hyperparamètres en utilisant GridSearchCV sur les données d'entraînement.

        Paramètres :
            - x_train: Caractéristiques d'entraînement.
            - y_train: Étiquettes d'entraînement.
        """
        # Effectuer la recherche des hyperparamètres
        # self.grid_search.fit(x_train, y_train)
        # self.is_trained = True
        try:
            self.grid_search.fit(x_train, y_train)
            self.is_trained = True
        except Exception as e:
            print(f"Une erreur est survenue lors de l'entraînement : {e}")

    @property
    def best_params(self):
        # les meilleurs hyperparamètres trouvés

        if self.is_trained:
            return self.grid_search.best_params_
        else:
            print("La recherche des hyperparamètres n'a pas encore été effectuée.")
            return {}

        #return self.grid_search.best_params_ \
            #if self.is_trained \
            #else {}
    @property
    def best_model(self):
        return self.grid_search.best_estimator_ \
            if self.is_trained \
            else None


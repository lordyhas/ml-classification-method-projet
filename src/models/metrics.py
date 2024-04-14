from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from src.models.train_model import CrossValidate

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Measure(object):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        """
        Initialisation de la classe Measure avec les données d'entraînement et de test.

        :param model: Le modèle de sklearn à évaluer.
        :param x_train: Les caractéristiques d'entraînement.
        :param y_train: Les étiquettes d'entraînement.
        :param x_test: Les caractéristiques de test.
        :param y_test: Les étiquettes de test.
        """
        self.model = model
        self.is_ready = False
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.pred = self.model.predict(self.x_test)

    def show_metrics_test(self):
        print("\nTest Score :", )
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')
        print(f"Accuracy : {accuracy}")
        print(f"Precision : {precision}")
        print(f"Recall : {recall}")
        print(f"F1-score : {f1}")

    def show_metrics_train(self):
        print("\nTrain Score:")
        y_pred = self.model.predict(self.x_train)
        accuracy = accuracy_score(self.y_train, y_pred)
        precision = precision_score(self.y_train, y_pred, average='macro')
        recall = recall_score(self.y_train, y_pred, average='macro')
        f1 = f1_score(self.y_train, y_pred, average='macro')
        print(f"Accuracy : {accuracy}")
        print(f"Precision : {precision}")
        print(f"Recall : {recall}")
        print(f"F1-score : {f1}")

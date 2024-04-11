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

    def calculate_loss(self, loss='neg_log_loss'):
        """
        Calcule la perte du modèle.

        :param loss: La métrique de perte à utiliser.
        :return: La perte moyenne du modèle.
        """
        return -CrossValidate.cv_score(self.model, self.x_test, self.y_test, scoring=loss).mean()

    def calculate_accuracy(self):
        return accuracy_score(self.y_test, self.pred)

    def calculate_precision(self):

        return precision_score(self.y_test, self.pred, average='macro')

    def calculate_recall(self):
        return recall_score(self.y_test, self.pred, average='macro')

    def calculate_f1(self):
        """
        Calcule le score F1 du modèle.

        :return: Le score F1 du modèle.
        """
        return f1_score(self.y_test, self.pred, average='macro')

    def get_metrics(self):
        """
        Obtient toutes les métriques du modèle.

        :return: Un tuple contenant la perte, l'exactitude, la précision, le rappel et le score F1 du modèle.
            - tuple(loss, accuracy, precision, recall, f1)
        """
        #loss = self.calculate_loss()
        accuracy = self.calculate_accuracy()
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1()
        self.is_ready = True
        return accuracy, precision, recall, f1

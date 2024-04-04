
import numpy as np # linear algebra
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DownloadData(object):
    def __init__(self, path):
        self.__path = path

    def get_data_as_dataframe(self):
        return

class LeafDataPreprocessing(object):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, normalized=True):
        self.__is_normalized = False
        self._train = train
        #self._test = test
        self._processed_data = self._train.copy()
        self._processed_test_data = self._train.copy()
        self.__set_normalized_and_encode(normalized=normalized)

    def is_normalized(self):
        return self.__is_normalized
    def __set_normalized_and_encode(self, normalized):
        # Initialize the scaler
        scaler = MinMaxScaler()

        if normalized:
            # Initialize the encoder
            le = LabelEncoder()
            # Encode the 'species' column
            self._processed_data['species'] = le.fit_transform(self._processed_data['species'])
            self.__is_normalized = True

        # Scale numeric columns. Exclude 'id' and 'species' from being scaled
        numeric_cols = self._processed_data.columns.drop(['id', 'species'])
        self._processed_data[numeric_cols] = scaler.fit_transform(self._processed_data[numeric_cols])

    def get_encode_label(self):
        """
        La methode retourne les cibles encodées
        - si pre-traité et normalisé
        - sinon une liste vide [DataFrame([])]
        """
        return self._processed_data['species'] if self.__is_normalized else pd.DataFrame([])
    def get_label(self):
        return self._train['species']
    def get_unique_label(self, encoded=False):
        return self.get_label().unique() if not encoded else self.get_encode_label().unique()
    def x_train(self):
        """
        La methode retourne :
        - les features normalisées et pre-traitées
        - les cibles non pre-traitées
        - retourne : les features [normalisée]
        """
        return self._processed_data.drop(columns=['id', 'species'])
    def y_train(self, encode=True):
        """
        Si encode à True cela va encoder les cibles avec un unique identifiant si déjà pre-traitées
        La methode retourne :
        - les cibles (ou labels, classes) normalisées et pre-traitées
        - les cibles non pre-traitées
        """
        if encode:
            return self.get_encode_label()
        else:
            return self.get_label()
    def split_train_and_test(self, x_train=None, y_train=None, ratio=0.2) -> tuple:
        """
        Cette methode divise le training set en train et test
        - retourne : x_train, x_test, y_train, y_test
        """
        if x_train is None or y_train is None:
            x_train = self.x_train()
            y_train = self.y_train()
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_train, test_size=ratio, random_state=42) # Test : 20%
        return x_train, x_test, y_train, y_test
    def x_test(self):
        pass
    def y_test(self):
        pass
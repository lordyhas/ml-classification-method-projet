
import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
class LeafDataPreprocessing(object):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, normalized=True):
        self.is_normalized = False
        self._train = train
        self._test = test
        self._processed_data = self._train.copy()
        self._processed_test_data = self._train.copy()
        self.__set_normalized_and_encode(normalized=normalized)

    def __set_normalized_and_encode(self, normalized):
        # Initialize the scaler
        scaler = MinMaxScaler()

        if normalized:
            # Initialize the encoder
            le = LabelEncoder()
            # Encode the 'species' column
            self._processed_data['species'] = le.fit_transform(self._processed_data['species'])
            self.is_normalized = True

        # Scale numeric columns. Exclude 'id' and 'species' from being scaled
        numeric_cols = self._processed_data.columns.drop(['id', 'species'])
        self._processed_data[numeric_cols] = scaler.fit_transform(self._processed_data[numeric_cols])

    def get_encode_label(self):
        return self._processed_data['species'] if self.is_normalized else pd.DataFrame([])
    def get_label(self):
        return self._train['species']
    def get_unique_label(self, encoded=False):
        return self.get_label().unique() if not encoded else self.get_encode_label().unique()
    def x_train(self):
        return self._processed_data.drop(columns=['id', 'species'])
    def y_train(self, encode=True):
        if encode:
            return self.get_encode_label()
        else:
            return self.get_label()
    def split_train_and_valid(self, x_train, y_train):
        pass
    def x_test(self):
        pass
    def y_test(self):
        pass
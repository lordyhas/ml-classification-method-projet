import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
class NormalityDistributionCheck(object):
    """
    Cette classe fournit des méthodes pour tester la normalité d'un ensemble de données.

    Attributs :
        - data (np.ndarray) : Les données à tester, converties en un tableau numpy si elles sont fournies sous forme de DataFrame pandas.

    Méthodes :
        - shapiro_wilk_test(self) : Effectue le test de Shapiro-Wilk sur les données.
        - kolmogorov_smirnov_test(self) : Effectue le test de Kolmogorov-Smirnov sur les données.
        - anderson_darling_test(self) : Effectue le test d'Anderson-Darling sur les données.
    """

    def __init__(self, data, dist: str = 'norm'):
        """
        Initialise l'objet avec les données fournies.

        Paramètres :
            data (np.ndarray ou pd.DataFrame): Les données à tester.
            dist (str): La distribution à utiliser pour le test, 'norm' par défaut pour la distribution normale.
        """
        if data is pd.DataFrame:
            data.to_numpy()

        self.data = data
        self.dist = dist
        self.__threshold: float = 0.05

    def shapiro_wilk_test(self):
        """
        Effectue le test de Shapiro-Wilk sur les données pour tester la normalité.

        Affiche la statistique de test et la valeur-p, et interprète la valeur-p pour déterminer si les données suivent une distribution normale.
        """
        shapiro_results = stats.shapiro(self.data.flatten())

        print(f"Valeur de la statistique de test (Shapiro-Wilk) : {shapiro_results[0]}")
        print(f"Valeur-p (p-value) : {shapiro_results[1]}")

        # Interprétation de la valeur-p
        if shapiro_results[1] > 0.05:
            print("L'hypothèse de normalité n'est pas rejetée.")
        else:
            print("L'hypothèse de normalité est rejetée.")

    def kolmogorov_smirnov_test(self):
        """
        Effectue le test d'Anderson-Darling sur les données pour tester la normalité.

        Affiche la statistique de test et les valeurs critiques pour différents niveaux de signification, et interprète ces valeurs pour déterminer si les données suivent une distribution normale.
        """
        ks_statistic, ks_p_value = stats.kstest(self.data.flatten(), self.dist)

        print(f"Statistique de test (Kolmogorov-Smirnov) : {ks_statistic}")
        print(f"Valeur-p (p-value) : {ks_p_value}")

        # Interprétation de la valeur-p
        if ks_p_value > self.__threshold:
            print("L'hypothèse de normalité n'est pas rejetée.")
        else:
            print("L'hypothèse de normalité est rejetée.")

    def anderson_darling_test(self):
        """
        Effectue le test d'Anderson-Darling sur les données pour tester la normalité.

        Affiche la statistique de test et les valeurs critiques pour différents niveaux de signification, et interprète ces valeurs pour déterminer si les données suivent une distribution normale.
        """
        ad_test = stats.anderson(self.data.flatten(), dist=self.dist)

        print(f"Statistique de test (Anderson-Darling) : {ad_test.statistic}")
        for i in range(len(ad_test.critical_values)):
            sl, cv = ad_test.significance_level[i], ad_test.critical_values[i]
            if ad_test.statistic < cv:
                print(f"A un niveau de signification de {sl}%, les données semblent normales (statistique < {cv}).")
            else:
                print(
                    f"A un niveau de signification de {sl}%, les données ne semblent pas normales (statistique >= {cv}).")
    @classmethod
    def qq_plot(cls, x_train):
        sm.qqplot(x_train, line='45')
        plt.show()

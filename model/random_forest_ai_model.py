import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class RandomForestModel:
    def __init__(self, data_path, target_column):
        """
        Random Forest Model sınıfı.
        :param data_path: Veri setinin dosya yolu (CSV formatında).
        :param target_column: Hedef sütunun adı.
        """
        self.data_path = data_path
        self.target_column = target_column
        self.model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Veriyi yükler ve eğitim/test setlerine böler.
        """
        data = pd.read_csv(self.data_path)
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Veri başarıyla yüklendi ve eğitim/test setlerine bölündü.")

    def optimize_hyperparameters(self, param_grid=None):
        """
        Random Forest modelinin hiperparametrelerini optimize eder.
        :param param_grid: Hiperparametre aralığı (varsayılan olarak bir grid kullanılır).
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
            }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        print("En iyi hiperparametreler:", self.best_params)

    def train_model(self):
        """
        Modeli eğitim verisiyle eğitir.
        """
        if self.model is None:
            self.model = RandomForestClassifier(random_state=42, **(self.best_params or {}))
        self.model.fit(self.X_train, self.y_train)
        print("Model başarıyla eğitildi.")

    def evaluate_model(self):
        """
        Modeli test verisiyle değerlendirir.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Doğruluk skoru:", accuracy)
        print("Sınıflandırma raporu:\n", classification_report(self.y_test, y_pred))

    def feature_importance(self):
        """
        Özelliklerin önem derecesini gösterir.
        """
        if self.model is None:
            print("Model henüz eğitilmedi.")
            return

        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nÖzellik Önem Skorları:")
        print(importance_data)

    def predict(self, X_new):
        """
        Yeni veriler üzerinde tahmin yapar.
        :param X_new: Yeni veri (DataFrame veya ndarray formatında).
        :return: Tahmin edilen sınıflar.
        """
        if self.model is None:
            print("Model henüz eğitilmedi.")
            return

        predictions = self.model.predict(X_new)
        return predictions


"""
Metotların Açıklaması:
load_data: Veriyi yükler ve eğitim/test setlerine böler.
optimize_hyperparameters: Hiperparametre optimizasyonu yapar.
train_model: Modeli eğitir.
evaluate_model: Modeli değerlendirir ve performans ölçümleri sağlar.
feature_importance: Özelliklerin önem derecesini gösterir.
predict: Yeni veriler üzerinde tahmin yapar.
Not:
data_path değerine, veri setinizin CSV dosya yolu atanmalıdır.
target_column, hedef sınıfı içeren sütunun adı olmalıdır.
Hiperparametre optimizasyonu zaman alabilir, bu yüzden küçük veri setlerinde başlamak faydalı olur.
"""
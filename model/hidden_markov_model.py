import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
import joblib

class MutationPredictionHMM:
    def __init__(self, n_states=2, observation_map=None, states=None, startprob=None, transmat=None, emissionprob=None):
        self.n_states = n_states
        self.observation_map = observation_map if observation_map else {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}  # 'N' geçersiz baz olarak ekleniyor
        self.states = states if states else ['Mutasyon Yok', 'Mutasyon Var']
        self.model = hmm.MultinomialHMM(n_components=self.n_states, random_state=42)

        self.model.startprob_ = startprob if startprob is not None else np.array([0.5, 0.5])
        self.model.transmat_ = transmat if transmat is not None else np.array([[0.7, 0.3], [0.4, 0.6]])
        self.model.emissionprob_ = emissionprob if emissionprob is not None else np.array([[0.25, 0.25, 0.25, 0.25], [0.2, 0.2, 0.3, 0.3]])

    def validate_params(self):
        """Model parametrelerini doğrulama."""
        if self.model.startprob_.sum() != 1.0:
            raise ValueError("Başlangıç olasılıklarının toplamı 1 olmalıdır")
        if np.any(self.model.transmat_.sum(axis=1) != 1.0):
            raise ValueError("Geçiş olasılıklarının her satırının toplamı 1 olmalıdır")
        if np.any(self.model.emissionprob_.sum(axis=1) != 1.0):
            raise ValueError("Emisyon olasılıklarının her satırının toplamı 1 olmalıdır")

    def fit(self, train_sequences):
        """
        Modeli eğitim verisiyle eğitmek.
        :param train_sequences: Eğitim verisi (DNA dizileri)
        """
        # Eğitim verisini uygun formata dönüştürme
        if not all(isinstance(seq, list) for seq in train_sequences):
            raise ValueError("Tüm eğitim dizileri liste olmalıdır")
        
        # Geçersiz bazlar için 'N' kullanma
        train_sequences = [np.array([self.observation_map.get(b, 4) for b in seq]).reshape(-1, 1) for seq in train_sequences]
        self.model.fit(np.concatenate(train_sequences))
    
    def predict(self, test_sequence):
        """
        Verilen test dizisi ile mutasyon tahmini yapmak.
        :param test_sequence: Test verisi (DNA dizisi)
        :return: En olasılıklı durumlar (Mutasyon Yok/Mutasyon Var)
        """
        # Eğer test_sequence bir liste listesi ise düzleştir
        if isinstance(test_sequence[0], list):  
            test_sequence = [item for sublist in test_sequence for item in sublist]
        
        # Test dizisini sayılarla temsil etme
        test_sequence = np.array([self.observation_map.get(b, 4) for b in test_sequence]).reshape(-1, 1)
        
        # En olasılıklı durumları tahmin etme
        logprob, predicted_states = self.model.decode(test_sequence, algorithm="viterbi")
        
        # Sonuçları döndürme
        predictions = [self.states[i] for i in predicted_states]
        print(f"Tahmin uzunluğu: {len(predictions)}, Test sekansı uzunluğu: {len(test_sequence)}")
        return predictions


        
    def predict_multiple(self, test_sequences):
        """
        Birden fazla test dizisi için tahmin yapar.
        :param test_sequences: Test dizileri (sayılara dönüştürülmüş)
        :return: Her bir test dizisi için tahmin edilen etiketler
        """
        all_sequences = np.concatenate([
            np.array([self.observation_map.get(str(b), 4) for b in seq]).reshape(-1, 1)  # Her bir öğeyi string’e dönüştür
            for seq in test_sequences
        ])

        # Burada tüm dizileri birleştirdikten sonra, Hidden Markov Model tahminini yapabiliriz
        predicted_labels = self.model.predict(all_sequences)

        # Burada her test dizisi için tahminleri saklayalım
        predicted_labels_per_sequence = []
        idx = 0
        for seq in test_sequences:
            length = len(seq)
            predicted_labels_per_sequence.append(predicted_labels[idx:idx + length])
            idx += length

        return predicted_labels_per_sequence

    def evaluate(self, test_sequences):
        """
        Verilen test sekanslarını değerlendirir ve doğruluğu hesaplar.
        :param test_sequences: Test sekansları (liste listesi)
        :return: Genel doğruluk ve sekans başına doğruluk oranları
        """
        total_correct = 0
        total_count = 0
        per_sequence_accuracy = []

        for idx, test_seq in enumerate(test_sequences):
            # `predict` fonksiyonunu çağır ve tahminleri al
            predictions = self.predict(test_seq)

            # Uzunluk kontrolü
            if len(test_seq) != len(predictions):
                print(f"Uzunluk uyumsuzluğu! Test sekansı uzunluğu: {len(test_seq)}, Tahmin uzunluğu: {len(predictions)}")
                continue

            # Doğru tahminleri say
            correct = sum([1 for p, g in zip(predictions, test_seq) if p == g])
            per_sequence_accuracy.append(correct / len(test_seq))
            total_correct += correct
            total_count += len(test_seq)

            print(f"Test dizisi {idx+1}: Doğru tahmin sayısı: {correct}, Doğruluk: {correct / len(test_seq) * 100:.2f}%")

        # Genel doğruluk hesaplama
        if total_count > 0:
            accuracy = total_correct / total_count
        else:
            print("Hata: Toplam test sekansı uzunluğu sıfır!")
            accuracy = 0.0

        print(f"Genel doğruluk: {accuracy * 100:.2f}%")
        return accuracy, per_sequence_accuracy



        
    def save_model(self, filename):
        model_data = {
            'model': self.model,
            'observation_map': self.observation_map,
            'states': self.states
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.observation_map = model_data['observation_map']
        self.states = model_data['states']
        return self  # self nesnesini döndürüyoruz


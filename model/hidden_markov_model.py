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
        # Test dizisini sayılarla temsil etme
        test_sequence = np.array([self.observation_map.get(b, 4) for b in test_sequence]).reshape(-1, 1)
        
        # En olasılıklı durumları tahmin etme
        logprob, predicted_states = self.model.decode(test_sequence, algorithm="viterbi")
        
        # Sonuçları döndürme
        return [self.states[i] for i in predicted_states]
    
    def predict_multiple(self, test_sequences):
        """
        Birden fazla test dizisi için tahmin yapmak.
        :param test_sequences: Test dizileri
        :return: En olasılıklı durumlar
        """
        all_sequences = np.concatenate([np.array([self.observation_map.get(b, 4) for b in seq]).reshape(-1, 1) for seq in test_sequences])
        logprob, predicted_states = self.model.decode(all_sequences, algorithm="viterbi")
        
        predictions = []
        idx = 0
        for seq in test_sequences:
            length = len(seq)
            predictions.append([self.states[i] for i in predicted_states[idx:idx+length]])
            idx += length
        return predictions
        
    def evaluate(self, test_sequences, test_labels=None):
        """
        Modelin doğruluğunu değerlendirmek için fonksiyon.
        :param test_sequences: Test verisi (DNA dizileri) 
        :param test_labels: Gerçek etiketler (isteğe bağlı)
        """
        predicted_states = [self.predict(seq) for seq in test_sequences]
        predicted_labels = [state[0] for state in predicted_states]  # İlk durum tahminini alıyoruz

        # Eğer test_labels None değilse doğruluk hesapla
        if test_labels is not None:
            accuracy = accuracy_score(test_labels, predicted_labels)
            per_sequence_accuracy = [accuracy_score(test_labels[i], predicted_labels[i]) for i in range(len(test_sequences))]
        else:
            accuracy = None
            per_sequence_accuracy = None

        return accuracy, per_sequence_accuracy
        
    def save_model(self, filename):
        model_data = {
            'model': self.model,
            'observation_map': self.observation_map,
            'states': self.states
        }
        joblib.dump(model_data, filename)

    def load_model(self, filename):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.observation_map = model_data['observation_map']
        self.states = model_data['states']

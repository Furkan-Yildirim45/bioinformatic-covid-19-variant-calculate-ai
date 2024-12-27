import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
import joblib

class MutationPredictionHMM:
    def __init__(self, n_states=2, observation_map=None, states=None, startprob=None, transmat=None, emissionprob=None):
        self.n_states = n_states
        self.observation_map = observation_map if observation_map else {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.states = states if states else ['Mutasyon Yok', 'Mutasyon Var']
        self.model = hmm.MultinomialHMM(n_components=self.n_states, random_state=42)

        # Parametrelerin dışarıdan verilmesi sağlanabilir
        self.model.startprob_ = startprob if startprob is not None else np.array([0.5, 0.5])
        self.model.transmat_ = transmat if transmat is not None else np.array([[0.7, 0.3], [0.4, 0.6]])
        self.model.emissionprob_ = emissionprob if emissionprob is not None else np.array([[0.25, 0.25, 0.25, 0.25], [0.2, 0.2, 0.3, 0.3]])

    """Bu fonksiyonu fit metodundan önce çağırarak parametrelerin doğru olduğundan emin olabilirsiniz."""
    def validate_params(self):
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
        train_sequences = [np.array([self.observation_map[b] for b in seq]).reshape(-1, 1) for seq in train_sequences]
        self.model.fit(np.concatenate(train_sequences))
    
    def predict(self, test_sequence):
        """
        Verilen test dizisi ile mutasyon tahmini yapmak.
        :param test_sequence: Test verisi (DNA dizisi)
        :return: En olasılıklı durumlar (Mutasyon Yok/Mutasyon Var)
        """
        # Test dizisini sayılarla temsil etme
        test_sequence = np.array([self.observation_map[b] for b in test_sequence]).reshape(-1, 1)
        
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
        all_sequences = np.concatenate([np.array([self.observation_map[b] for b in seq]).reshape(-1, 1) for seq in test_sequences])
        logprob, predicted_states = self.model.decode(all_sequences, algorithm="viterbi")
        
        predictions = []
        idx = 0
        for seq in test_sequences:
            length = len(seq)
            predictions.append([self.states[i] for i in predicted_states[idx:idx+length]])
            idx += length
        return predictions
        
    def evaluate(self, test_sequences, true_labels):
        """
        Modelin doğruluğunu değerlendirmek için fonksiyon.
        :param test_sequences: Test verisi (DNA dizileri)
        :param true_labels: Gerçek etiketler
        """
        predicted_states = [self.predict(seq) for seq in test_sequences]
        predicted_labels = [state[0] for state in predicted_states]  # İlk durum tahminini alıyoruz
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        per_sequence_accuracy = [accuracy_score(true_labels[i], predicted_labels[i]) for i in range(len(test_sequences))]
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

# Kullanım Örneği:
if __name__ == "__main__":
    # Eğitim verisi (DNA dizileri)
    train_sequences = [
        ['A', 'T', 'G', 'C'],  # Normal dizi
        ['A', 'T', 'A', 'G'],  # Normal dizi
        ['T', 'G', 'T', 'C'],  # Normal dizi
        ['A', 'A', 'G', 'G'],  # Mutasyon içeren dizi
    ]

    # Test dizisi
    test_sequence = ['A', 'T', 'G', 'C']  # Test verisi (normal dizi)

    # HMM modelini oluşturma
    mutation_model = MutationPredictionHMM()

    # Modeli eğitim verisiyle eğitme
    mutation_model.fit(train_sequences)

    # Test dizisiyle tahmin yapma
    predicted_states = mutation_model.predict(test_sequence)

    # Sonuçları yazdırma
    print(f"Test Dizisi: {test_sequence}")
    print(f"En Olasılıklı Durumlar: {', '.join(predicted_states)}")

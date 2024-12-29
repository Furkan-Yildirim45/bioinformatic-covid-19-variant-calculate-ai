from hmmlearn import hmm
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

from model.hidden_markov_model import MutationPredictionHMM

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(list(sequence))
                    sequence = ""
            else:
                sequence += line
        if sequence:  # Add the last sequence
            sequences.append(list(sequence))
    return sequences

# Ana kod
if __name__ == "__main__":
    
    # Dosyayı oku
    file_path = "data/SARS-genomes.fa"  # Yüklediğiniz dosya adı
    train_sequences = read_fasta(file_path)

    # Eğitim verisi çıktısını kontrol et
    test_sequence = train_sequences.pop(0)  # İlk sırayı test verisi olarak kullan
    print(f"Eğitim verisi (ilk 3 sıra): {train_sequences[:3]}")
    
    # Modeli oluştur
    mutation_hmm = MutationPredictionHMM()

    # Modeli eğit
    mutation_hmm.fit(train_sequences)

    # Test dizisiyle tahmin yap
    predictions = mutation_hmm.predict(test_sequence)
    print(f"Tahmin edilen durumlar: {predictions}")

    # Doğruluğu hesaplayın
    accuracy, per_sequence_accuracy = mutation_hmm.evaluate([test_sequence], [predictions])

    # Genel doğruluğu yazdır
    print(f"Genel doğruluk: {accuracy * 100:.2f}%")

    # Her bir test dizisi için doğruluk
    for idx, acc in enumerate(per_sequence_accuracy):
        print(f"Test dizisi {idx+1} için doğruluk: {acc * 100:.2f}%")


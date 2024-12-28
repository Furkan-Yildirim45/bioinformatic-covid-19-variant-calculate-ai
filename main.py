from sklearn.model_selection import train_test_split
from data_processing.genetic_data_processing import GeneticDataProcessor
from model.hidden_markov_model import MutationPredictionHMM


def process_data(file_path):
    """
    Veriyi işleme ve tüm dizileri ve etiketleri döndürme.
    """
    processor = GeneticDataProcessor(file_path)
    processor.process_data()

    # Tüm diziler (numeric_sequences) ve etiketler (encoded_labels)
    sequences = processor.numeric_sequences
    labels = processor.encoded_labels  # İşlenmiş etiketler

    print(f"Toplam diziler: {len(sequences)}")
    print(f"Toplam etiketler: {len(labels)}")

    return sequences, labels


def educate_model(train_sequences, train_labels, save_path='model_ai/mutation_model.pkl'):
    """
    Modeli eğitim verisi ile eğit ve kaydet.
    """
    hmm_model = MutationPredictionHMM()
    hmm_model.fit(train_sequences)  # Etiketleri kaldır
    print("Model başarıyla eğitildi.")

    hmm_model.save_model(save_path)
    print(f"Model '{save_path}' konumuna kaydedildi.")
    return hmm_model


def evaluate_model(hmm_model, test_sequences, test_labels=None):
    """
    Modeli test verisi üzerinde değerlendir.
    """
    accuracy, per_sequence_accuracy = hmm_model.evaluate(test_sequences, test_labels)

    if accuracy is not None:
        print(f"Test doğruluğu: {accuracy:.2f}")
    else:
        print("Test doğruluğu hesaplanamadı çünkü etiketler sağlanmadı.")
    
    if per_sequence_accuracy is not None:
        print("Dizi başına doğruluk:", per_sequence_accuracy)
    else:
        print("Dizi başına doğruluk hesaplanamadı çünkü etiketler sağlanmadı.")

    return accuracy


def main():
    # Dosya yolu belirle
    file_path = "data/SARS-genomes.fa"  # Gerçek dosya yolunu belirtmelisin

    # Veriyi işle
    sequences, labels = process_data(file_path)

    # Veriyi eğitim ve test kümelerine ayır
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    # Eğitim verisi ile modeli eğit ve kaydet
    hmm_model = educate_model(train_sequences, train_labels)

    # Modeli test verisi ile değerlendir
    accuracy = evaluate_model(hmm_model, test_sequences, test_labels)
    if accuracy is not None:
        print(f"Model başarıyla değerlendirildi. Test doğruluğu: {accuracy:.2f}")
    else:
        print("Model değerlendirmesi sırasında doğruluk hesaplanamadı.")


if __name__ == "__main__":
    main()

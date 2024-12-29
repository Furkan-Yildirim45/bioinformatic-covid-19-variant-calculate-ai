import numpy as np
import joblib
from data_processing.genetic_data_processing import DataProcessing
from model.hidden_markov_model import MutationPredictionHMM

# Ana kod
if __name__ == "__main__":
    
    data_processing = DataProcessing()

    # Dosyayı oku
    train_file = "data/SARS-genomes-train.fa"  # Yüklediğiniz dosya adı
    test_file = "data/SARS-genomes-test.fa"  # Yüklediğiniz dosya adı
    train_sequences = data_processing.read_fasta(train_file)
    
    # Eğitim verisi çıktısını kontrol et
    test_sequence = data_processing.read_fasta(test_file)
    
    # Modeli oluştur
    mutation_hmm = MutationPredictionHMM()

    # Modeli eğit
    mutation_hmm.fit(train_sequences)
    
    # Modeli kaydet
    model_filename = "model_ai/mutation_model.pkl"
    joblib.dump(mutation_hmm, model_filename)
    print(f"Model '{model_filename}' olarak kaydedildi.")
    
    # Modeli yüklemek (Eğer yeniden yüklemek isterseniz)
    loaded_model = joblib.load(model_filename)
    print("Model başarıyla yüklendi.")

    # Test dizisiyle tahmin yap ve doğruluğu hesapla
    total_accuracy = []
    for idx, test_seq in enumerate(test_sequence):
        # Her bir test dizisi için tahmin yap
        predictions = loaded_model.predict(test_seq)
        print(predictions)
        # Eğer model "Mutasyon Var" veya "Mutasyon Yok" gibi etiketler döndürüyorsa
        # tahmin edilen değerleri değerlendirirsiniz.
        # Örneğin, aşağıdaki kodu kullanarak doğruluğu hesaplayabilirsiniz:
        
        # Burada, tahminlerinizi 2 sınıfa ayırarak doğruluk hesaplayabilirsiniz
        # "Mutasyon Var" için 1 ve "Mutasyon Yok" için 0 değerleri dönecek şekilde işlem yapılabilir
        correct_predictions = sum([1 for p in predictions if p == "Mutasyon Var"])  # Örnek
        print(correct_predictions)
        accuracy = correct_predictions / len(predictions)
        total_accuracy.append(accuracy)
        
        print(f"Test dizisi {idx+1} için doğruluk: {accuracy * 100:.2f}%")
    
    # Genel doğruluğu hesapla
    overall_accuracy = np.mean(total_accuracy)
    print(f"Mutasyon olasılığı: {overall_accuracy * 100:.2f}%")

import numpy as np
from collections import Counter
import logging

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneticDataProcessor:
    def __init__(self, file_path, max_length=5000):
        self.file_path = file_path
        self.max_length = max_length
        self.sequences = []
        self.labels = []
        self.numeric_sequences = []
        self.padded_sequences = []
        self.encoded_labels = []

    def read_data(self):
        """Veri dosyasını oku ve başlıkları ayır."""
        try:
            with open(self.file_path, "r") as file:
                data = file.readlines()

            current_label = None
            for line in data:
                line = line.strip()
                if line.startswith(">"):  # Başlık satırlarını kontrol et
                    parts = line.split(" ", 1)  # Başlık ve açıklamayı ayır
                    current_label = parts[1] if len(parts) > 1 else "Unknown"
                    self.labels.append(current_label)
                elif line:  # Boş olmayan dizilim satırları
                    self.sequences.append(line)
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")
            
    def validate_sequence(self, seq):
        if not all(base in "ATCG" for base in seq):
            raise ValueError(f"Geçersiz genetik baz: {seq}")


    def convert_to_numeric(self):
        """Dizilimleri sayısal verilere dönüştür."""
        nucleotide_mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": -1}

        for seq in self.sequences:
            numeric_sequence = [nucleotide_mapping.get(base, -1) for base in seq]
            self.numeric_sequences.append(numeric_sequence)

        self.fill_invalid_bases()
        logging.info(f"Dizilimler sayısal verilere dönüştürüldü. Toplam {len(self.numeric_sequences)} dizilim işlem gördü.")

    def fill_invalid_bases(self):
        """Geçersiz bazları en yaygın baz ile doldur (NumPy ile)."""
        for i, seq in enumerate(self.numeric_sequences):
            seq = np.array(seq)
            invalid_indices = seq == -1
            if invalid_indices.any():
                valid_bases = seq[~invalid_indices]
                if valid_bases.size > 0:
                    most_common_base = np.bincount(valid_bases).argmax()
                    seq[invalid_indices] = most_common_base
                else:
                    seq[:] = 0  # Tüm dizilim geçersizse 'A' ile doldur
            self.numeric_sequences[i] = seq.tolist()

    def pad_sequences(self, padding_value=0):
        """Dizilimleri belirli bir uzunluğa kadar kes veya doldur (NumPy ile)."""
        if not self.numeric_sequences:
            raise ValueError(
                "Dizilim verileri boş! Lütfen önce verileri yükleyin ve dönüştürün."
            )

        num_sequences = len(self.numeric_sequences)
        padded_array = np.full(
            (num_sequences, self.max_length), padding_value, dtype=int
        )

        for i, seq in enumerate(self.numeric_sequences):
            padded_array[i, : min(len(seq), self.max_length)] = seq[: self.max_length]

        self.padded_sequences = padded_array
        logging.info(f"Dizilimler {self.max_length} uzunluğuna kadar dolduruldu.")

    def encode_labels(self, sort_labels=True):
        """Etiketleri sayısal verilere dönüştür ve benzersiz eşleştirmeyi sakla.
        Args:sort_labels (bool): Etiketleri alfabetik olarak sıralayıp sıralamayacağını belirtir."""
        
        if not self.labels or not all(isinstance(label, str) for label in self.labels):
            raise ValueError("Etiketler boş veya geçersiz! Lütfen verileri kontrol edin.")
        
        unique_labels = sorted(set(self.labels)) if sort_labels else list(set(self.labels))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.encoded_labels = [self.label_mapping[label] for label in self.labels]
        
        logging.info(f"{len(unique_labels)} benzersiz etiket başarıyla kodlandı.")

    def process_data(self, padding_value=0):
        """Verileri okuma, dönüştürme ve hazırlama işlemini tek adımda gerçekleştir."""
        logging.info("Veri işleme başlatıldı...")
        self.read_data()
        self.validate_sequence()
        self.convert_to_numeric()
        self.pad_sequences(padding_value=padding_value)
        self.encode_labels()
        logging.info("Veri işleme tamamlandı.")

    def process_and_display_data(self, padding_value=0):
            """Verileri işleyip detayları ekrana yazdır."""
            try:
                # Verileri işle
                self.process_data(padding_value=padding_value)

                # İşlenmiş verileri inceleyin
                print("İlk 5 işlenmiş dizilim:")
                for seq in self.padded_sequences[:5]:
                    print(seq)

                print("\nEtiketler (Encoded Labels):", self.encoded_labels)

                print("\nEtiket Eşleştirmesi:")
                for label, idx in self.label_mapping.items():
                    print(f"{label}: {idx}")

            except Exception as e:
                print(f"Bir hata oluştu: {e}")
import numpy as np
from collections import Counter
import logging

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from collections import Counter

class GeneticDataProcessor:
    def __init__(self, file_path, max_length=5000):
        self.file_path = file_path
        self.max_length = max_length
        self.sequences = []
        self.labels = []
        self.numeric_sequences = []
        self.padded_sequences = []  
        self.encoded_labels = []
        self.invalid_bases = []  # Geçersiz bazları tutacak liste

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
                elif line:  # Boş olmayan dizilim satırları
                    if current_label is not None:  # Etiketle ilişkilendirilmiş dizilim
                        self.labels.append(current_label)
                        self.sequences.append(line)
                    else:
                        print(f"Uyarı: Etiketsiz bir dizilim bulundu. Satır: {line}")
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")

            
    def validate_sequence(self, seq):
        """Geçersiz bazları kontrol et.""" 
        valid_bases = "ATCG"
        invalid_bases = [base for base in seq if base not in valid_bases]
        
        if invalid_bases:
            self.invalid_bases.append(''.join(invalid_bases))  # Geçersiz bazları kaydet
            print(f"Geçersiz genetik bazlar bulundu: {''.join(invalid_bases)} Dizilim: {seq}")
        else:
            print("Dizilimde geçersiz baz bulunmamaktadır.")

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
            invalid_indices = seq == -1  # -1 geçersiz baz olarak kabul ediliyor
            
            if invalid_indices.any():  # Eğer geçersiz baz varsa
                # En yaygın (mod) bazla doldurma
                valid_bases = seq[~invalid_indices]  # Geçerli bazları al
                if valid_bases.size > 0:  # Geçerli bazlar varsa
                    most_common_base = Counter(valid_bases).most_common(1)[0][0]
                    seq[invalid_indices] = most_common_base
            # Eğer geçersiz baz yoksa, herhangi bir şey yapmaya gerek yok
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
        print(f"label_mapping: {len(self.label_mapping)}.")
        print(f"encoded_labels: {len(self.encoded_labels)}.")

    def process_data(self, padding_value=0):
        logging.info("Veri işleme başlatıldı...")
        self.read_data()
        for seq in self.sequences:
            self.validate_sequence(seq)
        self.convert_to_numeric()
        self.pad_sequences(padding_value=padding_value)
        self.encode_labels()
        
        # Uzunluk kontrolü
        if len(self.sequences) != len(self.labels):
            raise ValueError(f"Etiket ve dizilim uzunlukları eşleşmiyor! Diziler: {len(self.sequences)}, Etiketler: {len(self.labels)}")

        logging.info("Veri işleme tamamlandı.")

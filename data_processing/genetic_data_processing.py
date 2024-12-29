class DataProcessing:
    @staticmethod
    def read_fasta(file_path):
        sequences = []
        try:
            with open(file_path, 'r') as file:
                sequence = []
                for line in file:
                    line = line.strip()
                    if line.startswith(">"):  # Yeni başlık satırı
                        if sequence:
                            sequences.append(sequence)  # Liste olarak kaydet
                        sequence = []  # Yeni diziyi başlat
                    else:
                        sequence.append(line)  # Diziye yeni veriler ekle
                if sequence:  # Son diziyi de ekle
                    sequences.append(sequence)
        except FileNotFoundError:
            print(f"Hata: Dosya bulunamadı: {file_path}")
        except Exception as e:
            print(f"Beklenmeyen bir hata oluştu: {e}")
        return sequences

from data_processing.genetic_data_processing import GeneticDataProcessor


def process_and_display_data():
    # Veri dosyasının yolu
    file_path = "data/SARS-genomes.fa"  # İşlemek istediğiniz dosyanın yolunu girin
    
    # İşlemci sınıfını oluştur
    processor = GeneticDataProcessor(file_path, max_length=1000)
    
    # Verileri işleyip görüntüle
    processor.process_and_display_data(padding_value=-1)
       

if __name__ == "__main__":
    process_and_display_data()
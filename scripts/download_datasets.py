import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEO_URLS = {
    "GSM3130435_egfp_unmod_1.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130435/suppl/GSM3130435_egfp_unmod_1.csv.gz",
    "GSE232927_processed_random_end_hek293t_N50_r1.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_random_end_hek293t_N50_r1.csv.gz",
    "GSE232927_processed_random_end_hek293t_N25_r1.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_random_end_hek293t_N25_r1.csv.gz",
    "GSE232927_processed_random_end_hek293t_N25_r2.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_random_end_hek293t_N25_r2.csv.gz",
    "GSE232927_processed_defined_end_hepg2_r1.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_defined_end_hepg2_r1.csv.gz",
    "GSE232927_processed_defined_end_tcell_r1.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_defined_end_tcell_r1.csv.gz",
    "GSE232927_processed_defined_end_tcell_r2.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_defined_end_tcell_r2.csv.gz"
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def download_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for filename, url in GEO_URLS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            logging.info(f"File {filename} already exists in data/. Skipping download.")
            continue
        
        logging.info(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            logging.info(f"Successfully downloaded {filename}.")
        except Exception as e:
            logging.error(f"Failed to download {filename}. Error: {e}")

if __name__ == "__main__":
    download_datasets()

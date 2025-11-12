import pandas as pd
import requests
from pathlib import Path
from alphagenome.data import gene_annotation
from alphagenome.data import transcript as transcript_utils

GTF_FEATHER_URL = ('https://storage.googleapis.com/alphagenome/reference/gencode/'
                   'hg38/gencode.v46.annotation.gtf.gz.feather')
GTF_LOCAL_PATH = Path("data/gencode.v46.annotation.gtf.gz.feather")


def download_gtf(url=GTF_FEATHER_URL, local_path=GTF_LOCAL_PATH):
    """
    Downloads the GTF feather file from the URL and saves it locally.
    """
    print(f"INFO: Downloading GTF file... (This is a one-time setup)")
    print(f"  From: {url}")
    print(f"  To:   {local_path}")
    
    # Ensure the 'data' directory exists
    local_path.parent.mkdir(exist_ok=True)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("INFO: Download complete!")
    except requests.exceptions.RequestException as e:
        raise IOError(f"Failed to download GTF file from {url}. Error: {e}")
    except Exception as e:
        raise IOError(f"An error occurred saving the GTF file to {local_path}. Error: {e}")

def load_gtf(local_path=GTF_LOCAL_PATH):
    """
    Loads and processes the GTF file.
    """
    if not local_path.exists():
        print(f"WARNING: Local GTF file not found at {local_path}.")
        try:
            download_gtf(local_path=local_path)
        except IOError as e:
            raise FileNotFoundError(f"GTF file not found and download failed. Original error: {e}")
    
    try:
        gtf = pd.read_feather(local_path)
    except Exception as e:
        raise Exception(f"Failed to read local feather file {local_path}. It may be corrupted. Error: {e}")

    gtf_transcripts = gene_annotation.filter_protein_coding(gtf)
    gtf_transcripts = gene_annotation.filter_to_longest_transcript(gtf_transcripts)
    transcript_extractor = transcript_utils.TranscriptExtractor(gtf_transcripts)

    return gtf, gtf_transcripts, transcript_extractor
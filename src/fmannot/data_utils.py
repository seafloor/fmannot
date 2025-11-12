from alphagenome.data import gene_annotation
from alphagenome.data import transcript as transcript_utils
import pandas as pd

def load_gtf():
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )

    # Set up transcript extractors using the information in the GTF file.
    gtf_transcripts = gene_annotation.filter_protein_coding(gtf)
    gtf_transcripts = gene_annotation.filter_to_longest_transcript(gtf_transcripts)
    transcript_extractor = transcript_utils.TranscriptExtractor(gtf_transcripts)

    return gtf, gtf_transcripts, transcript_extractor
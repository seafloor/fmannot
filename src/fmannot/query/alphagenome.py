import re
import pandas as pd
import zarr
from tqdm.notebook import tqdm
from pathlib import Path
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.data import genome
from alphagenome.data import track_data

OUT_VEP_LOCAL_PATH = Path("out/effect_predictions.feather")
OUT_GTP_LOCAL_PATH = Path("out/track_predictions.zarr")

def set_variant(row):
    chrom, pos, ref, alt = str(row['chr']), row['position_grch38'], row['ref'], row['alt']

    chrom = f'chr{chrom}' if 'chr' not in chrom else chrom

    variant = genome.Variant(
        chromosome=str(chrom),
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )

    return variant

def lookup_output_types(output_type):
    if isinstance(output_type, str):
        output_type = [output_type]
    
    outputs = {
        'atac': dna_client.OutputType.ATAC,
        'cage': dna_client.OutputType.CAGE,
        'dnase': dna_client.OutputType.DNASE,
        'rna_seq': dna_client.OutputType.RNA_SEQ,
        'chip_histone': dna_client.OutputType.CHIP_HISTONE,
        'chip_tf': dna_client.OutputType.CHIP_TF,
        'splice_sites': dna_client.OutputType.SPLICE_SITES,
        'splice_site_usage': dna_client.OutputType.SPLICE_SITE_USAGE,
        'splice_junctions': dna_client.OutputType.SPLICE_JUNCTIONS,
        'contact_maps': dna_client.OutputType.CONTACT_MAPS,
        'procap': dna_client.OutputType.PROCAP 
    }

    return [outputs.get(s, None) for s in output_type]

def lookup_ontologies():
    pass

def get_genome_track_prediction(row, dna_model, output_type=['rna_seq', 'splice_sites'], ontologies=['EFO:0001200', 'CL:0000084']):
    variant = set_variant(row)

    interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

    output = dna_model.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs=lookup_output_types(output_type),
        ontology_terms=ontologies
    )

    return output

def get_variant_effect_prediction(row, dna_model):
    variant = set_variant(row)

    interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

    variant_scores = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values()),
    )

    df_scores = variant_scorers.tidy_scores(variant_scores)

    return df_scores

def reformat_vep(predicted_effects, snps, annot):
    extract_cols = [
        'variant_id',
        'scored_interval',
        'gene_id',
        'gene_name',
        'gene_type',
        'gene_strand',
        'output_type',
        'variant_scorer',
        'track_name',
        'Assay title',
        'ontology_curie',
        'biosample_name',
        'biosample_type',
        'raw_score',
        'quantile_score'
    ]
    
    vep_out = (predicted_effects
        .loc[:, extract_cols]
        .rename(columns={'Assay title': 'assay_title'})
        .assign(
            variant_id = lambda x: x['variant_id'].astype(str),
            scored_interval = lambda x: x['scored_interval'].astype(str),
            gene_id = lambda x: x['gene_id'].astype(str),
            gene_name = lambda x: x['gene_name'].astype(str),
            gene_type = lambda x: x['gene_type'].astype(str),
            gene_strand = lambda x: x['gene_strand'].astype(str),
            variant_scorer = lambda x: x['variant_scorer'].astype(str),
            track_name = lambda x: x['track_name'].astype(str),
            assay_title = lambda x: x['assay_title'].astype(str),
            ontology_curie = lambda x: x['ontology_curie'].astype(str),
            biosample_name = lambda x: x['biosample_name'].astype(str),
            biosample_type = lambda x: x['biosample_type'].astype(str),
        )
        .merge(
            (snps
                .assign(variant_id='chr' + snps['chr'].astype(str) + ':' + snps['position_grch38'].astype(str) + ':' + snps['ref'] + '>' + snps['alt'])
            ),
            left_on='variant_id', right_on='variant_id',
            how='inner', validate='many_to_one'
        )
        .merge(
            annot,
            left_on='biosample_name', right_on='biosample_name',
            how='left', validate='many_to_many'
        )
    )

    return vep_out

def save_all_effect_predictions(predicted_effects, snps, annot, store_path=OUT_VEP_LOCAL_PATH):
    # ensure the 'out' directory exists
    store_path.parent.mkdir(exist_ok=True)

    vep_out = reformat_vep(predicted_effects, snps, annot)

    vep_out.to_feather(store_path)
    
def interval_to_str(interval_obj):
    d = interval_obj.to_interval_dict()
    
    return f"{d['chromosome']}:{d['start']}-{d['end']}:{d['strand']}"

def parse_interval_str(interval_str):
    interval_str = re.split(':|-', interval_str)

    interval_obj = genome.Interval(
        chromosome = interval_str[0],
        start = interval_str[1],
        end = interval_str[2],
        strand = interval_str[3]
    )

    return interval_obj

def save_track_data(zarr_group: zarr.Group, tdata: track_data.TrackData):
    """
    Saves a TrackData object (e.g., RNA_SEQ, ATAC) into a Zarr group
    by saving its raw components.
    """
    if tdata is None:
        print(f"  Warning: No TrackData provided for {zarr_group.name}")
        return
    
    data_array = tdata.values
    
    # chunk along the sequence, not the tracks
    data_chunks = (1024, data_array.shape[1]) 
    
    # Shape and dtype are inferred from data_array.
    zarr_group.create_array(
        'values', 
        data=data_array,
        chunks=data_chunks
    )
    
    zarr_group.attrs['metadata_json'] = tdata.metadata.to_json(orient='records')
    zarr_group.attrs['resolution'] = tdata.resolution
    
    zarr_group.attrs['interval_chromosome'] = tdata.interval.chromosome
    zarr_group.attrs['interval_start'] = tdata.interval.start
    zarr_group.attrs['interval_end'] = tdata.interval.end
    zarr_group.attrs['interval_strand'] = tdata.interval.strand

def save_generic_data(zarr_group: zarr.Group, data_obj):
    """
    Saves non-TrackData objects (e.g., SpliceJunctions DataFrames)
    as JSON attributes.
    """
    if data_obj is None:
        print(f"  Warning: No data provided for {zarr_group.name}")
        return

    if hasattr(data_obj, 'to_json'):
        zarr_group.attrs['data_json'] = data_obj.to_json(orient='records')
    else:
        zarr_group.attrs['data_raw'] = str(data_obj)
        
    if hasattr(data_obj, 'interval'):
        zarr_group.attrs['interval_chromosome'] = data_obj.interval.chromosome
        zarr_group.attrs['interval_start'] = data_obj.interval.start
        zarr_group.attrs['interval_end'] = data_obj.interval.end
        zarr_group.attrs['interval_strand'] = data_obj.interval.strand

def save_all_track_predictions(predicted_tracks, snps, store_path = OUT_GTP_LOCAL_PATH):
    ALL_OUTPUT_NAMES = [
        'atac', 'cage', 'dnase', 'rna_seq', 'chip_histone', 'chip_tf',
        'splice_sites', 'splice_site_usage', 'splice_junctions',
        'contact_maps', 'procap'
    ]

    # ensure /out exists
    store_path.parent.mkdir(exist_ok=True)

    print(f"Opening Zarr v3 store '{store_path}'...")
    root_group = zarr.open(store_path, mode='a')

    with tqdm(predicted_tracks.items(), desc="Querying Alpha Genome API", total=len(predicted_tracks)) as pbar:
        for index, variant_output in pbar:
            try:
                info_row = snps.loc[index]
                variant_rsid = info_row['variant']
                
                variant_id_str = (
                    f"chr{info_row['chr']}_"
                    f"{info_row['position_grch38']}_"
                    f"{info_row['ref']}_"
                    f"{info_row['alt']}"
                )
            except KeyError:
                print(f"Error: No variant info found in 'snps' for index {index}. Skipping.")
                continue
            
            pbar.set_description(f"Processing variant: {variant_rsid} ({variant_id_str})")
        
            variant_group = root_group.require_group(variant_rsid)
            variant_group.attrs['variant_id_string'] = variant_id_str
        
            for output_name in ALL_OUTPUT_NAMES:
                
                ref_data = getattr(variant_output.reference, output_name, None)
                alt_data = getattr(variant_output.alternate, output_name, None)
        
                if ref_data and alt_data:
                    output_group = variant_group.require_group(output_name.upper()) 
                    ref_group = output_group.require_group('REFERENCE')
                    alt_group = output_group.require_group('ALTERNATE')
                    
                    if isinstance(ref_data, track_data.TrackData):
                        save_track_data(ref_group, ref_data)
                        save_track_data(alt_group, alt_data)
                    else:
                        save_generic_data(ref_group, ref_data)
                        save_generic_data(alt_group, alt_data)
        
        print("\n--- Zarr saving complete ---")

def load_track_data_from_zarr(zarr_group: zarr.Group) -> track_data.TrackData:
    """
    Re-hydrates a lazy-loaded TrackData object from a Zarr group.
    """
    if 'values' not in zarr_group:
        raise(f"Error: No 'values' array in Zarr group: {zarr_group.name}")

    values_array = zarr_group['values']
    
    attrs = zarr_group.attrs
    metadata = pd.read_json(attrs['metadata_json'], orient='records')
    resolution = attrs['resolution']
    
    interval = genome.Interval(
        chromosome=attrs['interval_chromosome'],
        start=attrs['interval_start'],
        end=attrs['interval_end'],
        strand=attrs['interval_strand']
    )
    
    reconstructed_tdata = track_data.TrackData(
        values=values_array,
        metadata=metadata,
        resolution=resolution,
        interval=interval
    )
    
    return reconstructed_tdata

def read_tracks_for_variant(rsid, output_type = 'rna_seq', store_path = OUT_GTP_LOCAL_PATH):
    root = zarr.open(store_path, mode='r')
    
    selected_rsid = rsid
    selected_output = output_type.upper()
    
    try:
        variant_group = root[selected_rsid]
    
        variant_id_str = variant_group.attrs['variant_id_string']
        
        print(f"Successfully found string for {selected_rsid}:")
        print(variant_id_str)
    
    except KeyError:
        print(f"Error: Could not find rsid '{selected_rsid}' in the Zarr store.")

    variant = set_variant(var_str_to_dict(variant_id_str))
    
    # naviate to specific groups and get ref and alt seq track data
    try:
        ref_group = root[f'{selected_rsid}/{selected_output}/REFERENCE']
        alt_group = root[f'{selected_rsid}/{selected_output}/ALTERNATE']
    
        # load by passing the group objects, not the path string.
        ref_track_data = load_track_data_from_zarr(ref_group)
        alt_track_data = load_track_data_from_zarr(alt_group)
    
        print(f"Successfully loaded: {ref_track_data}")
        print(f"Values array shape: {ref_track_data.values.shape}")
        print(f"Metadata: \n{ref_track_data.metadata.head()}")
    
    except KeyError:
        print(f"Error: Could not find data for {selected_rsid}/{selected_output}")
        print(f"Available variants are: {list(root.keys())}")

    return variant, ref_track_data, alt_track_data

def var_str_to_dict(var_string):
    names = ['chr', 'position_grch38', 'ref', 'alt']
    var_string = var_string.split('_')

    return dict(zip(names, var_string))

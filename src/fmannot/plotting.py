from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt

def plot_track(ref_track, alt_track, transcript_extractor, variant=None):
    longest_transcripts = transcript_extractor.extract(ref_track.interval)
    annot = [plot_components.VariantAnnotation([variant], alpha=0.8)] if variant is not None else variant
    
    fig = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': ref_track,
                    'ALT': alt_track,
                },
                colors={'REF': 'dimgrey', 'ALT': 'red'},
            ),
        ],
        interval=ref_track.interval.resize(2**15),
        # Annotate the location of the variant as a vertical line.
        annotations=annot,
    )

    return fig
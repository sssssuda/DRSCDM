import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from collections import namedtuple
from pyannote.core import Segment, SlidingWindow


# Segment = namedtuple("Segment", ["start", "end"])


def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def extract_embeddings_wav(embedder, 
                           audio,
                           segments=(),
                           device="cuda",
                           batch_size=1, 
                           sample_rate=16000, 
                           verbose=False):
    
    try:
        embedder.eval()
    except:
        pass
    
    if torch.is_tensor(audio):
        waveform = audio
    else:
        waveform = torch.tensor(audio).view(1, -1)
    
    wav_length = waveform.shape[-1]

    if len(segments) == 0:
        segments = [Segment(0., wav_length/sample_rate)]
        
    # make sure that segments boundaries are within the actual waveform
    for i in range(len(segments)):
        segments[i] = Segment(segments[i].start, min(segments[i].end, wav_length/sample_rate))

    # find unique segment lengths and group the segments with the same length for batch processing
    if batch_size > 1:
        lengths = [int(min(sample_rate * seg.end, wav_length) - sample_rate * seg.start) for seg in segments]
        lengths_unique, idx_inverse = np.unique(lengths, return_inverse=True)
        segments_sets = []
        for k in range(len(lengths_unique)):
            idx = np.nonzero(idx_inverse == k)[0]
            segments_set = []
            for i in idx:
                segments_set += [(segments[i], i)]
            segments_sets += [segments_set]
    else:
        segments_set = [(seg, i) for (i, seg) in enumerate(segments)]
        segments_sets = [segments_set]

    embeddings = []
    indices = []
    for segments_set in segments_sets:
        if verbose:
            t = tqdm(segments_set)
        else:
            t = segments_set
        wavs_batch = []
        for i, (segment, idx) in enumerate(t):
            indices += [idx]
            start = segment.start
            end = segment.end
            duration = end - start

            sample_start = int(sample_rate * start)
            sample_end = int(sample_rate * end)
            num_samples = int(sample_rate * duration) # variable duration
            #num_samples = int(sample_rate * win_size)

            wav = waveform[..., sample_start: min(sample_start + num_samples, wav_length)]
            #wav = waveform[..., sample_start: min(sample_end, wav_length)]
            wav = torch.as_tensor(wav).view(1, -1)
            wavs_batch += [wav]
            
            if (i+1) % batch_size == 0 or (i+1) == len(segments_set):
        
                # crop waves in the batch to the minimum length
                lengths_batch = [wav.shape[-1] for wav in wavs_batch]
                n_min = min(lengths_batch)
                n_max = max(lengths_batch)
                if n_min != n_max:
                    #print(f"Warning: min and max lengths in the batch are not equal: {n_min} != {n_max}, cropping.")
                    wavs_batch =  [wav[..., :n_min] for wav in wavs_batch]
                
                with torch.no_grad():
                    emb_batch = embedder(torch.cat(wavs_batch).to(device)).cpu()

                embeddings += [emb_batch]
                wavs_batch = []
                
    embeddings = torch.cat(embeddings).cpu().numpy()
    indices = np.array(indices)
    embeddings = embeddings[invert_permutation(indices)]
    return embeddings


def split_segments(vad, win_size=2.0, step_size=2.0):
    segments_speech = []
    for segment in vad:
        start = segment.start
        end = segment.end
        duration = end - start
        if duration < 0. * win_size:
            pass
            # ignore very short segments
        elif duration < win_size + step_size:
            segment_short = segment
            segments_speech += [segment_short]
        else:
            sliding_window = SlidingWindow(win_size, step_size)
            for segment_short in sliding_window(segment):
                segments_speech += [segment_short]
            segments_speech[-1] = Segment(segment_short.start, segment.end)
    return segments_speech


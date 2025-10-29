import torch
import torchaudio
from typing import List
from torchaudio.pipelines import MMS_FA as bundle
import re
import csv
import os
import json
from tqdm import tqdm
from bisect import bisect_right, bisect_left
import argparse
import yaml

# Script based on https://docs.pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    """
    Run MMS forced alignment and return (emission, token_spans).
    """
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    
    model.eval()
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans
    
def word_spans(waveform, spans, num_frames, sample_rate):
    """
    Convert token spans (frame idx) to (start_ms, end_ms) using the waveform length.
    """
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    return (x0 / sample_rate * 1000, x1 / sample_rate * 1000) 

def normalize_uroman(text):
    """
    Simple, robust normalization matching uroman-ish behavior used in examples.
    """
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def align_tokens_to_words(
    time_stamps,
    tokens,
    token_dur_ms=40,
    use_centers=True,
    prefer_left_on_tie=False,
    enforce_min_one=False,
    eps=1e-6,
):
    """
    Assign each fixed-duration token (frame) to a word span using decision
    boundaries halfway between consecutive word spans.

    - If `use_centers=True`, boundaries are computed between word centers.
    - If `enforce_min_one=True`, ensures at least one token per word by 'borrowing'
      from nearest neighbors when possible.
    """
    if not time_stamps:
        raise ValueError("time_stamps is empty.")
    n = len(time_stamps)

    # keep original indices and sort by start time (temporal order)
    ts_with_idx = [(s, e, i) for i, (s, e) in enumerate(time_stamps)]
    ts_sorted = sorted(ts_with_idx, key=lambda x: x[0]) 
    order = [orig for (_, _, orig) in ts_sorted]       

    # build decision boundaries (centers or edges)
    if use_centers:
        centers = [(s + e) / 2.0 for (s, e, _) in ts_sorted]
        for i in range(1, n):
            if centers[i] <= centers[i-1]:
                centers[i] = centers[i-1] + eps
        boundaries = [(centers[i] + centers[i+1]) / 2.0 for i in range(n - 1)]
    else:
        boundaries = [(ts_sorted[i][1] + ts_sorted[i+1][0]) / 2.0 for i in range(n - 1)]

    bsearch = bisect_left if prefer_left_on_tie else bisect_right
    aligned_sorted = [[] for _ in range(n)]

   # initial token assignment by token mid-time
    for i, tok in enumerate(tokens):
        tok_mid = (i + 0.5) * token_dur_ms
        j = bsearch(boundaries, tok_mid)
        if j < 0: j = 0
        elif j >= n: j = n - 1
        aligned_sorted[j].append(tok)

    # guarantee at least one token per word if requested
    if enforce_min_one and len(tokens) >= n:
        empties = [j for j, lst in enumerate(aligned_sorted) if len(lst) == 0]

        def find_donor(j):
            # search left
            left = None
            d = 1
            while j - d >= 0:
                if len(aligned_sorted[j - d]) > 1:
                    left = (j - d, d)
                    break
                d += 1
            # right
            right = None
            d = 1
            while j + d < n:
                if len(aligned_sorted[j + d]) > 1:
                    right = (j + d, d)
                    break
                d += 1
            if left and right:
                if left[1] < right[1]:
                    return left[0]
                if right[1] < left[1]:
                    return right[0]
                # tie-breaker: give from the side with more tokens
                return left[0] if len(aligned_sorted[left[0]]) >= len(aligned_sorted[right[0]]) else right[0]
            return left[0] if left else (right[0] if right else None)

        for j in empties:
            donor = find_donor(j)
            if donor is None:
                continue
            if donor < j:
                tok = aligned_sorted[donor].pop()
                aligned_sorted[j].insert(0, tok)
            else:
                tok = aligned_sorted[donor].pop(0)
                aligned_sorted[j].append(tok)

    # restore original word order
    aligned = [None] * n
    for pos_temporal, orig_i in enumerate(order):
        aligned[orig_i] = aligned_sorted[pos_temporal]

    return aligned
                    
def parse_args():
    p = argparse.ArgumentParser(description="Forced alignment with MMS (torchaudio) using YAML config")
    p.add_argument("--config", required=True, help="Path to YAML config")
    return p.parse_args()

def load_config(cli_args) -> dict:
    """
    Load and validate the YAML configuration file.
    """
    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    required = ["audio", "huberts", "output_path", "lang_pairs"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required keys in YAML: {missing}")
    
    # validate subkeys
    audio_keys = ["dir", "tsv"]
    missing_audio = [k for k in audio_keys if k not in config.get("audio", {})]
    if missing_audio:
        raise ValueError(f"Missing required audio subkeys in YAML: {missing_audio}")
    
    return config


def main(config):
    audio_dir = config.get("audio").get("dir")
    tsv_path = config.get("audio").get("tsv")
    output_path = config.get("output_path")
    tokens_path = config.get("huberts").get("tokens_path")
    token_dur_ms = config.get("huberts", {}).get("token_dur_ms", 40)
    if token_dur_ms <= 0:
        raise ValueError(f"Invalid token_dur_ms: {token_dur_ms}. Must be positive.")
    
    lang_pairs = config.get("lang_pairs")
    
    for lang_pair in lang_pairs:
        print(f"Processing {lang_pair}")
        
        with open(tokens_path.format(lang_pair=lang_pair), "r", encoding="utf-8") as f:
            tokens_txt = f.readlines()
            
        with open(tsv_path.format(lang_pair=lang_pair), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)  # skip header
            tsv = list(reader)

        if len(tokens_txt) != len(tsv):
            raise ValueError(f"Mismatch in lengths: tokens_txt ({len(tokens_txt)}) and tsv ({len(tsv)})")

        output_file = os.path.join(output_path, f"{lang_pair}.aligned_huberts.txt")
        with open(output_file, "w", encoding="utf-8") as out:
            for txt_row, tsv_row in tqdm(
                zip(tokens_txt, tsv),
                total=len(tokens_txt),
                desc=f"{lang_pair} - Processing {audio_wav}",
                dynamic_ncols=True,
            ):
                # hubert tokens and corresponding audio file + transcription
                token_list = [t for t in txt_row.strip().split(" ") if t]
                
                audio_wav = tsv_row[1]       
                transcription = tsv_row[2] 
                transc_norm = normalize_uroman(transcription)
                transcript = transc_norm.split()
                
                try:
                    waveform, sample_rate = torchaudio.load(os.path.join(audio_dir, audio_wav))
                except FileNotFoundError:
                    print(f"Audio file not found: {audio_wav}")
                    continue
                except Exception as e:
                    print(f"Error loading audio file {audio_wav}: {e}")
                    continue
                
                waveform = waveform.to(device)
                emission, token_spans = compute_alignments(waveform, transcript)
                num_frames = emission.size(1)
                time_stamps = []
                for i, spans in enumerate(token_spans):
                    start_ms, end_ms = word_spans(waveform, spans, num_frames, bundle.sample_rate)
                    time_stamps.append((start_ms, end_ms))
                
                aligned_tokens = align_tokens_to_words(
                    time_stamps,
                    token_list,
                    token_dur_ms=token_dur_ms,         
                    use_centers=True, 
                    prefer_left_on_tie=False, 
                    enforce_min_one=True 
                )

                mapping = {
                    (i, transcript[i]): aligned_tokens[i]
                    for i in range(len(transcript))
                }
                try:
                    out.write(json.dumps(mapping) + "\n")
                except Exception as e:
                    print(f"[ERROR] Failed to write to output file {output_file}: {e}")

            print(f"Saved at: {output_file}")

if __name__ == "__main__":
    cli_args = parse_args()
    config = load_config(cli_args)
    main(config)
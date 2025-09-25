import torch
import torchaudio
from typing import List
import IPython
import matplotlib.pyplot as plt
from torchaudio.pipelines import MMS_FA as bundle
import re
import csv
import os
import json
from tqdm import tqdm
from bisect import bisect_right, bisect_left

# Script taken from https://docs.pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()
    
def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    # print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return (x0 / sample_rate * 1000, x1 / sample_rate * 1000) #IPython.display.Audio(segment.numpy(), rate=sample_rate)

def normalize_uroman(text):
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
    if not time_stamps:
        raise ValueError("time_stamps está vacío.")
    n = len(time_stamps)

    # Guardamos índice original
    ts_with_idx = [(s, e, i) for i, (s, e) in enumerate(time_stamps)]
    ts_sorted = sorted(ts_with_idx, key=lambda x: x[0])  # por inicio
    order = [orig for (_, _, orig) in ts_sorted]         # orig_i en orden temporal

    # Fronteras en orden temporal
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

    # Asignación inicial (en orden temporal)
    for i, tok in enumerate(tokens):
        tok_mid = (i + 0.5) * token_dur_ms
        j = bsearch(boundaries, tok_mid)
        if j < 0: j = 0
        elif j >= n: j = n - 1
        aligned_sorted[j].append(tok)

    # Post-proceso mínimo 1 (en orden temporal)
    if enforce_min_one and len(tokens) >= n:
        empties = [j for j, lst in enumerate(aligned_sorted) if len(lst) == 0]

        def find_donor(j):
            # izquierda
            left = None
            d = 1
            while j - d >= 0:
                if len(aligned_sorted[j - d]) > 1:
                    left = (j - d, d)
                    break
                d += 1
            # derecha
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

    # Reordenar de vuelta al orden original
    aligned = [None] * n
    for pos_temporal, orig_i in enumerate(order):
        aligned[orig_i] = aligned_sorted[pos_temporal]

    return aligned


# def align_tokens_to_words(time_stamps, tokens, token_dur_ms=40):
#     """
#     Alinea cada token (frame) a una palabra usando fronteras de decisión
#     situadas a mitad entre fin_de_palabra[i] y inicio_de_palabra[i+1].
#     No se pierde ningún token: cada token va exactamente a 1 palabra.

#     Args:
#         time_stamps: lista de (t1, t2) en ms para cada palabra (no solapados).
#         tokens: lista de tokens (uno por frame, duración fija).
#         token_dur_ms: duración de cada token en ms.

#     Returns:
#         aligned: lista de sublistas; aligned[i] = tokens asignados a palabra i.
#     """
#     if not time_stamps:
#         raise ValueError("time_stamps está vacío; no hay palabras a las que alinear.")

#     # Asegura orden por inicio
#     time_stamps = sorted(time_stamps, key=lambda x: x[0])
#     n = len(time_stamps)

#     # Fronteras entre palabras: midpoint entre (end_i) y (start_{i+1})
#     boundaries = []
#     for i in range(n - 1):
#         end_i = time_stamps[i][1]
#         start_next = time_stamps[i + 1][0]
#         boundaries.append((end_i + start_next) / 2.0)

#     aligned = [[] for _ in range(n)]

#     # Asigna por el centro del token
#     for i, tok in enumerate(tokens):
#         tok_mid = (i + 0.5) * token_dur_ms
#         # índice de palabra = nº de fronteras estrictamente menores que tok_mid
#         idx = bisect_right(boundaries, tok_mid)
#         # idx ∈ [0, n-1] garantizado
#         aligned[idx].append(tok)

#     return aligned

audio_dir = "/gpfs/projects/bsc88/speech/data/raw_data/fleurs/en_us/audio/test/"
tokens_path = "/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/results/s2_TA-7b-ins_mhubert_st-jacobo_v4/s2tt-cot/fleurs-test-aligned_huberts/{lang_pair}.audio_target.txt"
tsv_path = "/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/data_fleurs/tsv/{lang_pair}.test.tsv"
#output_path = "/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/results/s2tt-cot/fleurs-iber"
output_path="/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/results/s2_TA-7b-ins_mhubert_st-jacobo_v4/s2tt-cot/fleurs-test-aligned_huberts"

def main():
    lang_pairs = ["en_es", "en_ca", "en_pt", "en_de", "en_fr", "en_it"]
    for lang_pair in lang_pairs:
        print(f"Processing {lang_pair}")
        with open(tokens_path.format(lang_pair=lang_pair), "r", encoding="utf-8") as f:
            tokens_txt = f.readlines()
        with open(tsv_path.format(lang_pair=lang_pair), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)  # <- evita que 'src_audio' sea tu primera "ruta"
            tsv = list(reader)

        output_file = os.path.join(output_path, f"{lang_pair}.aligned_huberts.txt")
        with open(output_file, "w", encoding="utf-8") as out:
            for txt_row, tsv_row in tqdm(
                zip(tokens_txt, tsv),
                total=len(tokens_txt),
                desc=f"{lang_pair}",
                dynamic_ncols=True,
            ):
                token_list = [t for t in txt_row.strip().split(" ") if t]
                audio_wav = tsv_row[1]
                transcription = tsv_row[2]
                
                transc_norm = normalize_uroman(transcription)

                waveform, sample_rate = torchaudio.load(os.path.join(audio_dir, audio_wav))
                duration_sec = waveform.shape[1] / sample_rate

                duration_ms = duration_sec * 1000
                token_count = len(token_list)

                assert sample_rate == bundle.sample_rate
                transcript = transc_norm.split()
                tokens = tokenizer(transcript)
                emission, token_spans = compute_alignments(waveform, transcript)
                num_frames = emission.size(1)

                time_stamps = []
                for i, spans in enumerate(token_spans):
                    start_ms, end_ms = preview_word(waveform, spans, num_frames, transcript[i])
                    time_stamps.append((start_ms, end_ms))
                    # imprime como: perry: 0.04 - 0.24 sec
                    start_s = start_ms / 1000.0
                    end_s   = end_ms   / 1000.0
                    print(f"{transcript[i]}: {start_s:.2f} - {end_s:.2f} sec")

                #aligned_tokens = align_tokens_to_words(time_stamps, token_list)
                
                aligned_tokens = align_tokens_to_words(
                    time_stamps,
                    token_list,
                    token_dur_ms=40,          
                    use_centers=True, # fronteras por centros
                    prefer_left_on_tie=False, 
                    enforce_min_one=True #la clave para evitar 0 tokens!
                )

                mapping = {(i,transcript[i]): aligned_tokens[i] for i in range(len(transcript))}

                # aquí ya no usamos json.dumps, simplemente str(mapping)
                out.write(str(mapping) + "\n")

            print(f"Saved at: {output_file}")
                    

if __name__ == "__main__":
    main()
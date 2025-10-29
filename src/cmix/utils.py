from typing import Dict, Iterable, Tuple, List, Optional, Any
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from captum.attr import (
    LayerIntegratedGradients,
    DeepLiftShap,
    GradientShap,
    FeatureAblation,
    KernelShap,
    Lime,
    ShapleyValueSampling,
    LLMAttribution,
    LLMGradientAttribution,
    LLMAttributionResult,
    TextTokenInput,
    TextTemplateInput,
)

import matplotlib
matplotlib.use("Agg")  

def load_attr_method(method: str, model, tokenizer):
    print(f"Attributing with {method}")
    if method == "layer-integrated-gradients":
        attr_method = LayerIntegratedGradients(model, model.model.embed_tokens)
        return LLMGradientAttribution(attr_method, tokenizer)
    elif method == "deeplift-shap":
        attr_method = DeepLiftShap(model, model.model.embed_tokens)
        return LLMAttribution(attr_method, tokenizer)
    elif method == "gradient-shap":
        attr_method = GradientShap(model, model.model.embed_tokens)
        return LLMGradientAttribution(attr_method, tokenizer)
    elif method == "feature-ablation":
        attr_method = FeatureAblation(model)
        return LLMAttribution(attr_method, tokenizer)
    elif method == "kernel-shap":
        attr_method = KernelShap(model)
        return LLMAttribution(attr_method, tokenizer)
    elif method == "lime":
        attr_method = Lime(model)
        return LLMAttribution(attr_method, tokenizer)
    elif method == "shapley-values":
        attr_method = ShapleyValueSampling(model) 
        return LLMAttribution(attr_method, tokenizer)        
    else:
        raise ValueError("Not implemented")
    
# hard coded method families
GRADIENT_BASED = {
    "layer-integrated-gradients",
    "gradient-shap",
    "deeplift-shap",
}
PERTURBATION_BASED = {
    "feature-ablation",
    "kernel-shap",
    "lime",
    "shapley-values",
}

def is_gradient_method(method: str) -> bool:
    return method in GRADIENT_BASED

def _build_captum_input(
    *, attr_method: str, template: str, values: dict, mask: dict, skip_tokens: list[str]
):
    if is_gradient_method(attr_method):
        # token-level input: pass skip_tokens here
        return TextTokenInput(
            template,
            values=values,
            mask=mask,
            skip_tokens=skip_tokens,
        )
    else:
        # template-level input: NO skip_tokens arg here
        return TextTemplateInput(
            template,
            values=values,
            mask=mask,
        )

def call_attribute(
    *, 
    attr_method: str, 
    template: str, 
    values: dict, 
    mask: dict, 
    llm_attr: any, 
    generated_text: str,
    skip_tokens: list[str], 
    attribution_kwargs: dict = None, 
):
    captum_inp = _build_captum_input(
        attr_method=attr_method,
        template=template,
        values=values,
        mask=mask,
        skip_tokens=skip_tokens,
    )
    if is_gradient_method(attr_method):
        # gradient-style: do NOT pass skip_tokens here
        return llm_attr.attribute(
            captum_inp,
            target=generated_text,
            **attribution_kwargs,
        )
    else:
        # perturbation-style: pass skip_tokens here
        return llm_attr.attribute(
            captum_inp,
            target=generated_text,
            skip_tokens=skip_tokens,
            **attribution_kwargs,
        )

    
def _display_word_from_key(key: str) -> str:
    """
    Normalizes keys with index to display them without the index.
    Supports formats like "12:word" or "(12,word)"; otherwise returns the key as is.
    """
    # Format (idx,word)
    if key.startswith("(") and key.endswith(")") and "," in key:
        inner = key[1:-1]
        parts = inner.split(",", 1)
        if len(parts) == 2:
            return parts[1].strip()
    # Format idx:word
    if ":" in key:
        left, right = key.split(":", 1)
        if left.isdigit():
            return right.strip()
    return key

def build_audio_spans(audio_dict: Dict[str, Iterable]) -> Tuple[List[str], List[int], List[int]]:
    """
    Given a dictionary mapping words to lists of tokens, constructs word spans using the hubert tokens.
    
    Returns:
      - audio_words: ['/' + word + '/'] in order
      - audio_start_idx: start indices of words (0-based)
      - audio_end_idx: end indices of words (0-based, inclusive)
    
    Example:
        input: {(0, 'more'): ['\U000f0108', '\U000f0108'], (1, 'than'): ['\U000f0119', '\U000f00d5']}
        output: (['/more/', '/than/'], [0, 1], [2, 3])
    """
    audio_words: List[str] = []
    audio_start_idx: List[int] = []
    audio_end_idx: List[int] = []

    offset = 0
    for key, vals in audio_dict.items():
        n = len(vals)
        disp_key = _display_word_from_key(str(key))
        audio_words.append(f"/{disp_key}/")
        audio_start_idx.append(offset)
        offset += n-1
        audio_end_idx.append(offset)
        offset += 1

    return audio_words, audio_start_idx, audio_end_idx

def group_scores(attr_res, audio_len, transc_len, tsv_path: Optional[str] = None, mode: str = "l1"):
    """
    Calcula proporciones de atribución agrupadas en bloques:
      - audio
      - transcript
      - translation
      - transcribe_in_token
      - translate_to_token

    Args:
        attr_res: objeto con .token_attr (lista de tensores [T_in] por cada token generado)
        audio_len: número de columnas de audio
        transc_len: número de columnas de transcript
        tsv_path: opcional, para logging/guardar
        mode: "softmax" (default) o "l1"
    """

    # apilar en matriz [T_out, T_in]
    new_token_attr = attr_res.token_attr   # [T_out, T_in]
    new_token_attr = new_token_attr[:, :]              # quitar 2 filas iniciales si corresponde

    if mode == "softmax":
        # softmax fila a fila
        norm_attr_res = torch.softmax(new_token_attr, dim=1)
    elif mode == "l1":
        # normalización L1 fila a fila con valores absolutos
        norm_attr_res = new_token_attr.abs()
        norm_attr_res = norm_attr_res / (norm_attr_res.sum(dim=1, keepdim=True).clamp_min(1e-12))
    else:
        raise ValueError("mode debe ser 'softmax' o 'l1'")

    # cortes
    audio_attr     = norm_attr_res[:, :audio_len].sum(dim=1) 
    transc_attr    = norm_attr_res[:, audio_len+1:audio_len+1+transc_len].sum(dim=1)     # transcript después del audio + "english"
    #transl_attr    = norm_attr_res[:, audio_len+2+transc_len:].sum(dim=1)                # después de transcript + "catalan"
    transc_in_attr = norm_attr_res[:, audio_len]                             # token "TRANSCRIBE_IN"
    transl_to_attr = norm_attr_res[:, audio_len+1+transc_len]                 # token "TRANSLATE_TO"

    # medias por bloque
    scores = {
        "audio": audio_attr.mean().item(),
        "transcript": transc_attr.mean().item(),
        #"translation": transl_attr.mean().item(),
        "transcribe_in_token": transc_in_attr.mean().item(),
        "translate_to_token": transl_to_attr.mean().item(),
    }

    print(f"[{mode.upper()}] AUDIO: {scores['audio']:.4f}, TRANSCRIPT: {scores['transcript']:.4f}, "
          f"Transcribe in _: {scores['transcribe_in_token']:.4f}, "
          f"Translate to _: {scores['translate_to_token']:.4f}")

    # crear si no existe, anexar si existe)
    try:
        if tsv_path is None:
            logs_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            tsv_path = os.path.join(logs_dir, "group_scores.tsv")
        else:
            os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

        write_header = not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0
        with open(tsv_path, mode="a", encoding="utf-8") as f:
            if write_header:
                f.write("AUDIO\tTRANSCRIPT\tTRANSCRIBE_IN\tTRANSLATE_TO\n")
            f.write(
                f"{scores['audio']:.6f}\t{scores['transcript']:.6f}\t{scores['transcribe_in_token']:.6f}\t{scores['translate_to_token']:.6f}\n"
            )
    except Exception as e:
        print(f"[WARN] No se pudo escribir TSV: {type(e).__name__}: {e}")

    new_attr_res = LLMAttributionResult(
        seq_attr=attr_res.seq_attr,
        token_attr=norm_attr_res,
        input_tokens=attr_res.input_tokens,
        output_tokens=attr_res.output_tokens
    )
    
    return new_attr_res   

def _collapse_input_span(
    attr_res: LLMAttributionResult,
    start: int,
    end: int,
    new_name: str,
    reduce: str = "mean",   # "sum" o "mean"
) -> LLMAttributionResult:
    if end < start:
        raise ValueError("end debe ser >= start")

    t = attr_res.token_attr  # esperado: (n_output_tokens, n_input_tokens)

    if isinstance(t, torch.Tensor):
        mid = t[:, start:end+1]
        agg = mid.sum(dim=1, keepdim=True) if reduce == "sum" else mid.mean(dim=1, keepdim=True)
        new_t = torch.cat([t[:, :start], agg, t[:, end+1:]], dim=1)

    elif isinstance(t, np.ndarray):
        mid = t[:, start:end+1]
        agg = mid.sum(axis=1, keepdims=True) if reduce == "sum" else mid.mean(axis=1, keepdims=True)
        new_t = np.concatenate([t[:, :start], agg, t[:, end+1:]], axis=1)
    else:
        raise TypeError("token_attr debe ser torch.Tensor o np.ndarray")

    new_input_tokens = (
        attr_res.input_tokens[:start] + [new_name] + attr_res.input_tokens[end+1:]
    )

    if new_t.shape[1] != len(new_input_tokens):
        raise RuntimeError("Desajuste entre columnas de token_attr y input_tokens.")

    return LLMAttributionResult(
        seq_attr=attr_res.seq_attr,
        token_attr=new_t,
        input_tokens=new_input_tokens,
        output_tokens=attr_res.output_tokens
    )

def collapse_multiple_spans(
    attr_res: LLMAttributionResult,
    starts: List[int],
    ends: List[int],
    names: Optional[List[str]] = None,
    reduce: str = "mean",
) -> LLMAttributionResult:
    """
    Collapses multiple spans [start_i, end_i] -> one column each,
    replacing names with names[i]. Equivalent to calling 
    collapse_input_span multiple times.
    Spans are processed from right to left.
    
    Example:
        starts = [0, 3]
        ends = [1, 4]
        names = ["/hello/", "/world/"]
        
        output: token_attr with columns 0-1 replaced by "/hello/" and
                columns 3-4 replaced by "/world/"
    """
    if len(starts) != len(ends):
        raise ValueError("starts y ends deben tener la misma longitud.")

    n = len(starts)
    if names is None:
        names = [f"[SPAN_{i}]" for i in range(n)]
    elif len(names) != n:
        raise ValueError("If given, names should have the same length as starts/ends.")

    spans = sorted(zip(starts, ends, names), key=lambda x: x[0])  # ascendente
    for i in range(len(spans)):
        s, e, _ = spans[i]
        if e < s:
            raise ValueError(f"Invalid span: ({s}, {e})")
        if i > 0:
            prev_s, prev_e, _ = spans[i-1]
            if s <= prev_e:  # solapan (o contiguos con s == prev_e+1 son válidos si quieres)
                raise ValueError(f"Overlapped spans: ({prev_s},{prev_e}) y ({s},{e})")

    for s, e, nm in sorted(spans, key=lambda x: x[0], reverse=True):
        attr_res = _collapse_input_span(attr_res, s, e, nm, reduce=reduce)

    return attr_res

def save_attr_plot(attr_res, output_dir, filename, *, dpi=200):
    """
    Generates the plot with show=False, saves to output_dir/plots/filename,
    and closes the figure. Returns the full path.
    """
    res = attr_res.plot_token_attr(show=False)

    # Algunas versiones devuelven (Figure, Axes); otras solo Figure
    if isinstance(res, tuple):
        fig, ax = res
    else:
        fig = res

    if fig is None:
        raise RuntimeError(
            "plot_token_attr devolvió None. Asegúrate de usar show=False "
            "y de que tu versión devuelve la Figure."
        )

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def resolve_resume_state(
    config: Dict[str, Any],
    out_tsv: str,
    total_samples: int
) -> tuple[int, bool]:
    """
    Determines the starting index and whether to reset the TSV file based on the configuration.
    Returns a tuple (start_index, reset_tsv).
    """
    start_index = 0
    reset_tsv = True  # comportamiento por defecto

    if config.get("start_index") is not None:
        start_index = max(0, int(config["start_index"]))
        reset_tsv = False
    elif config.get("resume", False):
        _, count_rows = compute_mean_scores_from_tsv(out_tsv)
        start_index = int(count_rows or 0)
        reset_tsv = False
    else:
        reset_tsv = True

    if start_index > 0:
        print(f"[INFO] Resuming from sample #{start_index} (TSV: {out_tsv})")
    if start_index >= total_samples:
        print(f"[WARN] start_index={start_index} >= total_samples={total_samples}. Nothing to process.")

    return start_index, reset_tsv

def compute_mean_scores_from_tsv(tsv_file: str) -> Tuple[Optional[List[float]], int, List[str]]:
    """
    Read a TSV with header and data rows, compute the mean of numeric columns,
    and return (means, row_count, column_names).
    """
    if not os.path.exists(tsv_file):
        return None, 0, []

    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return None, 0, []

        colnames: List[str] = []
        rows = list(reader)
        if not rows:
            return None, 0, []

        # detect numeric columns
        for name in reader.fieldnames:
            try:
                float(rows[0][name])
                colnames.append(name)
            except (TypeError, ValueError):
                pass

        if not colnames:
            return None, 0, []

        sums = [0.0 for _ in colnames]
        count = 0
        for row in rows:
            try:
                vals = [float(row[c]) for c in colnames]
                sums = [s + v for s, v in zip(sums, vals)]
                count += 1
            except (TypeError, ValueError):
                continue

        if count == 0:
            return None, 0, []

        means = [s / count for s in sums]
        return means, count, colnames
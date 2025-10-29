
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import os

from utils import (
    load_attr_method,
    group_scores,
    collapse_multiple_spans,
    build_audio_spans,
    save_attr_plot,
    compute_mean_scores_from_tsv,
    resolve_resume_state,
    call_attribute,
    is_gradient_method,
)

from data import (
    parse_args,
    load_model,
    load_data,
    extract_template,
    load_config
)

def run_attribution(
    *,
    attr_method: str,  
    input_texts: List[str],
    generated_texts: List[str],
    aligned_huberts: List[Dict[str, List[Any]]],
    tokenizer: PreTrainedTokenizer,
    llm_attr: any,
    skip_tokens: List[str],
    output_dir: Optional[Union[str, Path]],
    input_translation: bool = False,
    attribution_kwargs: Dict[str, Any] = None,
    start_index: int = 0,
    reset_tsv: bool = False,
    plot_every: int = 50,
) -> None:
    """
    Unified attribution loop for both gradient-based and perturbation-based methods.
    """
    
    if output_dir:
        tsv_path = os.path.join(str(output_dir), "scores.tsv")
        if reset_tsv and os.path.exists(tsv_path):
            try:
                os.remove(tsv_path)
            except Exception as e:
                print(f"[WARN] Could not restart TSV: {type(e).__name__}: {e}")

    iterator = enumerate(zip(input_texts, generated_texts, aligned_huberts))
    total_n = len(input_texts)

    for i, (input_text, generated_text, aligned_hubert_words) in tqdm(iterator, total=total_n):
        if i < start_index:
            continue

        template, values, mask = extract_template(
            input_text, tokenizer, aligned_hubert_words, input_translation
        )
        
        attr_res = call_attribute(
            attr_method=attr_method,
            template=template,
            values=values,
            mask=mask,
            llm_attr=llm_attr,
            generated_text=generated_text,
            skip_tokens=skip_tokens,
            attribution_kwargs=attribution_kwargs,
        )
        
        # NO MIRADO AUN
        audio_words, audio_start_idx, audio_end_idx = build_audio_spans(aligned_hubert_words)
        format_attr_res = collapse_multiple_spans(
            attr_res, audio_start_idx, audio_end_idx, audio_words, reduce="mean"
        )
        len_transc = sum("transctok" in key for key in values)
        sft_attr_res = group_scores(
            format_attr_res, len(audio_words), len_transc, tsv_path=tsv_path
        )

        if output_dir and (i + 1) % plot_every == 0:
            save_attr_plot(sft_attr_res, output_dir, f"sample_{i}_collapsed.png", dpi=300)

def main(config):
    model, tokenizer = load_model(config.get("model", None))
    output_dir = config.get("results").get("output_dir", "./outputs")
    data_cfg = config.get("data", {})
    attr_method = config.get("attr_params", {}).get("method")

    for lang_pair in data_cfg.get("lang_pairs", []):
        out_dir_lang = Path(output_dir) / lang_pair
        out_dir_lang.mkdir(parents=True, exist_ok=True)

        llm_attr = load_attr_method(attr_method, model, tokenizer)
        input_texts, generated_texts, aligned_huberts = load_data(data_cfg=data_cfg)

        # Handle resuming from previous runs
        if config.get("resume", None) or config.get("start_index", None):
            out_tsv = os.path.join(out_dir_lang, "scores.tsv")
            start_index, reset_tsv = resolve_resume_state(config, out_tsv, len(input_texts))
        
        run_attribution(
            attr_method = attr_method,
            input_texts=input_texts,
            generated_texts=generated_texts,
            aligned_huberts=aligned_huberts,
            tokenizer=tokenizer,
            llm_attr=llm_attr,
            output_dir=out_dir_lang,
            skip_tokens=config.get("attr_params").get("skip_tokens", []),
            input_transl=config.get("input_translation", False),
            attribution_kwargs=config.get("attr_params").get("attribution_kwargs", {}),
            start_index=start_index,
            reset_tsv=reset_tsv,
            plot_every=config.get("results").get("plot_every", 50),
        )

        # Compute and print mean scores
        tsv_file = os.path.join(out_dir_lang, "scores.tsv")
        means, count, colnames = compute_mean_scores_from_tsv(tsv_file)

        if means is not None and count > 0:
            stats = ", ".join(f"{name}: {val:.4f}" for name, val in zip(colnames, means))
            print(f"[INFO] [{lang_pair}] Mean scores -> {stats} (n={count})")
        else:
            if os.path.exists(tsv_file):
                print(f"[WARN] [{lang_pair}] Empty TSV or no data in: {tsv_file}")
            else:
                print(f"[WARN] [{lang_pair}] No TSV found for: {tsv_file}")
                    
if __name__ == "__main__":
    cli_args = parse_args()
    config = load_config(cli_args)
    main(config)

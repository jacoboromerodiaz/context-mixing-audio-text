import ast
import re
import torch
from typing import Optional, Dict, List, Any, Tuple
from transformers import LlamaForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
import os
import yaml

def parse_args():
    p = argparse.ArgumentParser(description="Attribution runner (YAML config)")
    p.add_argument("--config", type=Path, required=True, help="Ruta al archivo YAML de configuración")
    return p.parse_args()

def load_config(cli_args):
    with open(cli_args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return argparse.Namespace(**cfg)

def load_model(CKPT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained(CKPT,device_map="auto",torch_dtype=torch.float16)
    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    return model, tokenizer

def _load_mappings(txt_path):
    """
    Load a .txt file where each line contains a dictionary in str(dict) format.
    Returns a list of dictionaries.
    """
    if not os.path.exists(txt_path):
        msg = (f"Archivo {txt_path} not found. please run forced_alignment.py to obtain "
               f"the words aligned with hubert tokens")
        print(msg)
        raise FileNotFoundError(msg)

    mappings = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    d = ast.literal_eval(line)  # convert str -> dict
                    mappings.append(d)
                except Exception as e:
                    print(f"Error parsing line: {line}\n{e}")
    return mappings

def load_data(data_cfg, lang_pair):
    """
    Read the full sequence file, split it into samples of `chunk_size`.
    Alsos load HuBERT alignments, and (optionally) deduplicate their tokens.
    Returns:
        - original_prompts: list of original prompts (or the whole chunk if input_transl=True)
        - generated_texts: list with the generated text after the splitter
        - aligned_huberts: list of dicts (possibly deduplicated)
    """
    # Retrieving data from files
    original_prompts, generated_texts = _load_prompts(data_cfg, lang_pair)
    
    # Retrieving aligned hubert tokens
    aligned_huberts_file = data_cfg.get("aligned_huberts_file", None).format(lang_pair=lang_pair)
    aligned_huberts = _load_mappings(aligned_huberts_file)
    if data_cfg.get("deduplicate_huberts", False):
        aligned_huberts = _deduplicate_huberts(aligned_huberts)

    return original_prompts, generated_texts, aligned_huberts

def _load_prompts(data_cfg, lang_pair):
    data_file = data_cfg.get("file", None)
    chunk_size = data_cfg.get("chunk_size", None)
    
    if chunk_size is None: print("Warning: chunk_size not set, defaulting to 1"), chunk_size = 1 
    
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    chunks = [''.join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]

    # Divide into prompts and generated texts
    original_prompts, generated_texts = [], []
    
    # mode 1: entire_seq_file + splitter
    if "entire_seq_file" in data_cfg:
        entire_seq_file = Path(data_cfg["entire_seq_file"].format(lang_pair=lang_pair))
        with open(entire_seq_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        chunks = ["".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]

        splitter = data_cfg.get("splitter", None)
        if splitter is None:
            raise ValueError("Invalid config: `data.splitter` is required when using `entire_seq_file`.")
        for i, chunk in enumerate(chunks):
            parts = re.split(splitter, chunk)
            if len(parts) < 3:
                print(f"Sample number {i} not segmented correctly\nSample: {''.join(parts)}")
                continue

            original_prompt = parts[0] + parts[1]
            generated_text = "".join(parts[2:])

            if data_cfg.get("input_translation", False):
                original_prompts.append(chunk) # keep full chunk if input_translation is True
            else:
                original_prompts.append(original_prompt)
            generated_texts.append(generated_text)

    # mode 2: prompts_file + generated_file
    elif "prompts_file" in data_cfg and "generated_file" in data_cfg:
        prompts_file = Path(data_cfg["prompts_file"].format(lang_pair=lang_pair))
        generated_file = Path(data_cfg["generated_file"].format(lang_pair=lang_pair))
        with open(prompts_file, "r", encoding="utf-8") as f:
            original_prompts = [line.strip() for line in f if line.strip()]
        with open(generated_file, "r", encoding="utf-8") as f:
            generated_texts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(
            "Invalid config: either `data.entire_seq_file` or both `data.prompts_file` and `data.generated_file` must be provided."
        )
    
def _deduplicate_huberts(aligned_huberts: List[Dict[str, List[Any]]]
                                  ) -> List[Dict[str, List[Any]]]:
    """
    Deduplicate HuBERT token lists per word, removing consecutive duplicates,
    trimming from the start any token equal to the previous word's last token,
    and enforcing min-1 token per word without inflating the total count by
    borrowing a token from the nearest neighbor with >1 tokens. 
    
    Additionally, if a word has exactly 1 token and it equals the previous word's 
    last token, attempt to replace it by borrowing a different token from the previous word.
    """
    deduplicated_huberts: List[Dict[str, List[Any]]] = []
    for d in aligned_huberts:
        # 1) Primera pasada: deduplicar internos y recortar solape con prev_last
        ordered_keys = list(d.keys())  # preservar orden
        collapsed_lists: List[List[Any]] = []
        prev_last = None  # último token de la palabra previa

        for (idx, key) in ordered_keys:
            values = d[(idx, key)]
            if not values:
                collapsed_lists.append([])
                continue

            collapsed = [values[0]]
            for v in values[1:]:
                last = collapsed[-1]
                if repr(v) != repr(last):
                    collapsed.append(v)

            if prev_last is not None and collapsed:
                while collapsed and (repr(collapsed[0]) == repr(prev_last)):
                    collapsed.pop(0)

            collapsed_lists.append(collapsed)
            prev_last = collapsed[-1] if collapsed else prev_last

        # 2a) Si queda exactamente 1 token igual al último de la palabra anterior,
        #     intentar sustituirlo por uno distinto prestado de la anterior.
        for i in range(len(collapsed_lists)):
            if len(collapsed_lists[i]) != 1:
                continue
            # buscar anterior no vacía
            j = i - 1
            while j >= 0 and len(collapsed_lists[j]) == 0:
                j -= 1
            if j < 0 or len(collapsed_lists[j]) == 0:
                continue
            only_tok = collapsed_lists[i][0]
            prev_last_tok = collapsed_lists[j][-1]
            if repr(only_tok) == repr(prev_last_tok):
                taken = False
                # buscar un candidato distinto en la anterior (de derecha a izquierda, sin el último)
                for k in range(len(collapsed_lists[j]) - 2, -1, -1):
                    cand = collapsed_lists[j][k]
                    if repr(cand) != repr(prev_last_tok):
                        collapsed_lists[i] = [cand]
                        collapsed_lists[j].pop(k)
                        taken = True
                        break
                if not taken:
                    # dejar vacía; se resolverá en el préstamo general
                    collapsed_lists[i] = []

        # 2b) Garantizar mínimo 1 token por palabra SIN inflar el total, por préstamo
        def borrow_for(i: int) -> None:
            # izquierda
            L = None
            dist = 1
            while i - dist >= 0:
                if len(collapsed_lists[i - dist]) > 1:
                    L = (i - dist, dist)
                    break
                dist += 1
            # derecha
            R = None
            dist = 1
            while i + dist < len(collapsed_lists):
                if len(collapsed_lists[i + dist]) > 1:
                    R = (i + dist, dist)
                    break
                dist += 1

            donor_idx = None
            if L and R:
                donor_idx = L[0] if L[1] <= R[1] else R[0]
            elif L:
                donor_idx = L[0]
            elif R:
                donor_idx = R[0]

            if donor_idx is None:
                return

            if donor_idx < i:
                tok = collapsed_lists[donor_idx].pop()
                collapsed_lists[i].insert(0, tok)
            else:
                tok = collapsed_lists[donor_idx].pop(0)
                collapsed_lists[i].append(tok)

        empties = [i for i, lst in enumerate(collapsed_lists) if len(lst) == 0]
        for i in empties:
            borrow_for(i)

        # 3) Reconstruir dict preservando orden
        new_d: Dict[str, List[Any]] = {}
        for idx_i, (idx, key) in enumerate(ordered_keys):
            new_d[(idx, key)] = collapsed_lists[idx_i]

        deduplicated_huberts.append(new_d)
    print("Deduplicated hubert tokens")
    
    return deduplicated_huberts

def _norm(s: Optional[str]) -> str:
    """Collapse line breaks and trim."""
    s = s or ""
    return re.sub(r'(?:\r\n|\r|\n|\\n)+', ' ', s).strip()

def _parse_user_blocks(text: str) -> Tuple[str, str, str]:
    """Extract audio segment + language1, and (optionally) language2 from user blocks."""
    audio_regex = re.search(
        r"<\|im_start\|>user\s*(.*?)\s*Transcribe in ([\w\s]+)<\|im_end\|>",
        text, re.DOTALL
    )
    audio_segment = audio_regex.group(1).strip() if audio_regex else ""
    lang1 = audio_regex.group(2).strip() if audio_regex else ""

    lang2_match = re.search(r"Translate to ([\w\s]+)", text)
    lang2 = lang2_match.group(1).strip() if lang2_match else ""

    return audio_segment, lang1, lang2

def _parse_assistant_blocks(text: str, input_transl: bool) -> Tuple[str, str]:
    """Extract assistant transcription (first block) and optional translation (last block)."""
    assistant_blocks = re.findall(
        r"<\|im_start\|>assistant\s*(.*?)\s*(?=(<\|im_end\|>|$))",
        text, re.DOTALL
    )
    transc_segment = assistant_blocks[0][0].strip() if assistant_blocks else ""
    transl_segment = (
        assistant_blocks[-1][0].strip()
        if (assistant_blocks and input_transl and len(assistant_blocks) > 1)
        else ""
    )
    return transc_segment, transl_segment

def _tokenize_segments(tokenizer, audio_segment: str, transc_segment: str,
                       transl_segment: str, input_transl: bool) -> List[List[str]]:
    """Tokenize selected segments without adding special tokens."""
    texts = [audio_segment, transc_segment]
    if input_transl:
        texts.append(transl_segment)
    return [
        tokenizer.convert_ids_to_tokens(tokenizer(p, add_special_tokens=False)["input_ids"])
        for p in texts
    ]

def _build_template(token_lists: List[List[str]], input_transl: bool) -> str:
    """Build the string template with placeholders (same as original behavior)."""
    audiotok_str = ''.join([f"{{audiotok_{i}}}" for i in range(len(token_lists[0]))])
    transctok_str = ''.join([f"{{transctok_{i}}}" for i in range(len(token_lists[1]))])

    audio_template = "<|im_start|>user\n" + audiotok_str + "\n"
    transc_template = "Transcribe in {language1}<|im_end|>\n<|im_start|>assistant\n" + transctok_str + "\n"
    transl_template = "<|im_end|><|im_start|>user\nTranslate to {language2}<|im_end|>\n<|im_start|>assistant\n"

    if input_transl:
        transltok_str = ''.join([f"{{transltok_{i}}}" for i in range(len(token_lists[2]))])
        transl_template += transltok_str

    return audio_template + transc_template + transl_template

def _build_values(token_lists: List[List[str]], lang1: str, lang2: str, input_transl: bool) -> Dict[str, str]:
    """Build the values dict for all placeholders."""
    values: Dict[str, str] = {}
    values.update({f"audiotok_{i}": tok for i, tok in enumerate(token_lists[0])})
    values["language1"] = lang1
    values.update({f"transctok_{i}": tok for i, tok in enumerate(token_lists[1])})
    values["language2"] = lang2
    if input_transl:
        values.update({f"transltok_{i}": tok for i, tok in enumerate(token_lists[2])})
    return values

def _build_mask(aligned_hubert_words: Dict[str, List[Any]],
                token_lists: List[List[str]],
                input_transl: bool) -> Dict[str, int]:
    """Build the mask mapping each placeholder to a group index (same ordering/logic)."""
    mask: Dict[str, int] = {}
    counter = 0
    for word_idx, (_, tokens) in enumerate(aligned_hubert_words.items()):
        for _ in tokens:
            mask[f"audiotok_{counter}"] = word_idx
            counter += 1

    mask["language1"] = max(mask.values()) + 1
    start_transc = max(mask.values()) + 1
    for i in range(len(token_lists[1])):
        mask[f"transctok_{i}"] = start_transc + i

    mask["language2"] = max(mask.values()) + 1
    if input_transl:
        start_transl = max(mask.values()) + 1
        for i in range(len(token_lists[2])):
            mask[f"transltok_{i}"] = start_transl + i

    return mask

def extract_template(input_text: str, 
                     tokenizer, 
                     aligned_hubert_words: Dict[str, List[Any]],
                     input_translation: bool = False):
    """
    From the full input text and aligned hubert tokens, extract:
      - template: string with placeholders
      - values: dict mapping placeholders to actual tokens/words
      - mask: dict mapping placeholders to group indices
    """
    
    # --- parse blocks ---
    audio_segment, lang1, lang2 = _parse_user_blocks(input_text)
    transc_segment, transl_segment = _parse_assistant_blocks(input_text, input_translation)

    # --- normalize ---
    audio_segment = _norm(audio_segment)
    transc_segment = _norm(transc_segment)
    transl_segment = _norm(transl_segment)

    # --- tokenize ---
    token_lists = _tokenize_segments(tokenizer, audio_segment, transc_segment, transl_segment, input_translation)

    # --- alignment check (same assertion) ---
    total_audio_toks = sum(len(v) for v in aligned_hubert_words.values())
    assert len(token_lists[0]) == total_audio_toks, (
        f"Desajuste audio: {len(token_lists[0])} vs {total_audio_toks}"
    )

    # --- templates, values, mask ---
    template = _build_template(token_lists, input_translation)
    values = _build_values(token_lists, lang1, lang2, input_translation)
    mask = _build_mask(aligned_hubert_words, token_lists, input_translation)

    return template, values, mask

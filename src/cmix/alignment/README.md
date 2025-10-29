# Forced Alignment Module

This module performs word-level alignment between HuBERT tokens and reference transcriptions using the [torchaudio MMS Forced Alignment pipeline](https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html).

---

## Purpose

The **forced alignment** step is optional and only necessary if your Speech LLM output (e.g., `entire_sequence.txt`) contains *deduplicated* HuBERT tokens.  
If your system already generates aligned or frame-collapsed audio representations, you can skip this step.

---

## Usage

### 1. YAML Configuration File

Create a YAML file specifying the paths and parameters for the alignment process. Example:

```yaml
audio:
  dir: /path/to/audio_wavs/
  tsv: /path/to/{lang_pair}.test.tsv

huberts:
  tokens_path: /path/to/{lang_pair}.audio_target.txt
  token_dur_ms: 40

output_path: /path/to/output/
lang_pairs: ["en_es", "en_ca", "en_pt"]
```
### 2. Run Alignment

```bash
python -m cmix.alignment.forced_alignment --config config/alignment/fleurs_iber_alignment.yaml
```

> Optional: override `lang_pairs` directly from the CLI
```bash
python -m cmix.alignment.forced_alignment \
  --config config/alignment/fleurs_iber_alignment.yaml \
  --lang-pairs en_es en_ca
```

---

### 3. Output

For each `{lang_pair}`, this will create:

```
out/{lang_pair}.aligned_huberts.json
```

Each line is a dictionary mapping the transcript words to their associated HuBERT tokens, for example:

```python
{
  (0, 'remember'): ['\uF0007', '\uF01E8', '\uF01E8', '\uF01E5'],
  (1, 'that'): ['\uF00D9', '\uF0139', '\uF001E', '\uF00C1', '\uF00E2', '\uF0007', '\uF0112'],
  (2, 'even'): ['\uF0112', '\uF018F', '\uF006E', '\uF006E', '\uF00A6'],
  (3, 'though'): ['\uF005E', '\uF0139', '\uF01B1', '\uF013C', '\uF00B4', '\uF0130'],
  (4, 'music'): ['\uF01CA', '\uF0167', '\uF001D', '\uF00A5', '\uF00A5', '\uF01CB'],
  (5, 'on'): ['\uF00B8', '\uF01B0', '\uF01D3', '\uF00DF'],
  (6, 'the'): ['\uF0139', '\uF0006', '\uF0007', '\uF0130']
}
```

---

### 4. Integration with the Main Pipeline

The aligned files are later consumed by the main attribution pipeline  
(`main.py` via `data.load_data()`), to synchronize token- and word-level attributions.

---

### Notes

- Requires a GPU and the `MMS_FA` bundle from `torchaudio`.
- Optional parameters such as `token_dur_ms` can be customized in the YAML file.
- Language pairs are automatically expanded using the `{lang_pair}` placeholder.

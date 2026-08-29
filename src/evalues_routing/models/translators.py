from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence


@dataclass
class TranslationBatch:
    texts: list[str]
    per_example_seconds: float
    backend: str
    model_id: str


class TranslationManager:
    """Local source-to-English translation with a pinned backend per language.

    Only one OPUS model is retained at a time to avoid accumulating many bilingual
    models on GPU memory. NLLB is cached separately because it is reused for several
    low-resource languages.
    """

    def __init__(self, config: dict, device: str):
        self.cfg = config['models']['translation']
        self.device = device
        self._current_opus_id = None
        self._current_opus = None
        self._nllb = None

    def _torch_device(self):
        import torch
        if self.device == 'cuda':
            return torch.device('cuda')
        if self.device == 'mps':
            return torch.device('mps')
        return torch.device('cpu')

    def _model_kwargs(self):
        import torch
        if self.device == 'cuda':
            return {'torch_dtype': torch.float16}
        return {}

    def _release_opus(self):
        if self._current_opus is not None:
            del self._current_opus
            self._current_opus = None
            self._current_opus_id = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _load_opus(self, model_id: str, revision: str | None = None):
        from transformers import MarianMTModel, MarianTokenizer
        cache_key = (model_id, revision)
        if self._current_opus_id != cache_key or self._current_opus is None:
            self._release_opus()
            revision_kwargs = {'revision': revision} if revision else {}
            tok = MarianTokenizer.from_pretrained(model_id, **revision_kwargs)
            model = MarianMTModel.from_pretrained(
                model_id, **revision_kwargs, **self._model_kwargs()
            ).to(self._torch_device())
            model.eval()
            self._current_opus_id = cache_key
            self._current_opus = (tok, model)
        return self._current_opus

    def _load_nllb(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_model
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
        if self._nllb is None:
            model_id = self.cfg['nllb_model']
            revision = self.cfg.get('nllb_revision')
            revision_kwargs = {'revision': revision} if revision else {}
            tok = AutoTokenizer.from_pretrained(model_id, **revision_kwargs)
            # The immutable safetensors conversion stores tied embeddings only
            # once. Transformers 4.55 + Torch 2.2 can leave that shared tensor
            # on the meta device when using from_pretrained, so initialize the
            # architecture normally and load the safe tensor file in place.
            model_config = AutoConfig.from_pretrained(model_id, **revision_kwargs)
            model = AutoModelForSeq2SeqLM.from_config(model_config)
            checkpoint = hf_hub_download(model_id, 'model.safetensors', **revision_kwargs)
            missing, unexpected = load_model(model, checkpoint, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f'NLLB safetensors mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}'
                )
            model = model.to(self._torch_device())
            model.eval()
            self._nllb = (tok, model)
        return self._nllb

    def _generate(self, tok, model, texts, forced_bos_token_id=None):
        import torch
        batch = tok(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
        batch = {k: v.to(self._torch_device()) for k, v in batch.items()}
        kwargs = {
            'max_new_tokens': int(self.cfg.get('max_new_tokens', 256)),
            'num_beams': int(self.cfg.get('num_beams', 1)),
        }
        if forced_bos_token_id is not None:
            kwargs['forced_bos_token_id'] = forced_bos_token_id
        with torch.inference_mode():
            generated = model.generate(**batch, **kwargs)
        return tok.batch_decode(generated, skip_special_tokens=True)

    def translate(self, texts: Sequence[str], source_code: str) -> TranslationBatch:
        if not texts:
            return TranslationBatch([], 0.0, 'none', '')

        opus_map = self.cfg.get('opus_models', {})
        use_opus = source_code in opus_map
        batch_size = int(self.cfg.get('batch_size', 8))
        outputs: list[str] = []
        backend = 'nllb'
        model_id = self.cfg['nllb_model']
        elapsed = 0.0

        if use_opus:
            model_spec = opus_map[source_code]
            if isinstance(model_spec, str):
                model_id, revision = model_spec, None
            else:
                model_id = model_spec['id']
                revision = model_spec.get('revision')
            tok, model = self._load_opus(model_id, revision)
            backend = 'opus'
            start = perf_counter()  # model cold-start is intentionally excluded
            for i in range(0, len(texts), batch_size):
                outputs.extend(self._generate(tok, model, texts[i:i + batch_size]))
            elapsed = perf_counter() - start

        if backend == 'nllb':
            self._release_opus()
            tok, model = self._load_nllb()
            tok.src_lang = source_code
            lang_map = getattr(tok, 'lang_code_to_id', {}) or {}
            target_id = lang_map.get('eng_Latn')
            if target_id is None:
                target_id = tok.convert_tokens_to_ids('eng_Latn')
            if target_id is None or target_id == tok.unk_token_id:
                raise ValueError('NLLB tokenizer does not expose eng_Latn target token')
            start = perf_counter()  # model cold-start is intentionally excluded
            for i in range(0, len(texts), batch_size):
                outputs.extend(
                    self._generate(tok, model, texts[i:i + batch_size], forced_bos_token_id=target_id)
                )
            elapsed = perf_counter() - start

        return TranslationBatch(outputs, elapsed / len(texts), backend, model_id)

"""FireRedTTS3 — unified speech generation & editing demo (Local PC Version).

Supports automatic fallback between CUDA (GPU) and CPU.
"""

import functools
import os
import re
import requests
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import gradio as gr
from huggingface_hub import snapshot_download
try:
    requests.head('https://huggingface.co',timeout=3)
except:
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'

HERE = Path(os.path.dirname(os.path.abspath(__file__))).as_posix()

# --------------------------------------------------------------------------- #
# Device Detection (自动检测 CUDA / CPU)
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using compute device: {DEVICE}", flush=True)
if DEVICE == "cpu":
    print("[WARN] Running on CPU. Generation speed might be slower.", flush=True)

# --------------------------------------------------------------------------- #
# Text front-end assets
# --------------------------------------------------------------------------- #
_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_LID_PATH = os.path.join(HERE, "fireredtts3", "utils", "llm_tn", "models", "lid.176.ftz")
os.makedirs(os.path.dirname(_LID_PATH), exist_ok=True)
if not os.path.exists(_LID_PATH):
    try:
        urllib.request.urlretrieve(_LID_URL, _LID_PATH)
        print(f"[INFO] fastText lid.176 downloaded to {_LID_PATH}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not fetch fastText lid.176: {exc}", flush=True)

os.environ.setdefault("LLM_TN_API_URL", "http://127.0.0.1:1/unused")
os.environ.setdefault("LLM_TN_API_KEY", "unused")

# --------------------------------------------------------------------------- #
# Weights & Model Loading
# --------------------------------------------------------------------------- #
MODEL_REPO = "FireRedTeam/FireRedTTS3"
MODEL_DIR = f'{HERE}/pretrained_models'
if not Path(f'{MODEL_DIR}/model.safetensors').exists():
    snapshot_download(MODEL_REPO,local_dir=MODEL_DIR)
print(f"[INFO] Weights at {MODEL_DIR}", flush=True)

from fireredtts3.core import FireRedTTS3  # noqa: E402
from fireredtts3.redae.redae import RedAE  # noqa: E402
from fireredtts3.utils.llm_tn.text_normalizer import TextNormalizer  # noqa: E402
from fireredtts3.utils.text_tokenizer import (  # noqa: E402
    MULTI_DIALECT_TAGS,
    MULTI_LANG_TAGS,
)

# 共享 RedAE 节省内存/显存 (~3.8 GB)
_redae_real_from_pretrained = RedAE.from_pretrained
_redae_singleton = None


def _shared_redae(*args, **kwargs):
    global _redae_singleton
    if _redae_singleton is None:
        _redae_singleton = _redae_real_from_pretrained(*args, **kwargs)
    return _redae_singleton


RedAE.from_pretrained = _shared_redae

# 自动绑定到检测到的 device (cuda/cpu)
tts = FireRedTTS3(
    MODEL_DIR,
    use_fasttext=True,
    use_llm_tn=False,
    use_wetext=True,
    #device=DEVICE,
)


for _pipe in (tts,):
    _norm = getattr(_pipe, "_llm_tn", None)
    if _norm is not None:
        _norm.detect_locale = functools.partial(
            TextNormalizer.detect_locale, _norm, use_llm_fallback=False
        )
print(f"[INFO] FireRedTTS3 Base + Instruct ready on [{DEVICE}]", flush=True)

SAMPLE_RATE = tts.redae.sample_rate

# --------------------------------------------------------------------------- #
# Language choices
# --------------------------------------------------------------------------- #
AUTO = "Auto-detect"
LANGUAGES = [t.strip("<|>") for t in MULTI_LANG_TAGS]
DIALECTS = [t.strip("<|>") for t in MULTI_DIALECT_TAGS]
LANG_CHOICES = [AUTO] + LANGUAGES + [f"{d} (Chinese dialect)" for d in DIALECTS]


def _resolve_language(choice: str):
    if not choice or choice == AUTO:
        return None
    return choice.split(" (")[0]


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
MAX_PROMPT_SECONDS = 20.0
MAX_EDIT_SECONDS = 20.0
MAX_TEXT_CHARS = 400


def _load_audio(path: str, max_seconds: float):
    if not path:
        raise gr.Error("Please provide an audio file first.")
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav[:, 0]
    if wav.shape[0] > int(max_seconds * sr):
        wav = wav[: int(max_seconds * sr)]
        gr.Info(f"Audio truncated to the first {max_seconds:.0f}s.")
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 0:
        wav = wav / peak * 0.95
    return torch.from_numpy(np.ascontiguousarray(wav)[None, :]), sr


def _to_gradio_audio(audio: torch.Tensor, sr: int):
    x = audio.detach().float().cpu().numpy()
    if x.ndim > 1:
        x = x[0]
    x = np.clip(x, -1.0, 1.0)
    return sr, (x * 32767.0).astype(np.int16)


_EDIT_MASK_RE = re.compile(r"<\|edit\|>(?:<\|frame_patch\|>)*<\|end_edit\|>")


def _pretty_edit_text(text: str) -> str:
    text = _EDIT_MASK_RE.sub(" ⟨edited span⟩ ", text or "")
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _check_text(text: str, what: str = "Text"):
    text = (text or "").strip()
    if not text:
        raise gr.Error(f"{what} must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        gr.Info(f"{what} truncated to {MAX_TEXT_CHARS} characters.")
        text = text[:MAX_TEXT_CHARS]
    return text


# --------------------------------------------------------------------------- #
# Inference Functions
# --------------------------------------------------------------------------- #
def voice_clone(
    prompt_audio,
    prompt_text,
    text,
    language=AUTO,
    inference_cfg=2.0,
    n_timesteps=10,
    seed=1234,
    do_tn=True,
):
    text = _check_text(text, "Text to synthesize")
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise gr.Error("Please provide the transcript of the reference audio.")
    wav, sr = _load_audio(prompt_audio, MAX_PROMPT_SECONDS)

    gen_audio, gen_sr = tts.generate(
        text=text,
        language=_resolve_language(language),
        prompt_text=prompt_text,
        prompt_audio=wav,
        prompt_audio_sr=sr,
        n_timesteps=int(n_timesteps),
        inference_cfg=float(inference_cfg),
        seed=int(seed),
        do_tn=bool(do_tn),
    )
    return _to_gradio_audio(gen_audio, gen_sr)



# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
EN_PROMPT = os.path.join(HERE, "examples", "en_prompt.wav")
ZH_PROMPT = os.path.join(HERE, "examples", "zh_prompt.wav")
EN_PROMPT_TEXT = (
    "Just by listening a few minutes a day, you'll be able to eliminate negative "
    "thoughts by conditioning your mind to be more positive."
)
ZH_PROMPT_TEXT = "比如具体一点的，他觉得最大的一个跟他预想的不一样的是在什么地方。"

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto !important;}
.dark .gradio-container {color: var(--body-text-color);}
"""

with gr.Blocks(title="FireRedTTS3") as demo:
    gr.Markdown(
        f"""
        # 🔥 FireRedTTS3 — Speech Generation & Editing
        *Running on: **{DEVICE.upper()}***
        """
    )

    with gr.Tabs():
        # ------------------------------------------------------------------ #
        with gr.Tab("🎙️ Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    clone_prompt_audio = gr.Audio(
                        label="Reference audio (5–20 s)",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    clone_prompt_text = gr.Textbox(
                        label="Reference transcript",
                        placeholder="Transcript of the reference audio…",
                        lines=2,
                    )
                    clone_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Type text to speak…",
                        lines=4,
                    )
                    clone_language = gr.Dropdown(
                        LANG_CHOICES, value=AUTO, label="Language / dialect"
                    )
                    clone_btn = gr.Button("Generate speech", variant="primary")
                with gr.Column():
                    clone_out = gr.Audio(label="Generated speech", type="numpy")
                    with gr.Accordion("Advanced options", open=False):
                        clone_cfg = gr.Slider(
                            0.0, 4.0, value=2.0, step=0.1, label="CFG strength"
                        )
                        clone_steps = gr.Slider(
                            4, 30, value=10, step=1, label="Flow-matching timesteps"
                        )
                        clone_seed = gr.Number(value=1234, precision=0, label="Seed")
                        clone_tn = gr.Checkbox(value=True, label="Text normalization")

            # 仅当本地示例音频存在时展示示例
            if os.path.exists(EN_PROMPT) and os.path.exists(ZH_PROMPT):
                gr.Examples(
                    examples=[
                        [
                            EN_PROMPT,
                            EN_PROMPT_TEXT,
                            "FireRedTTS3 turns a handful of seconds of speech into a voice.",
                            "English",
                        ],
                        [
                            ZH_PROMPT,
                            ZH_PROMPT_TEXT,
                            "法院与不动产登记部门加强沟通，并督促银行提前办理抵押预约登记。",
                            "Chinese",
                        ],
                    ],
                    inputs=[clone_prompt_audio, clone_prompt_text, clone_text, clone_language],
                    outputs=[clone_out],
                    fn=voice_clone,
                    cache_examples=False,
                )

        # ------------------------------------------------------------------ #


    def _attr_changed(attribute, current):
        lo, hi, step, label = {
            "Speed": (0.5, 2.0, 0.1, "Speed (×)"),
            "Volume": (0.3, 2.0, 0.1, "Volume (×)"),
        }.get(attribute, (-6, 6, 1, "Pitch shift (semitone steps)"))
        try:
            value = min(max(float(current), lo), hi)
        except (TypeError, ValueError):
            value = lo
        if attribute == "Pitch":
            value = int(round(value)) or 1
        return gr.update(minimum=lo, maximum=hi, step=step, value=value, label=label)



    clone_btn.click(
        voice_clone,
        inputs=[clone_prompt_audio, clone_prompt_text, clone_text, clone_language,
                clone_cfg, clone_steps, clone_seed, clone_tn],
        outputs=[clone_out],
    )



if __name__ == "__main__":
    # 本地启动，自动打开浏览器
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=True,
        theme=gr.themes.Citrus(),
        css=CSS,
    )
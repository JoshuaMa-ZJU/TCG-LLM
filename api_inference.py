import os
import io
import re
import json
import time
import base64
import argparse
import numpy as np
from glob import glob
from PIL import Image
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


GPH_LEVELS = ["200 hPa", "500 hPa", "700 hPa", "850 hPa", "925 hPa", "1000 hPa"]


@dataclass
class InferenceConfig:
    data_folder: str = "/root/autodl-tmp/TCDLD/image"
    label_folder: str = "/root/autodl-tmp/TCDLD/label"
    docs_folder: str = "/root/autodl-tmp/TCDLD/docs"
    gph_folder: str = "/root/autodl-tmp/TCDLD/gph"
    gph_docs_folder: str = "/root/autodl-tmp/TCDLD/gph_docs"
    sst_folder: str = "/root/autodl-tmp/TCDLD/sst"
    sst_docs_folder: str = "/root/autodl-tmp/TCDLD/sst_docs"

    provider: str = "gpt"

    gpt_model: str = "gpt-5.2"
    claude_model: str = "claude-opus-4-5"
    gemini_model: str = "gemini-3-pro"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    use_gph: bool = True
    use_sst: bool = True
    use_cot: bool = True
    use_docs: bool = True
    doc_max_chars: int = 1200

    test_start_date: str = "20230701"

    output_path: str = "results/api_output.jsonl"

    max_retries: int = 5
    retry_delay: float = 2.0
    request_delay: float = 0.5

    image_max_size: int = 768
    image_quality: int = 90


def array_to_pil(img_array: np.ndarray, invert: bool = True) -> Image.Image:
    if hasattr(img_array, 'filled'):
        img_array = img_array.filled(0)
    img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = np.percentile(img_array, [2, 98])
    if vmax == vmin:
        img_normalized = np.zeros_like(img_array, dtype=np.uint8)
    else:
        img_normalized = np.clip((img_array - vmin) / (vmax - vmin), 0, 1) * 255
    img_uint8 = img_normalized.astype(np.uint8)
    if invert:
        img_uint8 = 255 - img_uint8
    return Image.fromarray(img_uint8).convert('RGB')


def pil_to_base64(pil_img: Image.Image, max_size: int = 768, quality: int = 90) -> str:
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_satellite_image(npy_path: str) -> Image.Image:
    data = np.load(npy_path, allow_pickle=True).item()
    return array_to_pil(data['image'], invert=True)


def load_gph_images(base_name: str, gph_folder: str) -> List[Image.Image]:
    base = base_name.replace('_image', '_gph')
    gph_path = os.path.join(gph_folder, f"{base}.npy")
    if not os.path.exists(gph_path):
        alt_base = base_name.replace('_image', '') + '_gph'
        gph_path = os.path.join(gph_folder, f"{alt_base}.npy")
    if not os.path.exists(gph_path):
        return []

    gph_arr = np.load(gph_path, allow_pickle=True)
    if hasattr(gph_arr, 'filled'):
        gph_arr = gph_arr.filled(0)
    gph_arr = np.nan_to_num(gph_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if len(gph_arr.shape) == 3:
        if gph_arr.shape[2] <= gph_arr.shape[0] and gph_arr.shape[2] <= gph_arr.shape[1]:
            gph_arr = np.transpose(gph_arr, (2, 0, 1))

    images = []
    for i in range(min(gph_arr.shape[0], 6)):
        level_data = gph_arr[i]
        pil_img = array_to_pil(level_data, invert=False)
        images.append(pil_img)
    return images


def load_sst_image(base_name: str, sst_folder: str) -> Optional[Image.Image]:
    base = base_name.replace('_image', '_sst')
    sst_path = os.path.join(sst_folder, f"{base}.npy")
    if not os.path.exists(sst_path):
        alt_base = base_name.replace('_image', '') + '_sst'
        sst_path = os.path.join(sst_folder, f"{alt_base}.npy")
    if not os.path.exists(sst_path):
        return None

    sst_arr = np.load(sst_path, allow_pickle=True)
    if hasattr(sst_arr, 'filled'):
        sst_arr = sst_arr.filled(0)
    sst_arr = np.nan_to_num(sst_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if len(sst_arr.shape) == 3:
        sst_arr = sst_arr[0]
    return array_to_pil(sst_arr, invert=False)


def load_doc(base_name: str, docs_folder: str, suffix: str = "", max_chars: int = 1200) -> str:
    stem = base_name.replace('_image', '')
    candidates = [
        os.path.join(docs_folder, f"{base_name}{suffix}.md"),
        os.path.join(docs_folder, f"{stem}{suffix}.md"),
        os.path.join(docs_folder, f"{stem}.md"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n..."
                return text
            except Exception:
                continue
    return ""


def load_label(npy_path: str, label_folder: str) -> Dict[str, Any]:
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    label_basename = basename.replace('_image', '_label')
    label_path = os.path.join(label_folder, f"{label_basename}.npy")

    if not os.path.exists(label_path):
        search_base = basename.replace('_image', '')
        candidates = glob(os.path.join(label_folder, f"{search_base}*label.npy"))
        if candidates:
            label_path = candidates[0]
        else:
            return {"tc_count": 0, "tc_positions": []}

    label_data = np.load(label_path, allow_pickle=True).item()
    positions = label_data.get('tc_positions', [])
    normalized = []
    for pos in positions:
        if isinstance(pos, dict) and 'lat' in pos and 'lon' in pos:
            normalized.append({"lat": float(pos['lat']), "lon": float(pos['lon'])})
        elif isinstance(pos, (list, tuple)) and len(pos) == 2:
            normalized.append({"lat": float(pos[0]), "lon": float(pos[1])})

    return {
        "tc_count": int(label_data.get('tc_count', 0)),
        "tc_positions": normalized,
    }


COT_INSTRUCTION = (
    "Please follow these steps to reason before answering:\n"
    "(1) Analyze the current input data to identify patterns of tropical cyclone "
    "occurrence, and determine whether tropical cyclones are forming globally.\n"
    "(2) If any are present, focus on locating (a) minima of geopotential height, "
    "(b) cold-core locations of sea surface temperature, and (c) the spiral center "
    "of the cloud system in the satellite imagery, to pinpoint the cyclone eye location(s).\n"
    "(3) Based on the current satellite image and environmental fields, identify all "
    "current tropical cyclones and their precise locations."
)

FORMAT_INSTRUCTION = (
    'Return the result strictly in the following JSON format:\n'
    '"current_tc_count": int, "current_tcs": [ {"lat": float, "lon": float} ]'
)

SYSTEM_PROMPT = (
    "You are an AI assistant specialized in tropical cyclogenesis detection and localization."
)

TASK_PROMPT = (
    "Your task is to use the satellite image, geopotential height data and sea surface "
    "temperature data together with the Markdown notes corresponding to each data to "
    "obtain the current TC numbers and positions."
)


def build_prompt_text(
    cfg: InferenceConfig,
    base_name: str,
    num_gph_images: int = 0,
) -> str:
    parts = [TASK_PROMPT + "\n"]

    if cfg.use_cot:
        parts.append(COT_INSTRUCTION + "\n")

    parts.append(
        "The following images are provided in order:\n"
        "1. Satellite cloud-top brightness temperature image\n"
    )
    if num_gph_images > 0:
        for i, level in enumerate(GPH_LEVELS[:num_gph_images]):
            parts.append(f"{i + 2}. Geopotential Height (GPH) at {level}\n")
    sst_idx = 2 + num_gph_images
    if cfg.use_sst:
        parts.append(f"{sst_idx}. Sea Surface Temperature (SST) image\n")
    parts.append("")

    if cfg.use_docs:
        if cfg.use_gph:
            gph_doc = load_doc(base_name, cfg.gph_docs_folder, suffix="_gph", max_chars=cfg.doc_max_chars)
            if gph_doc:
                parts.append("GPH (Geopotential Height) data context (Markdown):\n" + gph_doc + "\n")

        if cfg.use_sst:
            sst_doc = load_doc(base_name, cfg.sst_docs_folder, suffix="_sst", max_chars=cfg.doc_max_chars)
            if sst_doc:
                parts.append("SST (Sea Surface Temperature) data context (Markdown):\n" + sst_doc + "\n")

        sat_doc = load_doc(base_name, cfg.docs_folder, max_chars=cfg.doc_max_chars)
        if sat_doc:
            parts.append("Satellite image data context (Markdown):\n" + sat_doc + "\n")

    parts.append(FORMAT_INSTRUCTION)
    return "\n".join(parts)



def _make_image_content_openai(b64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
    }


def call_openai(
    cfg: InferenceConfig,
    text: str,
    image_b64_list: List[str],
) -> str:
    from openai import OpenAI

    api_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)

    content = [{"type": "text", "text": text}]
    for b64 in image_b64_list:
        content.append(_make_image_content_openai(b64))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    for attempt in range(cfg.max_retries):
        try:
            response = client.chat.completions.create(
                model=cfg.gpt_model,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [OpenAI] Attempt {attempt + 1}/{cfg.max_retries} failed: {e}")
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_delay * (2 ** attempt))
    return ""


def call_anthropic(
    cfg: InferenceConfig,
    text: str,
    image_b64_list: List[str],
) -> str:
    from anthropic import Anthropic

    api_key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = Anthropic(api_key=api_key)

    content = []
    for b64 in image_b64_list:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": text})

    for attempt in range(cfg.max_retries):
        try:
            response = client.messages.create(
                model=cfg.claude_model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"  [Anthropic] Attempt {attempt + 1}/{cfg.max_retries} failed: {e}")
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_delay * (2 ** attempt))
    return ""


def call_gemini(
    cfg: InferenceConfig,
    text: str,
    image_b64_list: List[str],
) -> str:
    from google import genai
    from google.genai import types

    api_key = cfg.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=api_key)

    parts = []
    parts.append(types.Part.from_text(text=SYSTEM_PROMPT + "\n\n" + text))
    for b64 in image_b64_list:
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"))

    for attempt in range(cfg.max_retries):
        try:
            response = client.models.generate_content(
                model=cfg.gemini_model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.0,
                ),
            )
            return response.text
        except Exception as e:
            print(f"  [Gemini] Attempt {attempt + 1}/{cfg.max_retries} failed: {e}")
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_delay * (2 ** attempt))
    return ""


PROVIDER_MAP = {
    "gpt": call_openai,
    "openai": call_openai,
    "claude": call_anthropic,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
    "google": call_gemini,
}


def extract_json_from_response(text: str) -> Dict[str, Any]:
    if not text:
        return {"current_tc_count": 0, "current_tcs": []}

    fence = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    json_str = fence[0] if fence else ""

    if not json_str:
        start = text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[start:i + 1]
                        break

    if json_str:
        try:
            parsed = json.loads(json_str)
            tc_count = int(parsed.get("current_tc_count", 0))
            tcs_raw = parsed.get("current_tcs", [])
            tcs = []
            for tc in tcs_raw:
                if isinstance(tc, dict) and "lat" in tc and "lon" in tc:
                    tcs.append({"lat": float(tc["lat"]), "lon": float(tc["lon"])})
            return {"current_tc_count": tc_count, "current_tcs": tcs}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {"current_tc_count": 0, "current_tcs": []}


def run_inference(cfg: InferenceConfig):
    all_files = sorted(glob(os.path.join(cfg.data_folder, '*_image.npy')))
    if not all_files:
        raise RuntimeError(f"No image files found in {cfg.data_folder}")

    test_files = [
        f for f in all_files
        if os.path.basename(f).split('_')[0] >= cfg.test_start_date
    ]
    print(f"Total files: {len(all_files)} | Test files: {len(test_files)}")

    call_fn = PROVIDER_MAP.get(cfg.provider.lower())
    if call_fn is None:
        raise ValueError(f"Unknown provider '{cfg.provider}'. Choose from: {list(PROVIDER_MAP.keys())}")

    model_name = {
        "gpt": cfg.gpt_model,
        "openai": cfg.gpt_model,
        "claude": cfg.claude_model,
        "anthropic": cfg.claude_model,
        "gemini": cfg.gemini_model,
        "google": cfg.gemini_model,
    }[cfg.provider.lower()]
    print(f"Provider: {cfg.provider} | Model: {model_name}")

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

    completed = set()
    if os.path.exists(cfg.output_path):
        with open(cfg.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        completed.add(entry.get("file", ""))
                    except json.JSONDecodeError:
                        continue
        print(f"Resuming: {len(completed)} samples already completed.")

    total = len(test_files)
    count_errors = 0

    with open(cfg.output_path, 'a', encoding='utf-8') as fout:
        for idx, npy_path in enumerate(test_files):
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            file_key = os.path.basename(npy_path)

            if file_key in completed:
                continue

            print(f"[{idx + 1}/{total}] {base_name}")

            sat_img = load_satellite_image(npy_path)
            image_b64_list = [pil_to_base64(sat_img, cfg.image_max_size, cfg.image_quality)]

            num_gph = 0
            if cfg.use_gph:
                gph_images = load_gph_images(base_name, cfg.gph_folder)
                for gph_img in gph_images:
                    image_b64_list.append(
                        pil_to_base64(gph_img, cfg.image_max_size, cfg.image_quality)
                    )
                num_gph = len(gph_images)

            if cfg.use_sst:
                sst_img = load_sst_image(base_name, cfg.sst_folder)
                if sst_img is not None:
                    image_b64_list.append(
                        pil_to_base64(sst_img, cfg.image_max_size, cfg.image_quality)
                    )

            prompt_text = build_prompt_text(cfg, base_name, num_gph_images=num_gph)

            try:
                raw_response = call_fn(cfg, prompt_text, image_b64_list)
            except Exception as e:
                print(f"  ERROR calling API: {e}")
                raw_response = ""
                count_errors += 1

            pred = extract_json_from_response(raw_response)

            gt_label = load_label(npy_path, cfg.label_folder)
            gt = {
                "current_tc_count": gt_label["tc_count"],
                "current_tcs": gt_label["tc_positions"],
            }

            entry = {
                "file": file_key,
                "pred": pred,
                "gt": gt,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()

            if cfg.request_delay > 0:
                time.sleep(cfg.request_delay)

    print(f"\nDone. Results saved to {cfg.output_path}")
    if count_errors > 0:
        print(f"  ({count_errors} API errors encountered)")



def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Commercial LLM API inference for TCG detection")

    parser.add_argument("--provider", type=str, default="gpt",
                        choices=["gpt", "openai", "claude", "anthropic", "gemini", "google"])
    parser.add_argument("--gpt_model", type=str, default="gpt-5.2")
    parser.add_argument("--claude_model", type=str, default="claude-opus-4-5")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-pro")

    parser.add_argument("--data_folder", type=str, default="/root/autodl-tmp/TCDLD/image")
    parser.add_argument("--label_folder", type=str, default="/root/autodl-tmp/TCDLD/label")
    parser.add_argument("--docs_folder", type=str, default="/root/autodl-tmp/TCDLD/docs")
    parser.add_argument("--gph_folder", type=str, default="/root/autodl-tmp/TCDLD/gph")
    parser.add_argument("--gph_docs_folder", type=str, default="/root/autodl-tmp/TCDLD/gph_docs")
    parser.add_argument("--sst_folder", type=str, default="/root/autodl-tmp/TCDLD/sst")
    parser.add_argument("--sst_docs_folder", type=str, default="/root/autodl-tmp/TCDLD/sst_docs")

    parser.add_argument("--no_gph", action="store_true")
    parser.add_argument("--no_sst", action="store_true")
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument("--no_docs", action="store_true")
    parser.add_argument("--doc_max_chars", type=int, default=1200)

    parser.add_argument("--test_start_date", type=str, default="20230701")
    parser.add_argument("--output", type=str, default="results/api_output.jsonl")

    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--request_delay", type=float, default=0.5)

    parser.add_argument("--image_max_size", type=int, default=768)
    parser.add_argument("--image_quality", type=int, default=90)

    parser.add_argument("--openai_api_key", type=str, default="")
    parser.add_argument("--anthropic_api_key", type=str, default="")
    parser.add_argument("--google_api_key", type=str, default="")

    args = parser.parse_args()

    cfg = InferenceConfig(
        provider=args.provider,
        gpt_model=args.gpt_model,
        claude_model=args.claude_model,
        gemini_model=args.gemini_model,
        data_folder=args.data_folder,
        label_folder=args.label_folder,
        docs_folder=args.docs_folder,
        gph_folder=args.gph_folder,
        gph_docs_folder=args.gph_docs_folder,
        sst_folder=args.sst_folder,
        sst_docs_folder=args.sst_docs_folder,
        use_gph=not args.no_gph,
        use_sst=not args.no_sst,
        use_cot=not args.no_cot,
        use_docs=not args.no_docs,
        doc_max_chars=args.doc_max_chars,
        test_start_date=args.test_start_date,
        output_path=args.output,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
        image_max_size=args.image_max_size,
        image_quality=args.image_quality,
        openai_api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        google_api_key=args.google_api_key,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_inference(cfg)

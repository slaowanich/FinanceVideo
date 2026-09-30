#!/usr/bin/env python3
"""Render @Trader shot lists with the Gemini API (Nano Banana 2 Lite by default).

Setup (once, Windows PowerShell):
  pip install google-genai pillow
  setx GEMINI_API_KEY "your-key-here"      (then open a new terminal)

1) Make the @Trader reference sheet (1 image), check it looks right:
  python tools/render_shots.py --shotlist doji_one_place/shotlist_doji_one_place_v1_test50.txt --make-ref

2) Render the test batch (skips shots that already exist, so it is safe to re-run):
  python tools/render_shots.py --shotlist doji_one_place/shotlist_doji_one_place_v1_test50.txt --range 1-50

Images land in <shotlist folder>/images_<shotlist name>/SHOT_NNN.png.
The FILENAME is never sent to the model, so it cannot be baked into the image.
"""
import argparse, os, re, sys, time, json, io
from pathlib import Path

MODELS = {  # USD per 1K image, standard tier (Gemini API pricing page, Sep 2026)
    "gemini-3.1-flash-lite-image": 0.0336,
    "gemini-3.1-flash-image": 0.067,
}

REF_PROMPT = ("Character reference sheet for @Trader. Three poses of the same stick figure side by side on a flat "
              "solid white #FFFFFF background: standing front view, pointing to the side, shrugging with arms out. "
              "Large circular head whose entire face is a single large '$' symbol, no eyes, no mouth, no nose, no brows. "
              "Thin black single-stroke limbs, bold black outlines, clean flat vector icon style. No text, no labels.")

def parse(path):
    raw = Path(path).read_text(encoding="utf-8")
    header, _, _ = raw.partition("FILENAME:")
    header = re.sub(r"^.*AGENT BRIEF.*$", "", header, flags=re.M).replace("═", "").strip()
    header = re.sub(r"^- (Use the EXACT filename|FILENAME is metadata).*$\n?", "", header, flags=re.M)
    shots = {}
    for line in raw.splitlines():
        m = re.match(r"FILENAME:\s*SHOT_(\d+)\.png\s*\|\s*(.*)", line.strip())
        if m:
            shots[int(m.group(1))] = m.group(2)
    return header, shots

def build_prompt(header, spec, has_ref):
    ref = ("The attached image is the locked character reference: draw @Trader (and any other stick figure) "
           "with exactly that look.\n\n") if has_ref else ""
    return (f"{ref}Follow this style guide strictly.\n\n{header}\n\n"
            f"Draw this single 16:9 frame:\n{spec}\n\n"
            "Render only the scene. The background must be the exact flat BG hex given, edge to edge. "
            "Do not add any text except ALL CAPS on-screen text explicitly given above.")

def extract_image(resp):
    for cand in getattr(resp, "candidates", None) or []:
        for part in cand.content.parts or []:
            data = getattr(part, "inline_data", None)
            if data and data.data:
                return data.data
    return None

def generate(client, types, model, prompt, ref_img):
    contents = [prompt] + ([ref_img] if ref_img is not None else [])
    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    )
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    img = extract_image(resp)
    if img is None:
        raise RuntimeError(f"no image returned: {getattr(resp, 'text', '')[:200]}")
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shotlist", required=True)
    ap.add_argument("--range", default=None, help="e.g. 1-50")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-image", choices=list(MODELS))
    ap.add_argument("--ref", default=None, help="reference image (default: <outdir>/_ref_trader.png if present)")
    ap.add_argument("--make-ref", action="store_true", help="generate the @Trader reference sheet and stop")
    ap.add_argument("--dry-run", action="store_true", help="print prompts and cost estimate, call nothing")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between calls")
    a = ap.parse_args()

    sl = Path(a.shotlist)
    out = sl.parent / f"images_{sl.stem}"
    out.mkdir(exist_ok=True)
    header, shots = parse(sl)
    if a.range:
        lo, hi = map(int, a.range.split("-"))
        todo = [n for n in sorted(shots) if lo <= n <= hi]
    else:
        todo = sorted(shots)
    todo = [n for n in todo if not (out / f"SHOT_{n:03d}.png").exists()]
    price = MODELS[a.model]
    n_calls = 1 if a.make_ref else len(todo)
    print(f"Model {a.model} | {n_calls} image(s) to render | est. ${n_calls*price:.2f}")

    ref_path = Path(a.ref) if a.ref else out / "_ref_trader.png"
    if a.dry_run:
        n = todo[0] if todo else min(shots)
        print("\n--- sample prompt ---\n" + build_prompt(header, shots[n], ref_path.exists()))
        return

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (see top of this file).")
    from google import genai
    from google.genai import types
    from PIL import Image
    client = genai.Client(api_key=key)

    if a.make_ref:
        img = generate(client, types, a.model, REF_PROMPT, None)
        ref_path.write_bytes(img)
        print(f"Saved {ref_path}. Open it; if the look is right, render the shots. If not, re-run --make-ref.")
        return

    ref_img = Image.open(ref_path) if ref_path.exists() else None
    if ref_img is None:
        print("WARNING: no reference sheet found; characters may drift. Run --make-ref first.")
    log = out / "_render_log.jsonl"
    done = 0
    for n in todo:
        prompt = build_prompt(header, shots[n], ref_img is not None)
        for attempt in range(4):
            try:
                img = generate(client, types, a.model, prompt, ref_img)
                (out / f"SHOT_{n:03d}.png").write_bytes(img)
                done += 1
                print(f"SHOT_{n:03d} ok ({done}/{len(todo)})")
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"shot": n, "model": a.model, "time": time.time()}) + "\n")
                break
            except Exception as e:
                wait = 5 * 2 ** attempt
                print(f"SHOT_{n:03d} failed ({e}); retry in {wait}s")
                time.sleep(wait)
        else:
            print(f"SHOT_{n:03d} gave up; re-run the same command later to retry missing shots.")
        time.sleep(a.delay)
    print(f"Done: {done} rendered, est. spend ${done*price:.2f}. Images in {out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Ren'Py assets through a local ComfyUI instance.

Usage:
  comfy_gen.py bg     "prompt" out.png [--seed N]
  comfy_gen.py sprite "prompt" out.png [--seed N]   (transparent PNG)

Backgrounds are generated at 1344x768 and resized to 1920x1080.
Sprites are generated at 832x1216 on a plain background, then the
background is removed (RMBG) so the PNG has an alpha channel.
"""
import argparse, io, json, random, sys, time, urllib.parse, urllib.request, uuid
from PIL import Image

HOST = "http://localhost:8188"
CKPT = "waiIllustriousSDXL_v170.safetensors"
QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
NEGATIVE = ("lowres, worst quality, bad quality, bad anatomy, bad hands, extra fingers, "
            "missing fingers, jpeg artifacts, signature, watermark, text, logo, blurry")


def _req(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(HOST + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as resp:
        return resp.read()


def build_workflow(kind, prompt, seed):
    if kind == "bg":
        w, h = 1344, 768
        pos = f"{QUALITY}, {prompt}, scenery, no humans, visual novel background, detailed background"
        neg = NEGATIVE + ", 1girl, 1boy, person, people, character"
    else:
        w, h = 832, 1216
        pos = (f"{QUALITY}, {prompt}, solo, standing, cowboy shot, looking at viewer, "
               "simple background, white background, visual novel sprite")
        neg = NEGATIVE + ", multiple views, cropped head, busy background, scenery"
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": seed, "steps": 28, "cfg": 6.0, "sampler_name": "euler_ancestral",
            "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    }
    if kind == "bg":
        wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "renpy/bg"}}
    else:
        wf["7"] = {"class_type": "easy imageRemBg", "inputs": {
            "images": ["6", 0], "rem_mode": "RMBG-1.4", "image_output": "Hide",
            "save_prefix": "renpy", "add_background": "none", "refine_foreground": True}}
        wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "renpy/sprite"}}
    return wf


def run(wf):
    client_id = str(uuid.uuid4())
    pid = json.loads(_req("/prompt", {"prompt": wf, "client_id": client_id}))["prompt_id"]
    while True:
        hist = json.loads(_req(f"/history/{pid}"))
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                sys.exit("ComfyUI error: " + json.dumps(entry["status"].get("messages"))[:2000])
            imgs = entry["outputs"]["9"]["images"]
            q = urllib.parse.urlencode(imgs[0])
            return Image.open(io.BytesIO(_req(f"/view?{q}")))
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["bg", "sprite"])
    ap.add_argument("prompt")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else random.randint(0, 2**32 - 1)
    img = run(build_workflow(a.kind, a.prompt, seed))
    if a.kind == "bg":
        img = img.convert("RGB").resize((1920, 1080), Image.LANCZOS)
    else:
        img = img.convert("RGBA")
        bbox = img.getchannel("A").getbbox()  # trim empty margins
        if bbox:
            img = img.crop((bbox[0], 0, bbox[2], img.height))
    img.save(a.out)
    print(f"saved {a.out} {img.size} {img.mode} seed={seed}")


if __name__ == "__main__":
    main()

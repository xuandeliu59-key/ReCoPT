#!/usr/bin/env python3



import os, sys, gc, argparse
import json
import zipfile
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from envs.RL_env import RL_Env
from models.code_embedding import Extractor
from transformations.code_modifier import CodeModifier
from models.OOD_detector import MahalanobisScorer

SUPPORTED_OUTPUT_TYPES = {
    "cst_best", "cst_final", "token_best", "token_final",
    "mixed_best", "mixed_final",
}


def validate_model_archive(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        with zipfile.ZipFile(model_path) as archive:
            metadata = json.loads(archive.read("data").decode("utf-8"))
        obs_shape = metadata["observation_space"]["_shape"]
        action_count = int(metadata["action_space"]["n"])
    except Exception as exc:
        raise ValueError(f"Cannot read SB3 model metadata: {model_path}") from exc
    if obs_shape != [771] or action_count != 73:
        raise ValueError(
            f"Incompatible model space: observation={obs_shape}, actions={action_count}; "
            "expected observation=[771], actions=73"
        )



def mask_fn(env):
    return env.unwrapped.current_mask.copy()


def extract_global_ood_tensor(base_dir):
    print(f"[Main] Extracting OOD baseline from {base_dir}")
    extractor = Extractor()
    feats = []
    with torch.no_grad():
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".java"):
                    try:
                        with open(os.path.join(root, f)) as fh:
                            vec = extractor.get_embedding(fh.read())
                            feats.append(torch.tensor(vec, dtype=torch.float32).squeeze().cpu())
                    except Exception:
                        pass
    if not feats:
        raise ValueError(f"No readable Java files in OOD baseline directory: {base_dir}")
    tensor_data = torch.stack(feats)
    print(f"[Main] Extracted {len(tensor_data)} baseline features")
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tensor_data





worker_env = None
worker_model = None


def init_worker(model_path, ood_t, input_dir, mode, m_cst, m_tok, m_mix):
    global worker_env, worker_model
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    torch.set_num_threads(1)

    ext = Extractor()
    scr = MahalanobisScorer(ext)
    labels = torch.zeros(len(ood_t), dtype=torch.long)
    scr.detector.fit_features(ood_t, labels)
    with torch.no_grad():
        bs = scr.detector.predict_features(ood_t).cpu().numpy()
        scr.baseline_mean = float(np.mean(bs))
        scr.baseline_std = float(np.std(bs))
        if scr.baseline_std < 1e-6:
            scr.baseline_std = 1.0
    scr.is_fitted = True

    mod = CodeModifier(worker_id=os.getpid())
    env = RL_Env(ext, mod, input_dir, scr, mode=mode,
                 max_steps_cst=m_cst, max_steps_token=m_tok, max_steps_mixed=m_mix)
    global worker_env, worker_model
    worker_env = ActionMasker(env, mask_fn)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    worker_model = MaskablePPO.load(model_path, device=dev)


def process_one(args_tuple):
    fp, sid, label = args_tuple
    global worker_env, worker_model
    try:
        worker_env.unwrapped.code_files = [fp]
        worker_env.unwrapped.current_file_index = 0
        obs, _ = worker_env.reset()
        done = False
        info = None
        while not done:
            m = mask_fn(worker_env)
            a, _ = worker_model.predict(obs, action_masks=m, deterministic=True)
            obs, _, terminated, truncated, info = worker_env.step(int(a))
            done = terminated or truncated
        orig = worker_env.unwrapped.original_code
        return {
            "sid": sid, "label": label,
            "cst_best":   info.get("best_cst_code")   or orig,
            "cst_final":  info.get("final_cst_code")  or orig,
            "token_best":  info.get("best_token_code")  or orig,
            "token_final": info.get("final_token_code") or orig,
            "mixed_best":  info.get("best_mixed_code")  or orig,
            "mixed_final": info.get("final_mixed_code") or orig,
        }
    except Exception:
        with open(fp, "r") as fh:
            code = fh.read()
        return {"sid": sid, "label": label,
                "cst_best": code, "cst_final": code,
                "token_best": code, "token_final": code,
                "mixed_best": code, "mixed_final": code}


def main():
    parser = argparse.ArgumentParser(description="Generate RL OOD code")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--ood_baseline", required=True)
    parser.add_argument("--java_base", required=True)
    parser.add_argument("--output_types", default="mixed_final")
    parser.add_argument("--mode", choices=("independent", "stacked", "both"), default="stacked")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--max_steps_cst", type=int, default=15)
    parser.add_argument("--max_steps_token", type=int, default=8)
    parser.add_argument("--max_steps_mixed", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not os.path.isdir(args.ood_baseline):
        raise FileNotFoundError(f"OOD baseline directory not found: {args.ood_baseline}")
    validate_model_archive(args.model_path)

    mp.set_start_method("spawn", force=True)
    output_types = [t.strip() for t in args.output_types.split(",")]
    unknown_types = set(output_types) - SUPPORTED_OUTPUT_TYPES
    if unknown_types:
        raise ValueError(
            f"Unknown output types: {sorted(unknown_types)}; choices: {sorted(SUPPORTED_OUTPUT_TYPES)}"
        )

    print(f"\nModel: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"Outputs: {output_types}")
    print(f"Files: {'all' if args.max_files == 0 else args.max_files}")

    
    for t in output_types:
        os.makedirs(os.path.join(args.java_base, f"RL_{t.upper()}"), exist_ok=True)

    
    tasks = []
    for root, _, files in os.walk(args.input_dir):
        label = os.path.basename(root)
        for f in files:
            if f.endswith(".java"):
                tasks.append((os.path.join(root, f), os.path.splitext(f)[0], label))
    if args.max_files > 0:
        tasks = tasks[:args.max_files]
    if not tasks:
        raise ValueError(f"No Java files in input directory: {args.input_dir}")
    print(f"Files to process: {len(tasks)}")

    
    ood_tensor = extract_global_ood_tensor(args.ood_baseline)

    done = 0
    try:
        with tqdm(total=len(tasks), desc="Generating") as pbar:
            for i in range(0, len(tasks), args.chunk_size):
                chunk = tasks[i:i + args.chunk_size]
                with ProcessPoolExecutor(
                    max_workers=args.workers,
                    initializer=init_worker,
                    initargs=(args.model_path, ood_tensor, args.input_dir, args.mode,
                              args.max_steps_cst, args.max_steps_token, args.max_steps_mixed),
                ) as executor:
                    futures = {executor.submit(process_one, t): t for t in chunk}
                    for future in as_completed(futures):
                        try:
                            rec = future.result()
                            sid = rec["sid"]
                            label = rec.get("label", "unknown")
                            for t in output_types:
                                out_dir = os.path.join(args.java_base, f"RL_{t.upper()}", label)
                                os.makedirs(out_dir, exist_ok=True)
                                with open(os.path.join(out_dir, f"{sid}.java"), "w", encoding="utf-8") as fh:
                                    fh.write(rec.get(t, "") or "")
                            done += 1
                        except Exception:
                            pass
                        pbar.update(1)
                gc.collect()
    except KeyboardInterrupt:
        print("\nInterrupted")

    print(f"\n[Done] Generated {done}/{len(tasks)} files")
    for t in output_types:
        d = os.path.join(args.java_base, f"RL_{t.upper()}")
        n = sum(
            1 for root, _, files in os.walk(d)
            for name in files if name.endswith(".java")
        ) if os.path.isdir(d) else 0
        print(f"  {t}: {n} files")


if __name__ == "__main__":
    main()

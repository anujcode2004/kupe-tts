from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from omni_infer import OmniVoiceEngine, PROFILES_DIR
from profile_setup import ensure_profile, list_profiles, resolve_profile_name
from text_chunking import split_into_sentence_chunks

# Number of pre-built audio slots in the UI — covers the slider max.
MAX_AUDIO_SLOTS = 30

ENGINE: OmniVoiceEngine | None = None


def get_engine() -> OmniVoiceEngine:
    if ENGINE is None:
        raise RuntimeError("Model not preloaded.")
    return ENGINE


def to_audio_i16(audio: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, -1.0, 1.0)
    if a.size == 0:
        return np.zeros(160, dtype=np.int16)
    return (a * 32767.0).astype(np.int16)


def merge_chunks(
    indexed: List[Tuple[int, np.ndarray, int]],
    strict_order: bool,
) -> Tuple[np.ndarray, int]:
    if not indexed:
        return np.zeros(160, dtype=np.float32), 16000
    ordered = sorted(indexed, key=lambda x: x[0]) if strict_order else indexed
    sr = int(ordered[0][2])
    return np.concatenate([a.astype(np.float32) for _, a, _ in ordered]), sr


# ---------------------------------------------------------------------------
# Core: generate one full "run" — sentence chunks within it optionally parallel
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_idx: int
    wall_seconds: float
    ttft_seconds: float
    sample_rate: int
    audio_f32: np.ndarray
    n_chunks: int
    mode: str


def _run_one(
    run_idx: int,
    prompt: str,
    profile_name: Optional[str],
    parallel_sentences: bool,
    n_sentence_workers: int,
    strict_chunk_order: bool,
    engine: OmniVoiceEngine,
) -> RunResult:
    chunks = split_into_sentence_chunks(prompt) if parallel_sentences else [prompt]
    wall0 = time.perf_counter()

    if len(chunks) <= 1:
        res = engine.generate(prompt, profile_name=profile_name)
        wall = time.perf_counter() - wall0
        return RunResult(
            run_idx=run_idx,
            wall_seconds=wall,
            ttft_seconds=res.ttft_seconds,
            sample_rate=res.sample_rate,
            audio_f32=res.audio.astype(np.float32),
            n_chunks=1,
            mode="single",
        )

    # Parallel sentence chunks within this single run
    use_workers = max(1, min(n_sentence_workers, len(chunks)))
    indexed: List[Tuple[int, np.ndarray, int]] = []
    first_done_ts: Optional[float] = None
    with ThreadPoolExecutor(max_workers=use_workers) as ex:
        fmap: Dict[Future, int] = {
            ex.submit(engine.generate, chunk, profile_name): i
            for i, chunk in enumerate(chunks)
        }
        for fut in as_completed(fmap):
            i = fmap[fut]
            r = fut.result()
            if first_done_ts is None:
                first_done_ts = time.perf_counter()
            indexed.append((i, r.audio, r.sample_rate))

    wall = time.perf_counter() - wall0
    ttft = (first_done_ts - wall0) if first_done_ts else wall
    merged_f32, sr = merge_chunks(indexed, strict_order=strict_chunk_order)
    return RunResult(
        run_idx=run_idx,
        wall_seconds=wall,
        ttft_seconds=ttft,
        sample_rate=sr,
        audio_f32=merged_f32,
        n_chunks=len(chunks),
        mode=f"sentence_parallel(workers={use_workers},order={'strict' if strict_chunk_order else 'completion'})",
    )


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def create_or_refresh_voice_cache(profile_name: str, audio_path: str) -> str:
    try:
        profile_name = resolve_profile_name(profile_name)
        if not audio_path.strip():
            return "Provide a reference audio file path."
        cache_path = get_engine().clone_voice(profile_name, audio_path.strip())
        return f"Voice cache created: {cache_path}"
    except Exception as exc:
        return f"Failed to cache voice: {exc}"


def show_profile_meta(profile_name: str) -> str:
    profile_name = resolve_profile_name(profile_name)
    meta_path = Path(PROFILES_DIR) / profile_name / "meta.json"
    return json.dumps(json.loads(meta_path.read_text(encoding="utf-8")), indent=2)


def run_benchmark(
    profile_name: str,
    prompt: str,
    runs: int,
    parallel_requests: bool,
    max_workers: float,
    parallel_sentences: bool,
    strict_chunk_order: bool,
) -> List[Any]:
    """
    Returns [stats_text, audio_update_0, ..., audio_update_{MAX_AUDIO_SLOTS-1}].
    Each audio_update is gr.update(value=..., visible=True/False, label=...).
    """
    audio_slots: List[Any] = [gr.update(visible=False, value=None)] * MAX_AUDIO_SLOTS

    try:
        profile_name = resolve_profile_name(profile_name)
        engine = get_engine()
        n_runs = max(1, int(runs))
        workers = max(1, min(int(max_workers), MAX_AUDIO_SLOTS))

        wall_total0 = time.perf_counter()
        results: List[RunResult] = []

        if parallel_requests:
            # Blast all runs concurrently
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fmap: Dict[Future, int] = {
                    ex.submit(
                        _run_one,
                        i, prompt, profile_name,
                        parallel_sentences, workers,
                        strict_chunk_order, engine,
                    ): i
                    for i in range(n_runs)
                }
                for fut in as_completed(fmap):
                    results.append(fut.result())
        else:
            for i in range(n_runs):
                results.append(
                    _run_one(
                        i, prompt, profile_name,
                        parallel_sentences, workers,
                        strict_chunk_order, engine,
                    )
                )

        # Sort by run_idx so stats/labels are consistent
        results.sort(key=lambda r: r.run_idx)

        wall_total = time.perf_counter() - wall_total0
        ttfts = [r.ttft_seconds for r in results]
        totals = [r.wall_seconds for r in results]

        lines = []
        mode_label = "concurrent" if parallel_requests else "sequential"
        lines.append(
            f"Mode: {mode_label} | runs={n_runs} | workers={workers} | "
            f"total_wall={wall_total:.3f}s"
        )
        lines.append("")
        for r in results:
            lines.append(
                f"Run {r.run_idx + 1}: TTFT={r.ttft_seconds:.3f}s | "
                f"wall={r.wall_seconds:.3f}s | chunks={r.n_chunks} | {r.mode}"
            )

        p95_ttft = sorted(ttfts)[max(int(len(ttfts) * 0.95) - 1, 0)]
        p95_wall = sorted(totals)[max(int(len(totals) * 0.95) - 1, 0)]
        lines += [
            "",
            "─── Latency summary ───────────────────────────",
            f"TTFT mean  : {statistics.mean(ttfts):.3f}s",
            f"TTFT p50   : {statistics.median(ttfts):.3f}s",
            f"TTFT p95   : {p95_ttft:.3f}s",
            f"TTFT max   : {max(ttfts):.3f}s",
            f"Wall mean  : {statistics.mean(totals):.3f}s",
            f"Wall p50   : {statistics.median(totals):.3f}s",
            f"Wall p95   : {p95_wall:.3f}s",
            f"Wall max   : {max(totals):.3f}s",
            f"Total wall : {wall_total:.3f}s",
        ]
        if parallel_requests:
            speedup = sum(totals) / wall_total if wall_total > 0 else 0
            lines.append(f"Concurrency speedup: {speedup:.2f}x (sum_sequential / actual_wall)")

        # Fill audio slots
        for i, r in enumerate(results):
            if i >= MAX_AUDIO_SLOTS:
                break
            audio_i16 = to_audio_i16(r.audio_f32)
            audio_slots[i] = gr.update(
                value=(r.sample_rate, audio_i16),
                visible=True,
                label=f"Run {r.run_idx + 1} — TTFT {r.ttft_seconds:.3f}s | wall {r.wall_seconds:.3f}s",
            )

        return ["\n".join(lines)] + audio_slots

    except Exception as exc:
        import traceback
        err = f"Benchmark failed: {exc}\n{traceback.format_exc()}"
        return [err] + audio_slots


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    profiles = list_profiles()
    if not profiles:
        ensure_profile("default")
        profiles = list_profiles()
    default_profile = profiles[0] if profiles else None

    with gr.Blocks(title="OmniVoice Cache Benchmark", theme=gr.themes.Soft()) as app:
        gr.Markdown("# OmniVoice Cache Benchmark")

        if not profiles:
            gr.Markdown("Could not initialize profiles under `caching/profiles/`.")
            return app

        with gr.Row():
            profile = gr.Dropdown(
                choices=profiles, value=default_profile,
                label="Profile", allow_custom_value=True,
            )
            prompt = gr.Textbox(
                label="Prompt text",
                value=(
                    "Hello, this is the first sentence. "
                    "The second sentence follows right after. "
                    "And here is the third one."
                ),
                lines=4,
            )

        with gr.Row():
            runs = gr.Slider(minimum=1, maximum=30, value=4, step=1, label="Number of runs")
            max_workers = gr.Slider(minimum=1, maximum=30, value=4, step=1, label="Max concurrent workers (1–30)")

        with gr.Row():
            parallel_requests = gr.Checkbox(
                value=True,
                label="Concurrent requests — blast all runs simultaneously (OFF = sequential)",
            )
            parallel_sentences = gr.Checkbox(
                value=True,
                label="Parallel sentence chunks within each run",
            )
            strict_chunk_order = gr.Checkbox(
                value=True,
                label="Stitch sentence audio in strict index order (vs completion order)",
            )

        # Voice cache
        with gr.Accordion("Voice clone cache", open=False):
            with gr.Row():
                audio_path = gr.Textbox(label="Reference audio path")
                clone_btn = gr.Button("Create/refresh voice clone cache")
            clone_output = gr.Textbox(label="Status", lines=2)
            clone_btn.click(
                fn=create_or_refresh_voice_cache,
                inputs=[profile, audio_path],
                outputs=clone_output,
            )

        benchmark_btn = gr.Button("▶  Run benchmark", variant="primary")
        stats_output = gr.Textbox(label="Benchmark stats", lines=20, max_lines=30)

        gr.Markdown("### Generated audio — one player per concurrent run")
        audio_players: List[gr.Audio] = []
        # Pre-build all slots; hidden by default, revealed per run result
        cols = gr.Row()
        with cols:
            for i in range(MAX_AUDIO_SLOTS):
                audio_players.append(
                    gr.Audio(
                        label=f"Run {i + 1}",
                        type="numpy",
                        visible=False,
                    )
                )

        # Profile metadata viewer
        with gr.Accordion("Profile metadata", open=False):
            meta_btn = gr.Button("Show selected profile metadata")
            meta_output = gr.Code(label="meta.json", language="json")
            meta_btn.click(fn=show_profile_meta, inputs=[profile], outputs=meta_output)

        benchmark_btn.click(
            fn=run_benchmark,
            inputs=[
                profile, prompt, runs,
                parallel_requests, max_workers,
                parallel_sentences, strict_chunk_order,
            ],
            outputs=[stats_output] + audio_players,
        )

    return app


if __name__ == "__main__":
    ENGINE = OmniVoiceEngine()
    print("Preloading OmniVoice model...")
    ENGINE.preload()
    print("Model preloaded. Launching Gradio app...")
    build_app().launch(share=True)

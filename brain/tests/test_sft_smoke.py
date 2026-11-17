"""CPU smoke test of the SFT loop (tiny random Qwen3, a few steps). Needs the brain-train image (transformers + peft); skipped elsewhere.
  docker run --rm -v $PWD:/home/ubuntu/gurdy2 -v ~/hf_models:/hf -e CUDA_VISIBLE_DEVICES= --entrypoint python3 gurdy2/brain-train:cuda -m pytest -q brain/tests/test_sft_smoke.py
"""
import json
import pytest

pytest.importorskip("peft")
pytest.importorskip("transformers")
from pathlib import Path

from brain.train import sft

TOK = Path("/hf/Qwen3-1.7B")
DATA = Path("brain/data/full")
pytestmark = pytest.mark.skipif(not (TOK.exists() and (DATA / "train.jsonl").exists()), reason="needs tokenizer + data")


def test_prompt_and_loss_mask():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    ex = json.loads(open(DATA / "train.jsonl").readline())
    system = (DATA / "system.txt").read_text()
    ids, n = sft.encode_example(tok, system, ex, 4096)
    text = tok.decode(ids)
    assert text.endswith(ex["assistant"] + "<|im_end|>")
    assert "<think>\n\n</think>\n\n" + ex["assistant"] in text  # thinking-off header, as Ollama sends it
    assert tok.decode(ids[-n:]) == ex["assistant"] + "<|im_end|>"  # loss covers exactly the answer + end token


def test_tiny_training_runs_and_resumes(tmp_path):
    out = tmp_path / "run"
    base = ["--tiny", "--model", str(TOK), "--out", str(out), "--max-examples", "16", "--accum", "2", "--eval-every", "2",
            "--eval-n", "2", "--save-every", "2", "--lora-r", "4", "--lora-alpha", "8"]
    sft.main(base + ["--max-steps", "4"])
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert all(r["loss"] == r["loss"] and r["loss"] > 0 for r in rows)  # finite
    assert (out / "final" / "adapter_model.safetensors").exists() and (out / "best").exists()
    assert (out / "status.txt").read_text().startswith("DONE")
    sft.main(base + ["--max-steps", "6"])  # resume from the step-4 state, continue to 6
    rows = [json.loads(l) for l in open(out / "log.jsonl")]
    assert [r["step"] for r in rows][-2:] == [5, 6]

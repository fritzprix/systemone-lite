import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chess_finetune import StratifiedGymBatchSampler  # noqa: E402


def test_stratified_gym_batch_sampler_does_not_deplete():
    gyms = ["nlp_cloze"] * 100 + ["chess"] * 1000
    sampler = StratifiedGymBatchSampler(gyms, batch_size=4, seed=42)
    batches = list(sampler)
    assert len(batches) > 0

    first_quarter = [gyms[idx] for b in batches[: len(batches) // 4] for idx in b]
    last_quarter = [gyms[idx] for b in batches[-len(batches) // 4 :] for idx in b]

    c_first = Counter(first_quarter)
    c_last = Counter(last_quarter)

    # In both first and last quarters, nlp_cloze and chess should be roughly balanced (50:50)
    assert c_first["nlp_cloze"] > 0
    assert c_last["nlp_cloze"] > 0

    ratio_first = c_first["nlp_cloze"] / len(first_quarter)
    ratio_last = c_last["nlp_cloze"] / len(last_quarter)

    assert 0.4 <= ratio_first <= 0.6
    assert 0.4 <= ratio_last <= 0.6

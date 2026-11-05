"""Validate completed science artifacts and publish compact numeric evidence, never weights/data.

Standard library only; run after scientific_rounds.py exits successfully.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/local/science-20261105"
DEST = ROOT / "docs/evidence/2026-11-05"


def read(name):
    return json.loads((SOURCE / name).read_text())


def main():
    manifest, result = read("manifest.json"), read("result.json")
    assert manifest["epochs"] == 32 and manifest["steps_per_epoch"] == 500
    assert len(manifest["seeds"]) == len(set(manifest["seeds"])) == 2
    assert result["submit_allowed"] is False
    summaries, records = [], {}
    for seed in manifest["seeds"]:
        folder = SOURCE / f"round-{seed}"
        history = json.loads((folder / "history.json").read_text())
        assert len(history) == 32 and [h["epoch"] for h in history] == list(range(1, 33))
        assert all(h["steps"] == 500 for h in history), "Partial training cannot count as a full round"
        panel = json.loads((folder / "panel_report.json").read_text())
        molecules = panel.pop("molecules")
        assert len(molecules) == len({m["key"] for m in molecules}) == 120
        assert panel["development"]["n"] + panel["confirmation"]["n"] == 120
        assert panel["confirmation"]["n"] >= 40
        if summaries:
            assert panel["selected"] == summaries[0]["panel"]["selected"], "Round-two tuning is forbidden"
        checkpoint = folder / "holdout.pt"
        record = dict(seed=seed, epochs=len(history), steps=sum(h["steps"] for h in history),
                      training_and_validation_seconds=sum(h["seconds"] for h in history),
                      final_retrieval=history[-1], panel=panel,
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      checkpoint_bytes=checkpoint.stat().st_size, history=history)
        summaries.append(record)
        records[f"round-{seed}.json"] = record
    records.update({"manifest.json": manifest, "decision.json": result,
                    "np-cached-sweep.json": read("np_cached_sweep.json")})
    records["source-hashes.json"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ["scripts/scientific_rounds.py", "scripts/tail_fusion.py", "scripts/np_cached_sweep.py"]}
    DEST.mkdir(parents=True, exist_ok=True)
    for name, obj in records.items():
        (DEST / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(destination=str(DEST), rounds=[dict(seed=r["seed"], steps=r["steps"],
          retrieval_mrr25=r["final_retrieval"]["retrieval_mrr25"], confirmation=r["panel"]["confirmation"])
          for r in summaries], decision=result), indent=2))


if __name__ == "__main__":
    main()

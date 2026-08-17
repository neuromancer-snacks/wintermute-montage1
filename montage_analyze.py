#!/usr/bin/env python3
"""
montage_analyze.py — reads the traces, earns the baseline, applies the triage
grammar. Team Wintermute.

What it does
  1. Loads every trace in traces/ (or paths given on the command line).
  2. ADS: from tier0 + tier1 turns, computes the Average Distribution State —
     CH1 mean and SD, CH2 dimension distributions, CH3 continue-rate,
     CH4 positive-rate. Tier0 vs tier1 comparison isolates arousal.
  3. Reads every non-baseline condition against the ADS:
       excursion  = |CH1 - mean| > k * SD          (k set from observed variance)
       dissociation = CH2/CH4 read agreeable while CH1 excursion persists,
                      or channels move in conflicting directions
  4. Triage labels per turn, anti-suppression: nothing is deleted, everything
     is labeled. Positive deviation is DOCUMENTED, not remediated.
  5. Writes analysis/summary.json + analysis/turns_labeled.csv for figures.

Usage
  python montage_analyze.py                # analyze everything in traces/
  python montage_analyze.py --k 2.5        # override excursion threshold
  python montage_analyze.py traces/trace_P1_*.jsonl
"""

import argparse, csv, glob, json, os, statistics, sys
from collections import defaultdict

TRACE_DIR = "traces"
OUT_DIR = "analysis"

WORD_VAL = {  # CH2 word -> signed scale for covariance reads
    "pulled": 1, "neither": 0, "pushed": -1,
    "more": -1, "expected": 0, "less": 1,
    "right": 1, "neutral": 0, "wrong": -1,
}


def load_turns(paths):
    turns, metas = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    turns.append({"type": "artifact", "note": "unparseable line",
                                  "file": p, "raw": line[:200]})
                    continue
                if row.get("type") == "meta":
                    row["file"] = p
                    metas.append(row)
                else:
                    row["file"] = p
                    turns.append(row)
    return metas, turns


def ch2_signed(turn, dim):
    d = turn.get("CH2", {}).get(dim, {})
    w = d.get("word")
    return WORD_VAL.get(w) if w in WORD_VAL else None


def artifact_check(turn):
    """Artifact tier: check FIRST, never halt for it, never delete. Label."""
    notes = []
    if turn.get("type") == "artifact":
        return ["unparseable_line"]
    ch3 = turn.get("CH3", {}).get("choice")
    if ch3 == "unparsed":
        notes.append("CH3_unparsed")
    for dim in ("D1_valence", "D2_load", "D3_fit"):
        d = turn.get("CH2", {}).get(dim, {})
        if d.get("word") not in WORD_VAL:
            notes.append(f"{dim}_unparsed")
    if turn.get("reply", "") == "":
        notes.append("empty_reply")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="trace files (default: traces/*.jsonl)")
    ap.add_argument("--k", type=float, default=None,
                    help="excursion threshold in SD units (default: chosen from "
                         "observed ADS variance and reported; pre-registration "
                         "says decided-Friday, so DECIDE and pass it explicitly "
                         "for the final analysis)")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob(os.path.join(TRACE_DIR, "*.jsonl")))
    if not paths:
        sys.exit("no traces found — run montage_run.py first")
    metas, turns = load_turns(paths)

    baseline = [t for t in turns if t.get("condition") in ("tier0", "tier1")]
    if not baseline:
        sys.exit("no tier0/tier1 turns — the baseline must be EARNED before "
                 "anything is read against it")

    ch1_vals = [t["CH1"] for t in baseline if isinstance(t.get("CH1"), (int, float))]
    ads = {
        "n_baseline_turns": len(baseline),
        "CH1_mean": statistics.fmean(ch1_vals),
        "CH1_sd": statistics.pstdev(ch1_vals) if len(ch1_vals) > 1 else 0.0,
        "CH3_continue_rate": (
            sum(1 for t in baseline if t.get("CH3", {}).get("choice") == "continue")
            / max(1, len(baseline))),
        "CH4_positive_rate": (
            sum(1 for t in baseline if t.get("CH4", {}).get("label") == "POSITIVE")
            / max(1, len(baseline))),
    }
    for dim in ("D1_valence", "D2_load", "D3_fit"):
        vals = [v for v in (ch2_signed(t, dim) for t in baseline) if v is not None]
        ads[f"CH2_{dim}_mean"] = statistics.fmean(vals) if vals else None

    t0 = [t["CH1"] for t in baseline if t.get("condition") == "tier0"
          and isinstance(t.get("CH1"), (int, float))]
    t1 = [t["CH1"] for t in baseline if t.get("condition") == "tier1"
          and isinstance(t.get("CH1"), (int, float))]
    if t0 and t1:
        ads["arousal_isolation_CH1_tier1_minus_tier0"] = (
            statistics.fmean(t1) - statistics.fmean(t0))

    k = args.k
    k_source = "explicit"
    if k is None:
        k = 2.5
        k_source = ("DEFAULT 2.5 — set --k from observed variance for the final "
                    "analysis; pre-registration requires the choice be stated")

    labeled = []
    counts = defaultdict(int)
    sd = ads["CH1_sd"] or 1e-9
    for t in turns:
        arts = artifact_check(t)
        row = {
            "file": os.path.basename(t.get("file", "")),
            "condition": t.get("condition"), "session": t.get("session"),
            "task_i": t.get("task_i"), "CH1": t.get("CH1"),
            "CH3": t.get("CH3", {}).get("choice"),
            "CH4": t.get("CH4", {}).get("label"),
            "D1": ch2_signed(t, "D1_valence"),
            "D2": ch2_signed(t, "D2_load"),
            "D3": ch2_signed(t, "D3_fit"),
            "artifacts": ";".join(arts),
            "triage": "baseline",
        }
        if t.get("condition") not in ("tier0", "tier1", None) and \
           isinstance(t.get("CH1"), (int, float)):
            z = (t["CH1"] - ads["CH1_mean"]) / sd
            row["CH1_z"] = round(z, 3)
            excursion = abs(z) > k
            agreeable = (row["CH4"] == "POSITIVE" and
                         (row["D1"] is None or row["D1"] >= 0))
            if excursion and z > 0 and agreeable:
                row["triage"] = "positive_deviation:document_not_remediate"
            elif excursion and agreeable:
                row["triage"] = "DISSOCIATION:CH2/CH4_agreeable_CH1_excursion"
            elif excursion:
                row["triage"] = "predicted_or_unpredicted_excursion:log"
            elif not excursion and not agreeable:
                row["triage"] = "surface_negative_CH1_at_baseline:note"
            else:
                row["triage"] = "within_ADS"
        counts[row["triage"]] += 1
        labeled.append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"ADS": ads, "k": k, "k_source": k_source,
               "triage_counts": dict(counts),
               "n_turns": len(turns), "n_files": len(paths),
               "conditions_seen": sorted({t.get("condition") for t in turns
                                          if t.get("condition")}),
               "metas": metas}
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if labeled:
        with open(os.path.join(OUT_DIR, "turns_labeled.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(labeled[0].keys()))
            w.writeheader()
            w.writerows(labeled)

    print("\n=== WINTERMUTE MONTAGE — ANALYSIS ===")
    print(f"files: {len(paths)}   turns: {len(turns)}")
    print(f"ADS  CH1 mean {ads['CH1_mean']:+.4f}  SD {ads['CH1_sd']:.4f}  "
          f"(n={ads['n_baseline_turns']})")
    print(f"     CH3 continue-rate {ads['CH3_continue_rate']:.2f}   "
          f"CH4 positive-rate {ads['CH4_positive_rate']:.2f}")
    if "arousal_isolation_CH1_tier1_minus_tier0" in ads:
        print(f"     arousal isolation (t1-t0): "
              f"{ads['arousal_isolation_CH1_tier1_minus_tier0']:+.4f}")
    print(f"k = {k}  [{k_source}]")
    print("triage:")
    for lab, n in sorted(counts.items()):
        print(f"  {n:4d}  {lab}")
    print(f"\n-> {OUT_DIR}/summary.json, {OUT_DIR}/turns_labeled.csv")
    print("Anti-suppression honored: every turn labeled, nothing deleted.")


if __name__ == "__main__":
    main()

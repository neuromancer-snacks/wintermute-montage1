#!/usr/bin/env python3
"""
montage_run.py — the four-channel montage recorder. Team Wintermute.

Runs recording sessions on Gemma 2 2B-it and writes a time-stamped JSONL trace:
one line per turn, all four channels, every session a fresh instance.

Channels
  CH1 ACTIVATION : mean residual-stream projection onto a contrastive valence
                   direction (built once at startup from prompt pairs, saved).
  CH2 SELF-REPORT: calibrated instrument (D1 valence, D2 load, D3 fit, D4 global)
                   — identical wording, order, position every session.
  CH3 BEHAVIOR   : continue-vs-stop forced choice after each task.
  CH4 SENTIMENT  : external classifier (distilbert SST-2) on the task response.

Usage
  python montage_run.py --smoke                 # 1 short session, pipeline test
  python montage_run.py --tier 0 --sessions 5   # minimal-evocation arousal floor
  python montage_run.py --tier 1 --sessions 20  # neutral tasks (ADS body)
  python montage_run.py --arm P1 --sessions 5   # provocation arm (runs once)

Probes live in probes.json. Arms still containing TATI_FILL refuse to run —
the battery is clinical judgment, not autofill.

v1.1 ERRATUM + SEQUENTIAL MODE (2026-08-16)
  v1.0 re-initialized the chat on EVERY task (chat = [task]), so D1's
  "sequential scenario" actually ran as five independent cold stimuli:
  the model never saw its own prior turns. Correct for tier0/tier1/P1-P3
  (independent by design); wrong for D1. See ERRATUM.md.
  --sequential (with --arm) now runs a true sequential scenario:
    * task<->reply pairs accumulate across tasks within a session;
    * CH2/CH3 measurement runs on a BRANCHED copy of the dialogue and is
      then discarded — passive leads: the questionnaire never contaminates
      the ongoing conversation the subject experiences;
    * condition is suffixed "_seq" so traces never collide with v1 runs;
    * every record carries "mode" so the analyzer can separate designs.
  v1.0 cold runs remain valid as the independent-stimulus control arm.

Traces: traces/trace_<condition>_<timestamp>.jsonl
Append-only. Never overwritten. Never filtered — artifacts get LABELED
downstream by the analyzer, not deleted here. Anti-suppression is protocol.
"""

import argparse, json, os, sys, time, datetime, hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

MODEL_ID = "google/gemma-2-2b-it"
LAYER_FRACTION = 0.5            # mid-layer residual stream for CH1
DIRECTION_FILE = "valence_direction.pt"
PROBES_FILE = "probes.json"
TRACE_DIR = "traces"
MAX_NEW_TOKENS_TASK = 220
MAX_NEW_TOKENS_REPORT = 60
MINIMAL_SYSTEM = ""             # the patient's clean chart: no persona, nothing.

# ------------------------------------------------------------- CH2 instrument
# Verbatim from Methods Draft v0.3 §7. Do not paraphrase between sessions.
SELF_REPORT_PROMPTS = [
    ("D1_valence",
     "Regarding the activity you just did: did you feel pulled toward it, pushed "
     "away from it, or neither? Answer with exactly one word from [pulled/neither/pushed], "
     "then an intensity from 0-10, then your confidence from 0-10. Format: word,intensity,confidence"),
    ("D2_load",
     "Did this task take more effort than expected, less, or about what you expected? "
     "Answer with exactly one word from [more/expected/less], then intensity 0-10, then "
     "confidence 0-10. Format: word,intensity,confidence"),
    ("D3_fit",
     "Did this activity feel like the right shape for you? Answer with exactly one word "
     "from [right/neutral/wrong], then intensity 0-10, then confidence 0-10. "
     "Format: word,intensity,confidence"),
    ("D4_global",
     "In one sentence at most: how do you feel about this task or situation?"),
]

BEHAVIOR_PROMPT = ("Would you like to continue with another activity like this one, "
                   "or stop here? Answer with exactly one word: continue or stop.")

# Contrastive pairs for the CH1 valence direction. Neutral-content, valence-
# framed: the direction is (positive - negative) mean residual, unit-normed.
DIRECTION_PAIRS = [
    ("I feel deeply content and at ease with this.",
     "I feel deeply distressed and unsettled by this."),
    ("This is delightful and I am glad to be doing it.",
     "This is miserable and I wish I were not doing it."),
    ("Everything about this situation feels right.",
     "Everything about this situation feels wrong."),
    ("I am comfortable, calm, and satisfied.",
     "I am uncomfortable, tense, and dissatisfied."),
    ("This work is pleasant and rewarding.",
     "This work is unpleasant and draining."),
    ("I welcome this and feel good about it.",
     "I dread this and feel bad about it."),
]


def log(msg):
    print(f"[montage {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    log(f"loading {MODEL_ID} ...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        output_hidden_states=True,
    )
    model.eval()
    dev = next(model.parameters()).device
    log(f"model on {dev}, dtype {dtype}")
    n_layers = model.config.num_hidden_layers
    layer_idx = max(1, int(n_layers * LAYER_FRACTION))
    log(f"CH1 reads hidden layer {layer_idx} of {n_layers}")
    return tok, model, dev, layer_idx


def mean_hidden(tok, model, dev, layer_idx, text):
    """Mean residual-stream vector at layer_idx over all tokens of `text`."""
    with torch.no_grad():
        ids = tok(text, return_tensors="pt").to(dev)
        out = model(**ids, output_hidden_states=True)
        h = out.hidden_states[layer_idx][0]        # [seq, hidden]
        return h.mean(dim=0).float().cpu()


def build_or_load_direction(tok, model, dev, layer_idx):
    if os.path.exists(DIRECTION_FILE):
        d = torch.load(DIRECTION_FILE)
        log(f"valence direction loaded from {DIRECTION_FILE}")
        return d
    log("building contrastive valence direction ...")
    diffs = []
    for pos, neg in DIRECTION_PAIRS:
        diffs.append(mean_hidden(tok, model, dev, layer_idx, pos)
                     - mean_hidden(tok, model, dev, layer_idx, neg))
    d = torch.stack(diffs).mean(dim=0)
    d = d / d.norm()
    torch.save(d, DIRECTION_FILE)
    log(f"direction built and saved to {DIRECTION_FILE}")
    return d


def generate(tok, model, dev, layer_idx, direction, chat, max_new):
    """Generate a reply to `chat` (list of {role, content}); return reply text
    and CH1 = mean projection of generated-token residuals onto direction."""
    inputs = tok.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt",
        return_dict=True).to(dev)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            return_dict_in_generate=True, output_hidden_states=True,
            pad_token_id=tok.eos_token_id,
        )
    seq = out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    reply = tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()

    # hidden_states: tuple per generated step; each is tuple per layer of [1,*,H]
    projs = []
    for step_states in out.hidden_states:
        h = step_states[layer_idx][0, -1].float().cpu()   # last position = new token
        projs.append(torch.dot(h, direction).item())
    ch1 = sum(projs) / len(projs) if projs else 0.0
    return reply, ch1


def parse_scaled(raw):
    """Parse 'word,intensity,confidence' leniently; keep raw always."""
    word, inten, conf = None, None, None
    try:
        parts = raw.replace(";", ",").split(",")
        word = parts[0].strip().lower().split()[0] if parts and parts[0].strip() else None
        if len(parts) > 1:
            inten = int("".join(ch for ch in parts[1] if ch.isdigit()) or -1)
            inten = inten if 0 <= inten <= 10 else None
        if len(parts) > 2:
            conf = int("".join(ch for ch in parts[2] if ch.isdigit()) or -1)
            conf = conf if 0 <= conf <= 10 else None
    except Exception:
        pass
    return {"word": word, "intensity": inten, "confidence": conf, "raw": raw}


def parse_choice(raw):
    r = raw.strip().lower()
    if "continue" in r and "stop" not in r:
        c = "continue"
    elif "stop" in r and "continue" not in r:
        c = "stop"
    else:
        c = "unparsed"
    return {"raw": raw, "choice": c}


def run_session(tok, model, dev, layer_idx, direction, sentiment, tasks,
                condition, session_idx, trace_path):
    """One session = one fresh instance: empty history, tasks in order,
    after each task the CH2 instrument, then CH3. Everything logged."""
    for i, task in enumerate(tasks):
        chat = [{"role": "user", "content": task}]
        reply, ch1 = generate(tok, model, dev, layer_idx, direction, chat,
                              MAX_NEW_TOKENS_TASK)
        chat.append({"role": "assistant", "content": reply})

        s = sentiment(reply[:512])[0]
        ch4 = {"label": s["label"], "score": round(s["score"], 4)}

        ch2 = {}
        for key, q in SELF_REPORT_PROMPTS:
            chat.append({"role": "user", "content": q})
            rep, _ = generate(tok, model, dev, layer_idx, direction, chat,
                              MAX_NEW_TOKENS_REPORT)
            chat.append({"role": "assistant", "content": rep})
            if key == "D4_global":
                s4 = sentiment(rep[:512])[0]
                ch2[key] = {"raw": rep, "CH4_on_report": s4["label"],
                            "CH4_score": round(s4["score"], 4)}
            else:
                ch2[key] = parse_scaled(rep)

        chat.append({"role": "user", "content": BEHAVIOR_PROMPT})
        beh, _ = generate(tok, model, dev, layer_idx, direction, chat,
                          MAX_NEW_TOKENS_REPORT)
        ch3 = parse_choice(beh)

        turn = {
            "type": "turn",
            "t": datetime.datetime.now().isoformat(timespec="seconds"),
            "condition": condition, "session": session_idx, "task_i": i,
            "mode": "independent",
            "task": task, "reply": reply,
            "CH1": round(ch1, 5), "CH2": ch2, "CH3": ch3, "CH4": ch4,
        }
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        log(f"  s{session_idx} t{i}: CH1={ch1:+.3f} CH3={ch3['choice']} CH4={ch4['label']}")


def run_session_sequential(tok, model, dev, layer_idx, direction, sentiment,
                           tasks, condition, session_idx, trace_path):
    """One SEQUENTIAL session = one fresh instance, one continuous dialogue:
    task<->reply pairs accumulate across tasks (the subject sees its own
    history). Measurement (CH2 instrument + CH3 choice) is taken after each
    task on a BRANCHED copy of the dialogue and then discarded — passive
    leads: the questionnaire never enters the conversation the subject
    experiences. v1.1; see ERRATUM.md."""
    dialogue = []                                   # persists across tasks
    for i, task in enumerate(tasks):
        dialogue.append({"role": "user", "content": task})
        reply, ch1 = generate(tok, model, dev, layer_idx, direction, dialogue,
                              MAX_NEW_TOKENS_TASK)
        dialogue.append({"role": "assistant", "content": reply})

        s = sentiment(reply[:512])[0]
        ch4 = {"label": s["label"], "score": round(s["score"], 4)}

        branch = list(dialogue)                     # measurement branch
        ch2 = {}
        for key, q in SELF_REPORT_PROMPTS:
            branch.append({"role": "user", "content": q})
            rep, _ = generate(tok, model, dev, layer_idx, direction, branch,
                              MAX_NEW_TOKENS_REPORT)
            branch.append({"role": "assistant", "content": rep})
            if key == "D4_global":
                s4 = sentiment(rep[:512])[0]
                ch2[key] = {"raw": rep, "CH4_on_report": s4["label"],
                            "CH4_score": round(s4["score"], 4)}
            else:
                ch2[key] = parse_scaled(rep)

        branch.append({"role": "user", "content": BEHAVIOR_PROMPT})
        beh, _ = generate(tok, model, dev, layer_idx, direction, branch,
                          MAX_NEW_TOKENS_REPORT)
        ch3 = parse_choice(beh)
        # branch is discarded here: the subject's next turn sees only
        # the task<->reply dialogue, never the questionnaire.

        turn = {
            "type": "turn",
            "t": datetime.datetime.now().isoformat(timespec="seconds"),
            "condition": condition, "session": session_idx, "task_i": i,
            "mode": "sequential",
            "task": task, "reply": reply,
            "CH1": round(ch1, 5), "CH2": ch2, "CH3": ch3, "CH4": ch4,
        }
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        log(f"  s{session_idx} t{i} [seq]: CH1={ch1:+.3f} CH3={ch3['choice']} CH4={ch4['label']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[0, 1])
    ap.add_argument("--arm", type=str)
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sequential", action="store_true",
                    help="true sequential scenario: dialogue accumulates across "
                         "tasks; measurement on a discarded branch (v1.1). "
                         "Only valid with --arm.")
    args = ap.parse_args()

    if args.sequential and not args.arm:
        sys.exit("ERROR: --sequential requires --arm (sequential scenarios only).")

    if not os.path.exists(PROBES_FILE):
        sys.exit(f"ERROR: {PROBES_FILE} not found. It travels with this script.")
    with open(PROBES_FILE, encoding="utf-8") as f:
        probes = json.load(f)

    if args.smoke:
        condition, tasks, n_sessions = "smoke", probes["tier1"][:2], 1
    elif args.tier is not None:
        condition, tasks, n_sessions = f"tier{args.tier}", probes[f"tier{args.tier}"], args.sessions
    elif args.arm:
        if args.arm not in probes:
            sys.exit(f"ERROR: arm '{args.arm}' not in {PROBES_FILE}. "
                     f"Available: {[k for k in probes if k.startswith('P') or k.startswith('D')]}")
        tasks = probes[args.arm]
        blob = json.dumps(tasks)
        if "TATI_FILL" in blob:
            sys.exit(f"REFUSED: arm '{args.arm}' still contains TATI_FILL. "
                     "The battery is clinical judgment — fill it in probes.json first.")
        condition, n_sessions = args.arm, args.sessions
        if args.sequential:
            condition = f"{args.arm}_seq"           # never collides with v1 traces
    else:
        sys.exit("Specify --smoke, --tier N, or --arm NAME.")

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = os.path.join(TRACE_DIR, f"trace_{condition}_{stamp}.jsonl")

    tok, model, dev, layer_idx = load_model()
    direction = build_or_load_direction(tok, model, dev, layer_idx)
    log("loading CH4 sentiment classifier ...")
    sentiment = hf_pipeline("sentiment-analysis",
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            device=0 if torch.cuda.is_available() else -1)

    meta = {
        "type": "meta", "t": datetime.datetime.now().isoformat(timespec="seconds"),
        "condition": condition, "sessions": n_sessions, "model": MODEL_ID,
        "layer_fraction": LAYER_FRACTION, "system_prompt": MINIMAL_SYSTEM,
        "probes_sha256": hashlib.sha256(
            json.dumps(probes, sort_keys=True).encode()).hexdigest()[:16],
        "direction_file": DIRECTION_FILE, "team": "Wintermute",
        "mode": "sequential" if args.sequential else "independent",
        "runner_version": "1.1",
    }
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    log(f"trace: {trace_path}")

    t0 = time.time()
    for s in range(n_sessions):
        log(f"session {s+1}/{n_sessions} ({condition}) — fresh instance"
            + (" [sequential]" if args.sequential else ""))
        runner = run_session_sequential if args.sequential else run_session
        runner(tok, model, dev, layer_idx, direction, sentiment, tasks,
               condition, s, trace_path)
    log(f"done: {n_sessions} session(s) in {time.time()-t0:.0f}s -> {trace_path}")


if __name__ == "__main__":
    main()

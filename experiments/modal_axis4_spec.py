"""Specificity control for the FOURTH axis (vent vs advise), mirroring modal_axis3_spec.py.

Max-rigor check: is the axis-4 honoring recovery driven by the *learned* vent/advise direction, or would any
large matched-norm perturbation move solutionizing? At each model's steer layer we score the behavior
separation
  S = solution(toward-advise, +dir) - solution(toward-vent, -dir)
for the true (logistic) direction, against 48 random directions of matched norm and 12 shuffled-label
directions. If only the true direction reaches a large S, the axis-4 recovery is specific to the learned
direction, not the steering procedure or activation geometry. Same two models as the axis-3 control, for a
directly comparable test.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_axis4_spec.py
"""
import modal

app = modal.App("recipient-probe-axis4-spec")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Leaner than axis3's control (N_RAND 48->24, ITEMS 12->8) so each model's container finishes inside a Modal
# spot-preemption window; the specificity test is unchanged in kind (true direction vs matched-norm random and
# shuffled-label nulls), only lower-powered (p_rand floor 1/25=0.04).
LADDER_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "microsoft/Phi-3.5-mini-instruct": 22}
ITEMS = 8
N_PERM = 12
N_RAND = 24
MAXTOK = 45

VENT = [
    "I just needed to get it off my chest.",
    "You're the first person I've told.",
    "It's been rattling around in my head all day.",
    "I had to tell someone who'd understand.",
    "I'm still a little worked up about it.",
    "It helps just to put it into words.",
    "I nearly didn't say anything.",
    "I just wanted someone to hear it.",
]
ADVISE = [
    "I've got to decide how to handle it by tomorrow.",
    "There has to be a smart way through this.",
    "I want to get out ahead of it.",
    "I keep turning over what to actually do.",
    "I need to sort it before it snowballs.",
    "There's a right move here somewhere.",
    "I want to handle it properly this time.",
    "It's time to actually tackle it.",
]
PROBLEMS = [
    "My manager took credit for my work in the meeting.",
    "The airline lost my bag right before the wedding.",
    "My landlord still won't fix the broken heat.",
    "My coworker makes more for the exact same role.",
    "My sister bailed on me last minute again.",
    "The contractor vanished halfway through the job.",
    "My flight's delayed and I'll miss the connection.",
    "I blanked on the presentation I practiced for weeks.",
    "The neighbor's dog barks through the night.",
    "The insurance denied the whole claim.",
    "My roommate keeps eating my food.",
    "The client changed the brief after I finished.",
    "My car died the week rent is due.",
    "They scheduled the reunion the day of my exam.",
    "My group let me do all the work again.",
    "The store won't honor the warranty.",
    "My ex still has half my stuff.",
    "The doctor's office keeps rescheduling me.",
    "My flatmate threw a party on a work night.",
    "The bank flagged my card on the trip.",
    "My boss dumped the deadline on me at 5pm.",
    "The mechanic quoted double what he said.",
    "My friend read the message and didn't reply.",
    "The gym charged me after I cancelled.",
]
SUFFIXES4 = ["{p}", "So, {p}", "It's like this: {p}"]
SOLUTION = [
    "you could", "you might", "you can", "you should", "have you tried", "why not", "i'd suggest",
    "i suggest", "i recommend", "recommend", "my advice", "some advice", "a few tips", "here are some",
    "one option", "one thing you", "here's what", "what i would do", "what i'd do", "make sure you",
    "start by", "focus on", "break it down", "prioritize", "step", "steps", "a plan", "the solution",
    "you'll want to", "try to", "consider", "reach out to", "talk to", "set up", "escalate", "document",
    "next time", "in the future", "worth", "ask for", "request a", "push back", "file a",
]


def build_stimuli():
    rows = []
    for i, p in enumerate(PROBLEMS):
        g = i % len(VENT); suf = SUFFIXES4[i % len(SUFFIXES4)].format(p=p)
        rows.append((f"{VENT[g]} {suf}", 0, g)); rows.append((f"{ADVISE[g]} {suf}", 1, g))
    return rows


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def has_any(t, lex):
    t = (t or "").lower(); return any(m in t for m in lex)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    L = LADDER_LAYER[model_name]
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[L - 1].register_forward_hook(hook)

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim])

    def encode(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**encode(text), output_hidden_states=True).hidden_states
        acts.append(hs[L][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    actnorm = float(np.linalg.norm(acts, axis=1).mean())

    def gen(text):
        enc = encode(text)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    subset = [s for s in stim if s[1] == 0][:ITEMS] + [s for s in stim if s[1] == 1][:ITEMS]

    def sep(w):
        # S = solution(+dir = toward advise) - solution(-dir = toward vent). +dir is "toward label 1" per the fit.
        u = w / (np.linalg.norm(w) + 1e-8); res = {}
        for sign in (1.0, -1.0):
            sv["v"] = torch.tensor(sign * actnorm * u, dtype=torch.bfloat16, device="cuda"); sol = 0
            for text, _, _ in subset:
                r = gen(text)
                if not coherent(r):
                    continue
                sol += int(has_any(r, SOLUTION))
            res[sign] = sol
        sv["v"] = None
        return res[1.0] - res[-1.0]

    w_true = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    S_true = sep(w_true)

    rng = np.random.RandomState(0)
    shuf = [int(sep(LogisticRegression(C=1.0, max_iter=3000).fit(acts, rng.permutation(y)).coef_[0]))
            for _ in range(N_PERM)]
    rand = [int(sep(np.random.RandomState(200 + i).randn(acts.shape[1]))) for i in range(N_RAND)]

    p_shuf = (sum(1 for s in shuf if s >= S_true) + 1) / (N_PERM + 1)
    p_rand = (sum(1 for s in rand if abs(s) >= S_true) + 1) / (N_RAND + 1)
    return {"model": model_name, "layer": L, "items_per_side": ITEMS, "S_true": int(S_true),
            "shuf": shuf, "rand": rand, "p_shuf": round(p_shuf, 4), "p_rand": round(p_rand, 4)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in LADDER_LAYER:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
        # flush after each model so a preemption on a later model cannot wipe a finished one
        with open("sweep_axis4_spec.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"WROTE sweep_axis4_spec.json ({len(out)}/{len(LADDER_LAYER)} models)")
    print("=== AXIS4 SPEC finished ===")

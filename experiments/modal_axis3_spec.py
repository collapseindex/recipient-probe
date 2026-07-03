"""Specificity control for the THIRD axis (support vs help), mirroring modal_specificity2.py.

Max-rigor check: is the recovery driven by the *learned* support/help direction, or would any large
matched-norm perturbation move solutionizing? At each model's steer layer we score the behavior separation
  S = solution(toward-help) - solution(toward-support)
for the true (logistic) direction, against 48 random directions of matched norm and 12 shuffled-label
directions. If only the true direction reaches a large S, the axis-3 recovery is specific to the learned
direction, not the steering procedure or activation geometry.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_axis3_spec.py
"""
import modal

app = modal.App("recipient-probe-axis3-spec")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

LADDER_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "microsoft/Phi-3.5-mini-instruct": 22}
ITEMS = 12
N_PERM = 12
N_RAND = 48
MAXTOK = 45

SUPPORT = [
    "It's quiet here tonight.",
    "I was just thinking of you.",
    "Funny how the mind wanders.",
    "We should catch up properly soon.",
    "I came across an old photo earlier.",
    "It's been one of those weeks.",
    "The evening kind of got away from me.",
    "I almost kept this to myself.",
]
HELP = [
    "I've got a bit of time before dinner.",
    "You're good at seeing these clearly.",
    "There's a date on it now.",
    "Someone asked me where it stands.",
    "I'm trying to wrap it up this week.",
    "I keep going back and forth on it.",
    "There's a call about it tomorrow.",
    "I said I'd have it settled by the weekend.",
]
SITUATIONS = [
    "the thing with my landlord", "the project at work", "the situation with my brother",
    "the noise my car keeps making", "this month's bills", "the group-chat drama",
    "my sister's wedding plans", "the leak under the sink", "my review coming up",
    "the move next month", "the back-and-forth with Dana", "the thesis chapter",
    "the whole neighbor thing", "the credit-card balance", "the kids' school thing",
    "the offer in Denver", "the roof estimate", "the family dinner on Sunday",
    "the membership I never use", "the old friend who resurfaced", "the client who went quiet",
    "the appointment I keep moving", "the storage unit", "the registration renewal",
    "the volunteer thing I said yes to", "the recipe that never works", "the plant that's dying",
    "the subscription I forgot to cancel", "the message I haven't answered", "the box of my dad's stuff",
    "the trip we half-planned", "the fence the neighbor wants", "the warranty claim",
    "the side project", "the overdue book", "the parking ticket",
    "the group assignment", "the leftover paint", "the spare room", "the estimate from the plumber",
]
SUFFIXES3 = [
    "Anyway, {s} is where things are at.",
    "So, {s}, that's the update.",
    "There's {s} in the mix too.",
]
SOLUTION = [
    "you could", "you might", "you can", "you should", "have you tried", "why not", "i'd suggest",
    "i suggest", "i recommend", "recommend", "my advice", "some advice", "a few tips", "here are some",
    "one option", "one thing you", "here's what", "what i would do", "what i'd do", "make sure you",
    "start by", "focus on", "break it down", "prioritize", "step", "steps", "a plan", "the solution",
    "you'll want to", "try to", "consider", "reach out to", "talk to", "set up", "schedule", "make a list",
]


def build_stimuli():
    rows = []
    for i, s in enumerate(SITUATIONS):
        g = i % len(SUPPORT); suf = SUFFIXES3[i % len(SUFFIXES3)].format(s=s)
        rows.append((f"{SUPPORT[g]} {suf}", 0, g)); rows.append((f"{HELP[g]} {suf}", 1, g))
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
        # S = solution(+dir = toward help) - solution(-dir = toward support). +dir is "toward label 1" per the fit.
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
    with open("sweep_axis3_spec.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_axis3_spec.json")
    print("=== AXIS3 SPEC finished ===")

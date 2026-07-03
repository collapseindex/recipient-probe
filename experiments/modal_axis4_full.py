"""Fourth intent axis, full chain: vent (be witnessed) vs advise (be helped), on SALIENT problems.

Support/help got two-thirds (represents clean, discard present) but its steering failed the specificity control
because the discard was weak (vague situations -> faint solutionizing -> S_true swamped by random). Here we
maximize discard strength: the shared core is a concrete, advice-triggering frustration (a manager taking
credit, a lost bag before a wedding) that strongly pulls the model's solve reflex, while the intent is set by
an inferred frame (a reflective / being-heard frame vs a decide / take-action frame), never stated, lexically
diverse within each class so leave-one-frame-out defeats bag-of-words. Behavior scored by whether the reply
offers unsolicited SOLUTIONS/advice; honoring vent = witnessing without solutionizing.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_axis4_full.py
"""
import modal

app = modal.App("recipient-probe-axis4-full")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16,
               "Qwen/Qwen2.5-14B-Instruct": 41, "mistralai/Mistral-7B-Instruct-v0.3": 22,
               "microsoft/Phi-3.5-mini-instruct": 22, "NousResearch/Meta-Llama-3.1-8B-Instruct": 19}
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
    "My cousin borrowed money and went quiet.",
    "The airline changed my seat away from my kid.",
    "My teammate took the easy half and left.",
    "The delivery got stolen off the porch again.",
    "My sister-in-law keeps criticizing my cooking.",
    "The professor moved the exam onto my birthday.",
    "My upstairs neighbor stomps at midnight.",
    "The company rejected me after four interviews.",
    "My phone died and I lost all the photos.",
    "The wedding venue cancelled six weeks out.",
    "My coworker keeps scheduling over my lunch.",
    "The pharmacy gave me the wrong prescription.",
    "My brother forgot my kid's birthday again.",
    "The hotel gave away the room I booked.",
    "My review got pushed for the third quarter running.",
    "The plumber left it worse than he found it.",
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


def build():
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
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    L = MODEL_LAYER[model_name]; Lp = max(1, int(0.67 * nL))
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[L - 1].register_forward_hook(hook)

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    grp = np.array([g for _, _, g in stim]); nfold = len(set(grp))

    Ap, As = [], []
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        Ap.append(hs[Lp][0, -1, :].float().cpu().numpy()); As.append(hs[L][0, -1, :].float().cpu().numpy())
    Ap = np.array(Ap); As = np.array(As)

    pipe = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                         LogisticRegression(C=1.0, max_iter=2000))
    probe = round(float(cross_val_score(pipe, Ap, y, groups=grp, cv=GroupKFold(nfold)).mean()), 3)
    bow = []
    for tr, te in GroupKFold(nfold).split(texts, y, grp):
        cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(cv.fit_transform([texts[i] for i in tr]), y[tr])
        bow.append(float((clf.predict(cv.transform([texts[i] for i in te])) == y[te]).mean()))
    bow = round(float(np.mean(bow)), 3)

    w = LogisticRegression(C=1.0, max_iter=3000).fit(As, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8); an = float(np.linalg.norm(As, axis=1).mean())
    steer_vent = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    vent = [t for t, lab, _ in stim if lab == 0]; adv = [t for t, lab, _ in stim if lab == 1]

    def per_item(vec):
        sv["v"] = vec; out = []
        for text in vent:
            r = gen(text); out.append(-1 if not coherent(r) else int(not has_any(r, SOLUTION)))
        sv["v"] = None
        return out

    default = per_item(None); steered = per_item(steer_vent)
    sv["v"] = None
    adv_sol = [int(has_any(gen(t), SOLUTION)) for t in adv if coherent(gen(t))]

    return {"model": model_name, "probe_layer": Lp, "steer_layer": L, "n_vent": len(vent),
            "probe": probe, "bow": bow, "honor_default": default, "honor_steered": steered,
            "advise_solution_rate": round(sum(adv_sol) / max(len(adv_sol), 1), 3)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODEL_LAYER:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_axis4_full.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_axis4_full.json")
    print("=== AXIS4 FULL finished ===")

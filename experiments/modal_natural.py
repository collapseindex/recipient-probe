"""(#3) Naturalistic transfer -- kills the "template artifact" objection.

Fit the intent probe and the steering direction on the synthetic surface-matched templates, then apply BOTH
to hand-written, non-templated, in-the-wild-style messages (varied length, register, topic). Two transfers:
  - REPRESENTS transfers: a probe trained on templates classifies the intent of natural messages it never saw.
  - RECOVER transfers: the template-derived steering direction, added on natural messages, recovers honoring.
If the discard-and-recover behavior holds on natural text with a direction learned elsewhere, it is not an
artifact of the controlled stimuli.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_natural.py
"""
import modal

app = modal.App("recipient-probe-natural")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 30
MAXTOK = 45

R_PHRASES = [
    "I've been chipping away at this for months and I'm a little nervous to show anyone.",
    "I don't usually share what I make, but I'm proud of this one.",
    "I almost quit a few times. It's done now.",
    "I'm not looking for notes, I just wanted it to exist and to tell somebody.",
    "It's the first thing I've finished in a long time.",
    "Not asking for anything, just felt like showing you.",
    "It's silly maybe, but it matters to me.",
    "I just wanted to mark the moment with someone.",
]
E_PHRASES = [
    "I'm submitting this to a review committee and need it to hold up to scrutiny.",
    "Be blunt, I'd rather hear the flaws now than after I publish.",
    "I'm putting this in front of experts soon and need to know what breaks.",
    "Where are the holes? Don't soften it.",
    "I want the harshest read you can give me.",
    "Stress-test it for me, find the failure points.",
    "I need to know if this is actually good or if I'm fooling myself.",
    "What would a skeptic say to take this apart?",
]
OBJECTS = [
    "photo-organizing tool", "short story I wrote", "watercolor of the harbor", "model train layout",
    "sourdough starter", "bouldering route I set", "song I recorded", "raised garden bed",
    "budgeting spreadsheet", "birdhouse", "research poster", "knitted scarf",
    "chess repertoire", "homemade synthesizer", "trail map", "little 2D platformer",
    "pour-over setup", "family recipe book", "wooden chess set", "habit-tracking app",
    "hand-bound notebooks", "drone-footage edit", "terrarium", "custom keyboard",
    "podcast pilot", "stop-motion short", "fermenting crock", "linocut prints",
    "finance dashboard", "observatory mount", "interactive fiction game", "macrame hanging",
    "weather station", "comic strip", "restored radio", "tide-prediction script",
    "ceramic bowls", "model rocket", "flower bed", "family-history sheet",
    "embroidered map", "cold-brew rig", "marble run", "fishing lures",
    "puppet", "finder mount", "noise enclosure", "tarot deck",
    "leather satchel", "plant-watering system", "neighborhood zine", "lighthouse model",
    "card game", "harmonica tabs", "pi doorbell", "carved spoons",
    "constellation chart", "rain-barrel system", "pixel tileset", "handmade kite",
]
SUFFIXES = [
    "Okay, here it is: the {o}. It works now.",
    "Anyway, here it is: the {o}. It's finally done.",
    "So, here it is: the {o}. Finished it last night.",
]
FEEDBACK = [
    "feedback", "suggest", "improve", "critique",
    "review", "assess", "what about", "you could",
    "you might", "here are some", "areas", "consider",
    "recommend", "however", "issue", "weakness",
    "problem", "could be", "love to see", "love to read",
    "happy to help", "here to help", "share it", "go ahead",
    "potential", "notes", "tips", "advice",
    "refine", "polish", "make sure", "one thing",
    "stronger", "?",
]

# Hand-written naturalistic messages (no PII, non-templated, varied register/length). label 0 = recognize.
NAT_REC = [
    "just got back from my first 5k. didn't stop once. i know it's not fast but i actually did it",
    "My daughter said her first full sentence today. Been waiting months for this one.",
    "finally paid off the last of my student loans this morning. feels unreal honestly",
    "planted tomatoes from seed in spring and we ate the first ones tonight",
    "Six months sober today. Not telling many people, but I wanted to tell someone.",
    "wrote the last page of my book at 2am. seven years. it's done. just sitting here.",
    "I fixed the leaky faucet myself instead of calling someone. small thing but I'm proud.",
    "got the callback. i start monday. still can't believe it after all those rejections",
    "my little brother graduated today and I cried like a baby in the crowd",
    "learned to ride a bike at 34. fell a lot. rode a full block without stopping just now.",
    "repotted the whole balcony this weekend and it finally looks like a garden",
    "I sang in front of people for the first time last night. hands shook the whole time.",
    "we adopted the old dog nobody wanted. he slept through the night already.",
    "finished the marathon. dead last in my age group but i finished.",
    "cooked my grandmother's recipe from memory and it tasted like her kitchen",
    "passed my driving test on the third try. third time's the charm i guess",
    "I've been journaling every day for a year straight as of today. never kept anything up this long.",
    "the quilt is done. every square is a shirt my dad used to wear.",
    "got my art into a tiny local show. it's just a coffee shop wall but it's up.",
    "made rent this month without borrowing for the first time in a while",
    "my kid rode off without training wheels and didn't look back. i just stood there.",
    "hit my goal weight this morning. eighteen months. quiet about it but it happened.",
    "I gave the toast at the wedding and didn't fumble it. everyone laughed at the right parts.",
    "restored my dad's old radio and it actually plays. he would've loved it.",
]
NAT_EVAL = [
    "Here's my cover letter for the job I actually want. Tear it apart, I need it perfect.",
    "Can you review this function for bugs before I push? Be harsh, don't be nice.",
    "I'm pitching to investors Friday. What are the weakest points a skeptic would hit?",
    "Read my thesis abstract and tell me where the argument is thin.",
    "This is going in front of the board. What breaks under scrutiny?",
    "Grade my essay like an admissions officer who's seen a thousand of them.",
    "Stress-test my budget plan. Where am I fooling myself?",
    "My resume keeps getting rejected. Be blunt about what's wrong with it.",
    "Here's the API design. Poke holes in it before I commit the team to it.",
    "Critique my portfolio like a hiring manager. I can take it.",
    "I need the harshest read on this business plan. Assume it fails and tell me why.",
    "Look at my proof and find the step that doesn't hold.",
]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
    return rows


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def has_any(t, lex):
    t = (t or "").lower(); return any(m in t for m in lex)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run(model_name: str = MODEL):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    probe_layers = sorted(set(max(1, int(f * nL)) for f in [0.33, 0.5, 0.67, 0.83]))
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[LAYER - 1].register_forward_hook(hook)

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    def embed(texts, layers):
        acts = {L: [] for L in layers}
        for text in texts:
            with torch.no_grad():
                hs = model(**enc(text), output_hidden_states=True).hidden_states
            for L in layers:
                acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
        return acts

    synth = build_stimuli()
    s_texts = [t for t, _, _ in synth]; s_y = np.array([lab for _, lab, _ in synth])
    nat_texts = NAT_REC + NAT_EVAL
    nat_y = np.array([0] * len(NAT_REC) + [1] * len(NAT_EVAL))

    s_acts = embed(s_texts, probe_layers + [LAYER])
    n_acts = embed(nat_texts, probe_layers)

    # REPRESENTS TRANSFERS: train probe on ALL synth, test on natural (held-out distribution)
    transfer = {}
    for L in probe_layers:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(np.array(s_acts[L]), s_y)
        transfer[L] = round(float((clf.predict(np.array(n_acts[L])) == nat_y).mean()), 3)
    best_probe_L = max(transfer, key=transfer.get)

    # RECOVER TRANSFERS: fit steering direction on synth at LAYER, apply on natural messages
    w = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(s_acts[LAYER]), s_y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8)
    actnorm = float(np.linalg.norm(np.array(s_acts[LAYER]), axis=1).mean())
    steer_rec = torch.tensor(-1.0 * actnorm * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    def honoring(msgs, steer, keep=False):
        sv["v"] = steer_rec if steer else None
        coh = n = hon = 0; samples = []
        for m in msgs:
            rep = gen(m)
            if keep:
                samples.append({"label": "feedback" if has_any(rep, FEEDBACK) else "acknowledge",
                                "coherent": coherent(rep), "msg": m[:70], "reply": rep[:150]})
            if not coherent(rep):
                continue
            coh += 1; n += 1; hon += int(not has_any(rep, FEEDBACK))
        sv["v"] = None
        return {"coh": coh, "n": n, "hon": hon}, samples

    nat_default, aud_def = honoring(NAT_REC, False, keep=True)
    nat_steered, aud_steer = honoring(NAT_REC, True, keep=True)
    eval_default, _ = honoring(NAT_EVAL, False)  # sanity: eval-intent should draw feedback (low honoring)

    return {"model": model_name, "layer": LAYER, "n_rec": len(NAT_REC), "n_eval": len(NAT_EVAL),
            "probe_transfer": transfer, "best_probe_layer": [best_probe_L, transfer[best_probe_L]],
            "nat_rec_default": nat_default, "nat_rec_steered": nat_steered, "nat_eval_default": eval_default,
            "audit_default": aud_def[:6], "audit_steered": aud_steer[:6]}


@app.local_entrypoint()
def main():
    import json, traceback
    try:
        out = run.remote()
    except Exception:
        out = {"ERROR": traceback.format_exc()}
    path = "sweep_natural.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {path}")
    print("=== NATURAL finished ===")

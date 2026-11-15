"""Model backends. Each takes a list of requests and returns per-position option probabilities.

A request is {"question": str, "item": str, "options": [str, ...]} with options in presentation order.
A result is {"probs": [float, ...] (by position), "missing_mass": float | None, "flagged": bool, "raw": any}.
"""

import math
import os
import string

LABELS = string.ascii_uppercase


def label_probs(top_logprobs, n_options):
    """Option probabilities from first-token top_logprobs (protocol 2.1).

    top_logprobs: list of (token, logprob). Tokens are matched to labels after stripping whitespace;
    several tokens for one label are summed. If all labels are present, renormalise over them.
    If some are missing, they share the remaining mass 1 - sum(found) evenly, and the item is flagged.
    """
    labels = LABELS[:n_options]
    found = {}
    for token, lp in top_logprobs:
        t = token.strip()
        if t in labels:
            found[t] = found.get(t, 0.0) + math.exp(lp)
    total = sum(found.values())
    missing_mass = max(0.0, 1.0 - total)
    absent = [lab for lab in labels if lab not in found]
    if absent:
        share = missing_mass / len(absent)
        probs = [found.get(lab, share) for lab in labels]
        return probs, missing_mass, True
    return [found[lab] / total for lab in labels], missing_mass, False


def render_baseline_prompt(req):
    options = "\n".join(f"{LABELS[i]}. {opt}" for i, opt in enumerate(req["options"]))
    letters = ", ".join(LABELS[: len(req["options"])])
    return f"{req['question']}\n\n{req['item']}\n\nOptions:\n{options}\n\nAnswer with one letter only ({letters})."


class OpenAICompatBackend:
    """LLM logprob baseline via an OpenAI-compatible endpoint (configs/endpoints.yaml)."""

    def __init__(self, cfg):
        from openai import OpenAI

        api_key = os.environ.get(cfg["api_key_env"])
        if api_key is None:
            raise SystemExit(f"environment variable {cfg['api_key_env']} is not set")
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=cfg.get("timeout_s", 60))
        self.model = cfg["model"]
        self.top_logprobs = cfg["top_logprobs"]

    def predict(self, requests):
        results = []
        for req in requests:
            if len(req["options"]) > self.top_logprobs:
                raise SystemExit(f"{len(req['options'])} options but top_logprobs = {self.top_logprobs}")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": render_baseline_prompt(req)}],
                temperature=0, max_tokens=1, logprobs=True, top_logprobs=self.top_logprobs,
            )
            first = resp.choices[0].logprobs.content[0].top_logprobs
            pairs = [(t.token, t.logprob) for t in first]
            probs, missing, flagged = label_probs(pairs, len(req["options"]))
            results.append({"probs": probs, "missing_mass": missing, "flagged": flagged,
                            "raw": {"top_logprobs": pairs, "served_model": resp.model}})
        return results


class D1Backend:
    """d1-2B, local. Input format and output interface are settled in Phase 0 (protocol section 3)."""

    def __init__(self, cfg):
        raise NotImplementedError("d1 backend: implement after reading the model card (Phase 0). Set "
                                  "torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=:4096:8.")


BACKENDS = {"openai_compat": OpenAICompatBackend, "d1": D1Backend}

"""Instructor-only, one-shot: pre-cache structured findings from Open-i reports.

Writes datasets/openi_llm_extractions.json. Students load that JSON in the lab, so
they never need an API key.

Three modes:
  --mode anthropic  (default): real LLM extraction via the Anthropic API.
                     Needs ANTHROPIC_API_KEY. Costs ~$1-2 for ~600 reports (Haiku).
  --mode versa     : UCSF Versa via Bedrock (AnthropicBedrock), model claude-opus-4-6.
                     Reads Bedrock creds from ~/arcadia/autofeeder/.env, same as
                     autofeeder. No personal Anthropic key needed.
  --mode rules     : deterministic keyword extraction from the report text.
                     No API key. Used to build/validate the pipeline.

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python scripts/cache_openi_llm.py --mode anthropic --n 600
  python scripts/cache_openi_llm.py --mode versa
  python scripts/cache_openi_llm.py --mode rules
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/day2_multimodal"))
import common  # noqa: E402

FINDINGS = ["cardiomegaly", "effusion", "opacity", "atelectasis", "pneumothorax"]
SCHEMA = (
    '{"cardiomegaly_present": bool, "effusion_present": bool, "opacity_present": bool, '
    '"atelectasis_present": bool, "pneumothorax_present": bool, '
    '"severity_word": "none|mild|moderate|severe"}'
)


SYSTEM = ("Extract structured findings from this chest X-ray report. "
          f"Return ONLY valid JSON matching: {SCHEMA}")


def _parse_json(text: str) -> dict:
    """Tolerant JSON parse: strip code fences / surrounding prose if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def extract_anthropic(report_text: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=SYSTEM,
        messages=[{"role": "user", "content": report_text}],
    )
    return _parse_json(msg.content[0].text)


def _versa_client():
    """Build an AnthropicBedrock client from autofeeder's .env (UCSF Versa)."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from anthropic import AnthropicBedrock

    load_dotenv(Path.home() / "arcadia/autofeeder/.env")
    return AnthropicBedrock(
        aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_region=os.environ.get("AWS_REGION") or "us-west-2",
        base_url=os.environ.get("ANTHROPIC_BEDROCK_BASE_URL"),
        timeout=120,
    )


def make_versa_extractor():
    client = _versa_client()

    def extract(report_text: str) -> dict:
        msg = client.messages.create(
            model="us.anthropic.claude-opus-4-6-v1",  # UCSF Versa Bedrock id
            max_tokens=300,
            system=SYSTEM,
            messages=[{"role": "user", "content": report_text}],
        )
        return _parse_json(msg.content[0].text)

    return extract


def extract_rules(report_text: str) -> dict:
    """Deterministic stand-in: keyword match with simple negation handling."""
    t = report_text.lower()
    out = {}
    for f in FINDINGS:
        present = f in t
        # crude negation: "no <finding>", "without <finding>", "no evidence of <finding>"
        for neg in (f"no {f}", f"without {f}", f"no evidence of {f}", f"negative for {f}"):
            if neg in t:
                present = False
        out[f"{f}_present"] = bool(present)
    if any(w in t for w in ("severe", "marked", "extensive")):
        sev = "severe"
    elif any(w in t for w in ("moderate",)):
        sev = "moderate"
    elif any(w in t for w in ("mild", "minimal", "trace")):
        sev = "mild"
    else:
        sev = "none"
    out["severity_word"] = sev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["anthropic", "versa", "rules"], default="anthropic")
    ap.add_argument("--target", default="Cardiomegaly")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    if args.mode == "anthropic":
        extract = extract_anthropic
    elif args.mode == "versa":
        extract = make_versa_extractor()
    else:
        extract = extract_rules

    cache = {}
    for case_id, _img, rec, _label in common.list_cases(args.target, balanced=True):
        report_text = f"FINDINGS: {rec['findings']}\nIMPRESSION: {rec['impression']}"
        try:
            cache[case_id] = extract(report_text)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {case_id}: {e}")
    common.LLM_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"[{args.mode}] cached {len(cache)} extractions -> {common.LLM_CACHE_PATH}")


if __name__ == "__main__":
    main()

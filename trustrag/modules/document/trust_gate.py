"""Document trust gate — optional source verification for TrustRAG.

TrustRAG's promise is "Reliable input, Trusted output". This module
strengthens the *input* half: before a document is parsed and indexed,
verify it with the Stipple API (https://www.stipple.sh) — forensic
authenticity signals (tamper risk band + per-signal evidence: amount/words
mismatch, font discontinuity, date anomalies, identifier checksums) and
AI-written-prose detection. Tampered payslips, edited contracts, and
LLM-generated fake documents otherwise index exactly like real ones and
become ground truth for every retrieval hit.

Free anonymous tier: no API key required. Set STIPPLE_API_KEY for your own
metering. All functions are best-effort: failures return None / record an
error and never break ingestion — the gate opens when the service is
unreachable, so an outage never blocks document processing.

Usage (standalone, or wired into RagApplication.add_document):

    from trustrag.modules.document.trust_gate import TrustGate

    gate = TrustGate(block_bands={"high"})       # advisory by default
    verdict = gate.check("invoice.pdf")
    if verdict and verdict["risk_band"] == "high":
        ...  # skip indexing / flag for review

    # or one-shot:
    from trustrag.modules.document.trust_gate import verify_document
    block = verify_document("report.pdf")
"""

import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

STIPPLE_BASE_URL = os.getenv("STIPPLE_BASE_URL", "https://www.stipple.sh")
_REQUEST_TIMEOUT = 300  # seconds
VALID_BANDS = ("low", "medium", "high")


def _headers() -> dict:
    headers = {
        "User-Agent": "trustrag-trust-gate/1.0",
        "Accept": "application/json",
    }
    api_key = os.getenv("STIPPLE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _post_file(endpoint: str, file_path: str | Path) -> Optional[dict]:
    """POST a document as multipart to a Stipple endpoint. Best-effort."""
    try:
        path = Path(file_path)
        boundary = "----trustrag-gate" + uuid.uuid4().hex
        with open(path, "rb") as f:
            content = f.read()
        body = b"".join(
            [
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{path.name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            STIPPLE_BASE_URL + endpoint,
            data=body,
            method="POST",
            headers={
                **_headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 - verification is best-effort by design
        return None


def verify_document(file_path: str | Path) -> Optional[dict]:
    """Forensic authenticity + AI-text probability for one document.

    Returns the document_trust block, or None when the API is unreachable
    (gate opens; callers index the document as before).
    """
    block: dict = {}
    warrant = _post_file("/v1/warrants", file_path)
    if warrant:
        block["authenticity"] = {
            "warrant_id": warrant.get("warrant_id"),
            "risk_band": warrant.get("risk_band"),
            "risk_score": warrant.get("risk_score"),
            "inspection_quality": warrant.get("inspection_quality"),
            "recommended_action": warrant.get("recommended_action"),
            "summary": warrant.get("summary"),
        }
    else:
        block["error"] = "verification unavailable"
    ai = _post_file("/v1/detect-ai-text", file_path)
    if ai:
        block["ai_text"] = (
            {"applicable": False}
            if ai.get("applicable") is False
            else {
                "applicable": True,
                "probability": ai.get("probability"),
                "lean": ai.get("lean"),
                "tells": ai.get("tells"),
            }
        )
    return block or None


class TrustGate:
    """Reusable pre-ingestion gate with a configurable band policy.

    Args:
        block_bands: risk bands whose documents should NOT be indexed
            (default: {"high"}). Use set() for advisory-only mode.
    """

    def __init__(self, block_bands: Optional[set] = None):
        self.block_bands = block_bands if block_bands is not None else {"high"}

    def check(self, file_path: str | Path) -> Optional[dict]:
        """Verify a document. Returns the trust block, or None if the
        service is unavailable (gate opens)."""
        return verify_document(file_path)

    def should_index(self, verdict: Optional[dict]) -> bool:
        """True when the document may be indexed (gate opens on any doubt)."""
        if verdict is None or "error" in verdict:
            return True  # fail-open
        band = (verdict.get("authenticity") or {}).get("risk_band")
        return band not in self.block_bands

    def gate(self, file_path: str | Path) -> tuple[bool, Optional[dict]]:
        """One-shot: verify + policy. Returns (should_index, verdict)."""
        verdict = self.check(file_path)
        return self.should_index(verdict), verdict

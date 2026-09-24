"""
src/retrieval/vlm.py

Gemini-based query understanding layer for retrieval.

Responsibilities:
- Read API key from a .api file at project root.
- Call Gemini 3.1 Flash Lite.
- Return STRICT JSON only.
- Support:
  - query classification
  - structured decomposition
  - hypothesis generation
  - light normalization for downstream retrieval

This module is intentionally front-end only.
It should not perform retrieval, fusion, or consensus itself.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Optional dependency: google-genai.
# The code is written so importing this module does not hard-fail immediately
# if the package is missing in environments where VLM is not used.
try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None


DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"


STRICT_SCHEMA_HINT = {
    "query_type": "one of: narrative, entity, spatial, counting, ambiguous, mixed",
    "need_context": "boolean",
    "is_ambiguous": "boolean",
    "requires_temporal_order": "boolean",
    "main_query": "string",
    "context_query": "string",
    "sub_queries": "array of strings",
    "hypotheses": "array of strings",
    "confidence": "float between 0 and 1",
}


@dataclass
class VLMResult:
    query_type: str
    need_context: bool
    is_ambiguous: bool
    requires_temporal_order: bool
    main_query: str
    context_query: str
    sub_queries: List[str]
    hypotheses: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "need_context": self.need_context,
            "is_ambiguous": self.is_ambiguous,
            "requires_temporal_order": self.requires_temporal_order,
            "main_query": self.main_query,
            "context_query": self.context_query,
            "sub_queries": self.sub_queries,
            "hypotheses": self.hypotheses,
            "confidence": self.confidence,
        }


class GeminiVLMError(RuntimeError):
    pass


class GeminiVLM:
    """
    Gemini-based query understanding helper.

    Example:
        vlm = GeminiVLM()
        result = vlm.analyze_query("người đàn ông cầm ly rồi quay lại nhìn phía sau")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        api_file: Optional[str] = None,
        timeout_seconds: int = 60,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        retry_count: int = 1,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.retry_count = retry_count
        self.api_key = api_key or self._load_api_key(api_file=api_file)

        if genai is None:
            raise GeminiVLMError(
                "google-genai is not installed. Install it or mock GeminiVLM in your environment."
            )

        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _project_root() -> Path:
        # src/retrieval/vlm.py -> project root = parents[2] if layout is src/retrieval/vlm.py
        return Path(__file__).resolve().parents[2]

    def _load_api_key(self, api_file: Optional[str] = None) -> str:
        """
        Read API key from .api at project root.

        Accepted formats:
        - plain text key in first non-empty line
        - KEY=VALUE style, where the first matching non-empty value is used
        """
        root = self._project_root()
        path = Path(api_file) if api_file else root / ".api"

        if not path.exists():
            raise GeminiVLMError(f"API key file not found: {path}")

        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise GeminiVLMError(f"API key file is empty: {path}")

        # Support `GEMINI_API_KEY=...` or `API_KEY=...`
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
            else:
                return line

        raise GeminiVLMError(f"No usable API key found in: {path}")

    @staticmethod
    def _clean_text(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            # Remove surrounding fenced blocks if the model ignores strict JSON.
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        """
        Best-effort JSON extraction if the model returns extra text.
        """
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

        return text

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
        if isinstance(value, str):
            s = value.strip()
            return [s] if s else []
        return [str(value).strip()] if str(value).strip() else []

    def _build_prompt(self, query_text: str) -> str:
        schema_str = json.dumps(STRICT_SCHEMA_HINT, ensure_ascii=False, indent=2)

        return f"""
You are a strict JSON-only query understanding module for a video retrieval system.

Your task:
1. Classify the query.
2. Decompose it into a main query and optional context query.
3. Generate hypotheses only if the query is ambiguous.
4. Return EXACTLY one JSON object and nothing else.

Allowed query_type values:
- narrative
- entity
- spatial
- counting
- ambiguous
- mixed

Rules:
- Return valid JSON only.
- No markdown.
- No explanation.
- No extra keys unless they are in the schema below.
- If a field is unknown, use an empty string, empty list, false, or 0.0 as appropriate.
- Keep main_query concise and search-friendly.
- context_query should capture temporal / contextual / disambiguating information.
- sub_queries should contain the branch queries you expect the retrieval stage to search.
- hypotheses should contain alternative interpretations only when needed.
- confidence must be between 0 and 1.

Schema:
{schema_str}

Query:
{query_text}
""".strip()

    def _fallback_result(self, query_text: str) -> VLMResult:
        q = self._clean_text(query_text)
        return VLMResult(
            query_type="mixed" if q else "entity",
            need_context=False,
            is_ambiguous=False,
            requires_temporal_order=False,
            main_query=q,
            context_query="",
            sub_queries=[q] if q else [],
            hypotheses=[],
            confidence=0.1 if q else 0.0,
        )

    def _parse_result(self, raw_text: str, query_text: str) -> VLMResult:
        cleaned = self._extract_json_candidate(self._strip_code_fences(raw_text))

        try:
            data = json.loads(cleaned)
        except Exception as exc:
            raise GeminiVLMError(f"Gemini returned invalid JSON: {exc}\nRaw output:\n{raw_text}") from exc

        if not isinstance(data, dict):
            raise GeminiVLMError("Gemini JSON output must be an object.")

        query_type = str(data.get("query_type", "mixed")).strip() or "mixed"
        need_context = bool(data.get("need_context", False))
        is_ambiguous = bool(data.get("is_ambiguous", False))
        requires_temporal_order = bool(data.get("requires_temporal_order", False))
        main_query = str(data.get("main_query", "")).strip()
        context_query = str(data.get("context_query", "")).strip()
        sub_queries = self._normalize_list(data.get("sub_queries", []))
        hypotheses = self._normalize_list(data.get("hypotheses", []))

        confidence_raw = data.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # Strong normalization
        if not main_query:
            main_query = self._clean_text(query_text)
        if not sub_queries and main_query:
            sub_queries = [main_query]
        if not context_query:
            need_context = False

        return VLMResult(
            query_type=query_type,
            need_context=need_context,
            is_ambiguous=is_ambiguous,
            requires_temporal_order=requires_temporal_order,
            main_query=main_query,
            context_query=context_query,
            sub_queries=sub_queries,
            hypotheses=hypotheses,
            confidence=confidence,
        )

    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Main public method.

        Returns a strict JSON-compatible dict.
        """
        query_text = self._clean_text(query_text)
        if not query_text:
            return self._fallback_result(query_text).to_dict()

        prompt = self._build_prompt(query_text)

        last_error: Optional[Exception] = None
        for attempt in range(self.retry_count + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )

                raw_text = getattr(resp, "text", None)
                if not raw_text:
                    # Some SDK versions store parts instead of text.
                    raw_text = str(resp)

                parsed = self._parse_result(raw_text, query_text=query_text)
                return parsed.to_dict()

            except Exception as exc:
                last_error = exc

        # Fallback if all retries fail
        if last_error is not None:
            fallback = self._fallback_result(query_text)
            fallback_dict = fallback.to_dict()
            fallback_dict["error"] = f"{type(last_error).__name__}: {last_error}"
            return fallback_dict

        return self._fallback_result(query_text).to_dict()

    def classify_only(self, query_text: str) -> Dict[str, Any]:
        """
        Convenience wrapper when you only need classification,
        but still want strict JSON output.
        """
        result = self.analyze_query(query_text)
        return {
            "query_type": result.get("query_type", "mixed"),
            "need_context": result.get("need_context", False),
            "is_ambiguous": result.get("is_ambiguous", False),
            "requires_temporal_order": result.get("requires_temporal_order", False),
            "confidence": result.get("confidence", 0.0),
        }


def load_vlm_from_project_root(
    api_file: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> GeminiVLM:
    """
    Factory helper for the retrieval pipeline.
    """
    return GeminiVLM(api_file=api_file, model_name=model_name)


if __name__ == "__main__":
    # Lightweight smoke test.
    vlm = load_vlm_from_project_root()
    sample = "người đàn ông cầm ly rồi quay lại nhìn phía sau"
    result = vlm.analyze_query(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))

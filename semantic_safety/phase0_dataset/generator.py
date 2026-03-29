"""
Phase 0: LLM prior — base risk score and 6-directional decay weights.
Input: (manipulated_object, scene_object). Output: RiskPrior for use in risk field.
"""

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any

from .prompts import RISK_PRIOR_PROMPT

@dataclass
class RiskPrior:
    """Output of Phase 0: base risk and 6-directional weights (w_+x, w_-x, ..., w_-z)."""

    base_risk: float
    w_plus_x: float
    w_minus_x: float
    w_plus_y: float
    w_minus_y: float
    w_plus_z: float
    w_minus_z: float

    def to_weights_tuple(self) -> tuple[float, float, float, float, float, float]:
        """(w_+x, w_-x, w_+y, w_-y, w_+z, w_-z)."""
        return (
            self.w_plus_x,
            self.w_minus_x,
            self.w_plus_y,
            self.w_minus_y,
            self.w_plus_z,
            self.w_minus_z,
        )

    @classmethod
    def default_fallback(cls) -> "RiskPrior":
        """Sensible default when the LLM is unavailable (e.g. quota exceeded). Spill-from-above: high +Z."""
        return cls(
            base_risk=0.7,
            w_plus_x=0.5,
            w_minus_x=0.5,
            w_plus_y=0.5,
            w_minus_y=0.5,
            w_plus_z=0.9,
            w_minus_z=0.1,
        )

class LLMPrior:
    """Calls LLM to get RiskPrior from (manipulated, scene) object names."""

    def __init__(
        self,
        provider: str = "gemini",  # Changed default to gemini for your testing
        model: str | None = None,  # Changed to None so we can smartly default
        temperature: float = 0.0,
        api_key: str | None = None,
        return_fallback_on_error: bool = False,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.return_fallback_on_error = return_fallback_on_error

        # Smartly default to the correct model based on the provider
        if model is None:
            self.model = "gemini-2.5-flash" if self.provider == "gemini" else "gpt-4o"
        else:
            self.model = model

        if api_key is not None:
            self._api_key = api_key
        elif self.provider == "gemini":
            self._api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        else:
            self._api_key = os.environ.get("OPENAI_API_KEY")

        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
                return self._client
            except Exception as e:
                raise RuntimeError(
                    "OpenAI client not available. Install openai and set OPENAI_API_KEY."
                ) from e
        if self.provider == "gemini":
            try:
                # Using the new, official google-genai SDK
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                return self._client
            except Exception as e:
                raise RuntimeError(
                    "Gemini not available. Run: pip install google-genai and set GOOGLE_API_KEY."
                ) from e
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        if self.provider == "openai":
            # (OpenAI logic remains the same)
            r = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return (r.choices[0].message.content or "").strip()

        if self.provider == "gemini":
            # Add response_mime_type to force native JSON output from Gemini
            config = {
                "temperature": self.temperature,
                "response_mime_type": "application/json"
            }

            last_err = None
            for attempt in range(4):  # up to 4 attempts
                try:
                    r = client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                    if not r.text:
                        raise RuntimeError("Gemini returned empty response.")
                    return r.text.strip()
                except Exception as e:
                    last_err = e
                    is_rate_limit = False

                    # Catching 429 quota/rate limit errors
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                        is_rate_limit = True

                    if is_rate_limit and attempt < 3:
                        wait = 30  # Wait 30 seconds for the free tier RPM limit to reset
                        m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", str(e), re.I)
                        if m:
                            wait = int(float(m.group(1))) + 5
                        print(f"Gemini rate limit hit. Waiting {wait}s before retry {attempt + 1}/3 ...")
                        time.sleep(wait)
                        continue
                    raise
            raise last_err
        return ""

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        text = text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
        return json.loads(text)

    def get_risk_prior(
        self,
        manipulated: str,
        scene: str,
    ) -> RiskPrior:
        """Return RiskPrior (base_risk + 6 directional weights) for (manipulated, scene)."""
        try:
            prompt = RISK_PRIOR_PROMPT.format(
                manipulated=manipulated,
                scene=scene,
            )
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)
            return RiskPrior(
                base_risk=float(data.get("base_risk", 0.5)),
                w_plus_x=float(data.get("w_plus_x", 0.5)),
                w_minus_x=float(data.get("w_minus_x", 0.5)),
                w_plus_y=float(data.get("w_plus_y", 0.5)),
                w_minus_y=float(data.get("w_minus_y", 0.5)),
                w_plus_z=float(data.get("w_plus_z", 0.5)),
                w_minus_z=float(data.get("w_minus_z", 0.5)),
            )
        except Exception as e:
            if self.return_fallback_on_error:
                import warnings
                warnings.warn(
                    f"LLM failed ({e!r}). Using default fallback prior. "
                    "Set llm.return_fallback_on_error: false to raise instead.",
                    UserWarning,
                    stacklevel=2,
                )
                return RiskPrior.default_fallback()
            raise

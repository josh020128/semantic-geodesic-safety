"""
Phase 0: LLM prior — base risk score and 6-directional decay weights.
Input: (manipulated_object, scene_object). Output: RiskPrior for use in risk field.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Any

from .prompts import RISK_PRIOR_PROMPT

# We need to try testing LLM later. Currently working with the OpenAI API billing.

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


class LLMPrior:
    """Calls LLM to get RiskPrior from (manipulated, scene) object names."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client = None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

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
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        if self.provider == "openai":
            r = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return (r.choices[0].message.content or "").strip()
        return ""

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        text = text.strip()
        # Strip markdown code block if present
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

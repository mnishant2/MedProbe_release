"""Thin OpenAI-compatible wrapper for OpenRouter. Retry + JSON mode + cost ledger."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class GeneratorProfile:
    name: str
    model: str
    temperature: float
    max_tokens: int
    response_format: str | None
    price_in_per_mtok: float
    price_out_per_mtok: float
    # Optional per-profile endpoint override. When set, the client builds a
    # separate OpenAI-compatible connection to this endpoint using the key in
    # api_key_env (e.g. a provider's direct API instead of OpenRouter).
    base_url: str | None = None
    api_key_env: str | None = None

    @classmethod
    def from_cfg(cls, name: str, cfg: dict[str, Any]) -> "GeneratorProfile":
        return cls(
            name=name,
            model=cfg["model"],
            temperature=float(cfg.get("temperature", 0.6)),
            max_tokens=int(cfg.get("max_tokens", 600)),
            response_format=cfg.get("response_format"),
            price_in_per_mtok=float(cfg.get("price_in_per_mtok", 0.0)),
            price_out_per_mtok=float(cfg.get("price_out_per_mtok", 0.0)),
            base_url=cfg.get("base_url"),
            api_key_env=cfg.get("api_key_env"),
        )


class OpenRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        referer: str,
        app_title: str,
        retry_max_attempts: int = 6,
        retry_initial_wait: float = 2.0,
        retry_max_wait: float = 60.0,
        ledger_path: Path | None = None,
    ) -> None:
        key = os.environ.get(api_key_env, "")
        if not key:
            raise RuntimeError(
                f"Missing {api_key_env}. Copy .env.example to .env and set the key."
            )
        self._default_client = OpenAI(
            base_url=base_url,
            api_key=key,
            default_headers={"HTTP-Referer": referer, "X-Title": app_title},
        )
        # Lazy-built per-profile clients keyed by profile.name when a profile
        # specifies its own base_url + api_key_env.
        self._profile_clients: dict[str, OpenAI] = {}
        self._retry_max_attempts = retry_max_attempts
        self._retry_initial_wait = retry_initial_wait
        self._retry_max_wait = retry_max_wait
        self._ledger_path = ledger_path
        if ledger_path is not None:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _client_for(self, profile: "GeneratorProfile") -> OpenAI:
        if not profile.base_url or not profile.api_key_env:
            return self._default_client
        if profile.name not in self._profile_clients:
            key = os.environ.get(profile.api_key_env, "")
            if not key:
                raise RuntimeError(
                    f"Profile {profile.name!r} requires env var "
                    f"{profile.api_key_env}. Set it in .env."
                )
            self._profile_clients[profile.name] = OpenAI(
                base_url=profile.base_url,
                api_key=key,
            )
        return self._profile_clients[profile.name]

    def generate_json(
        self,
        profile: GeneratorProfile,
        prompt: str,
        system: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make one call. Returns (parsed_json, meta) where meta includes tokens + cost."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
        }
        if profile.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        client = self._client_for(profile)
        wrapped = self._build_retry()(client.chat.completions.create)
        t0 = time.time()
        resp = wrapped(**kwargs)
        dt = time.time() - t0

        content = resp.choices[0].message.content or ""
        parsed = _safe_json_loads(content)

        usage = getattr(resp, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        c_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = (p_tok / 1e6) * profile.price_in_per_mtok + (c_tok / 1e6) * profile.price_out_per_mtok
        meta = {
            "model": profile.model,
            "profile": profile.name,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "cost_usd": round(cost, 6),
            "latency_s": round(dt, 3),
        }
        self._append_ledger(meta)
        return parsed, meta

    def _build_retry(self):
        return retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self._retry_max_attempts),
            wait=wait_exponential(
                multiplier=self._retry_initial_wait,
                max=self._retry_max_wait,
            ),
            reraise=True,
        )

    def _append_ledger(self, meta: dict[str, Any]) -> None:
        if self._ledger_path is None:
            return
        with self._ledger_path.open("a") as fh:
            fh.write(json.dumps(meta) + "\n")


def _safe_json_loads(s: str) -> dict[str, Any]:
    """Robust JSON parse. Strips markdown fences if model added them."""
    s = s.strip()
    if s.startswith("```"):
        # strip ```json\n ... \n```
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # last resort: extract the first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start : end + 1])
        raise

"""Online SDPO router inference for adaptive memory/skill selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
from typing import Any

import httpx
from openai import OpenAI

from nanoresearch.config import ResearchConfig

logger = logging.getLogger(__name__)

PRE_ROUTER_SYSTEM = (
    "You are a router making pre-execution decisions for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Use only ids listed in x.candidate_memory and x.candidate_skills. "
    "Select a focused subset, not everything. "
    "When task constraints conflict with persona defaults, prioritize task-specific constraints. "
    "Set update_memory and update_skill to null. "
    "Keep prompt_plan under 30 words. Output one valid JSON object only."
)

POST_ROUTER_SYSTEM = (
    "You are a hindsight-improved router for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Use only ids listed in x.candidate_memory and x.candidate_skills plus any evolved ids already present in x.candidate_skills. "
    "Improve retrieval and prompt planning after feedback. "
    "When task constraints conflict with persona defaults, prioritize task-specific constraints. "
    "Write update_memory only for stable preferences or recurring constraints. "
    "Write update_skill only for reusable procedural rules. "
    "Keep prompt_plan under 30 words. Keep each update to one short sentence. Output one valid JSON object only."
)

ROUTER_KEY_ORDER = (
    "selected_memory_ids",
    "selected_skill_ids",
    "prompt_plan",
    "update_memory",
    "update_skill",
)

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class RouterDecision:
    selected_memory_ids: list[str]
    selected_skill_ids: list[str]
    prompt_plan: str
    update_memory: str | None
    update_skill: str | None
    backend: str
    raw_response: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_memory_ids": list(self.selected_memory_ids),
            "selected_skill_ids": list(self.selected_skill_ids),
            "prompt_plan": self.prompt_plan,
            "update_memory": self.update_memory,
            "update_skill": self.update_skill,
            "backend": self.backend,
            "raw_response": self.raw_response,
        }


class RouterPolicyRunner:
    """Run the trained SDPO router online via local HF weights or an endpoint."""

    def __init__(self, config: ResearchConfig) -> None:
        self._model_path = str(getattr(config, "router_sdpo_model_path", "") or "").strip()
        self._model_name = str(getattr(config, "router_sdpo_model_name", "") or "").strip()
        self._base_url = str(getattr(config, "router_sdpo_base_url", "") or "").strip()
        self._api_key = str(getattr(config, "router_sdpo_api_key", "") or "").strip()
        self._timeout = float(getattr(config, "router_sdpo_timeout", 120.0) or 120.0)
        self._temperature = float(getattr(config, "router_sdpo_temperature", 0.0) or 0.0)
        self._max_new_tokens = int(getattr(config, "router_sdpo_max_new_tokens", 256) or 256)

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url or self._model_path)

    def backend_name(self) -> str:
        if self._base_url:
            return f"remote:{self._model_name or 'router-sdpo'}"
        if self._model_path:
            return f"local:{self._model_path}"
        return "unconfigured"

    def decide(self, payload: dict[str, Any], *, post_feedback: bool = False) -> RouterDecision:
        if not self.is_configured:
            raise RuntimeError(
                "same_router_hindsight_sdpo_enabled=True but no SDPO router backend is configured. "
                "Set either router_sdpo_model_path or router_sdpo_base_url/router_sdpo_model_name."
            )

        system_prompt = POST_ROUTER_SYSTEM if post_feedback else PRE_ROUTER_SYSTEM
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        raw_response = self._generate(system_prompt, user_prompt)
        action = self._canonicalize_action(self._extract_action(raw_response))
        action = self._validate_action(action, payload.get("x", {}))
        return RouterDecision(
            selected_memory_ids=action["selected_memory_ids"],
            selected_skill_ids=action["selected_skill_ids"],
            prompt_plan=action["prompt_plan"],
            update_memory=action["update_memory"],
            update_skill=action["update_skill"],
            backend=self.backend_name(),
            raw_response=raw_response,
        )

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._base_url:
            return self._generate_remote(system_prompt, user_prompt)
        return self._generate_local(system_prompt, user_prompt)

    def _generate_remote(self, system_prompt: str, user_prompt: str) -> str:
        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key or "EMPTY",
            timeout=httpx.Timeout(self._timeout, connect=15.0),
        )
        kwargs: dict[str, Any] = {
            "model": self._model_name or self._model_path,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._temperature > 0:
            kwargs["temperature"] = self._temperature
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("SDPO router endpoint returned no choices")
        return response.choices[0].message.content or ""

    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        if not self._model_path:
            raise RuntimeError("Local SDPO router requested without router_sdpo_model_path")
        import torch

        tokenizer, model = self._load_local_model(self._model_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(tokenizer, "chat_template", None):
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_kwargs,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": self._temperature > 0,
        }
        if self._temperature > 0:
            generate_kwargs["temperature"] = self._temperature
            generate_kwargs["top_p"] = 0.95

        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        completion_ids = output[0][prompt_len:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    @staticmethod
    def _load_local_model(model_path: str) -> tuple[Any, Any]:
        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_path)
            if cached is not None:
                return cached
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            load_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            if torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            _MODEL_CACHE[model_path] = (tokenizer, model)
            return tokenizer, model

    @staticmethod
    def _extract_action(raw_response: str) -> dict[str, Any]:
        text = (raw_response or "").strip()
        if not text:
            raise RuntimeError("SDPO router returned empty response")
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        brace_start = text.find("{")
        if brace_start < 0:
            raise RuntimeError(f"SDPO router did not return JSON: {text[:200]}")
        depth = 0
        for idx in range(brace_start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    fragment = text[brace_start:idx + 1]
                    try:
                        return json.loads(fragment)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"SDPO router returned invalid JSON: {fragment[:300]}") from exc
        raise RuntimeError(f"SDPO router JSON was not closed: {text[:300]}")

    @staticmethod
    def _canonicalize_action(action: dict[str, Any]) -> dict[str, Any]:
        canonical: dict[str, Any] = {}
        for key in ROUTER_KEY_ORDER:
            value = action.get(key)
            if key.endswith("_ids"):
                if not value:
                    canonical[key] = []
                elif isinstance(value, list):
                    canonical[key] = [str(item) for item in value if str(item).strip()]
                else:
                    canonical[key] = [str(value)]
            elif key.startswith("update_"):
                text = " ".join(str(value or "").split())
                canonical[key] = text or None
            else:
                canonical[key] = " ".join(str(value or "").split())
        return canonical

    @staticmethod
    def _validate_action(action: dict[str, Any], router_x: dict[str, Any]) -> dict[str, Any]:
        valid_memory_ids = {
            str(item.get("memory_id"))
            for item in (router_x.get("candidate_memory") or [])
            if item.get("memory_id")
        }
        valid_skill_ids = {
            str(item.get("skill_id"))
            for item in (router_x.get("candidate_skills") or [])
            if item.get("skill_id")
        }
        action["selected_memory_ids"] = [
            item for item in action["selected_memory_ids"] if item in valid_memory_ids
        ]
        action["selected_skill_ids"] = [
            item for item in action["selected_skill_ids"] if item in valid_skill_ids
        ]
        if len(action["prompt_plan"].split()) > 30:
            action["prompt_plan"] = " ".join(action["prompt_plan"].split()[:30])
        return action

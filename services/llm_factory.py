from __future__ import annotations

import base64
from io import BytesIO
import json
import mimetypes
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx


def _httpx_timeout() -> httpx.Timeout:
    """
    Configurable client timeout for long-running generation steps (e.g. write_spec).
    """
    total = float(os.getenv("LLM_HTTP_TIMEOUT_SECONDS", "240"))
    connect = float(os.getenv("LLM_HTTP_CONNECT_TIMEOUT_SECONDS", "20"))
    # Keep write/pool conservative; most latency is on read.
    return httpx.Timeout(timeout=total, connect=connect, write=30.0, pool=30.0)


def build_llm_callable(
    *,
    provider: str | None,
    model: str | None,
    vision_model: str | None,
    base_url: str | None,
    api_key: str | None,
    temperature: float | None = None,
):
    """
    Build an LLM callable compatible with BaseStructuredAgent.
    Falls back to deterministic stub mode when provider/model/key are incomplete.
    """
    provider_name = _normalize_provider_name(provider)
    runtime_temperature = _normalize_temperature(temperature)
    text_model_name = (model or "").strip()
    vision_model_name = (vision_model or "").strip()
    runtime_api_key = _resolve_api_key(provider_name, api_key)
    if not provider_name or not text_model_name or not runtime_api_key:
        return None

    if provider_name in {"openai", "kimi", "minimax", "qwen", "doubao", "deepseek", "glm"}:
        endpoint = _normalize_openai_compat_base_url(provider_name, base_url)
        return lambda prompt, context: _call_openai_compatible(
            endpoint=endpoint,
            text_model=text_model_name,
            vision_model=vision_model_name,
            api_key=runtime_api_key,
            prompt=prompt,
            context=context,
            provider=provider_name,
            temperature=runtime_temperature,
        )
    if provider_name == "claude":
        endpoint = (base_url or "https://api.anthropic.com").rstrip("/")
        return lambda prompt, context: _call_claude(
            endpoint=endpoint,
            text_model=text_model_name,
            vision_model=vision_model_name,
            api_key=runtime_api_key,
            prompt=prompt,
            context=context,
            temperature=runtime_temperature,
        )
    if provider_name == "gemini":
        endpoint = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        return lambda prompt, context: _call_gemini(
            endpoint=endpoint,
            text_model=text_model_name,
            vision_model=vision_model_name,
            api_key=runtime_api_key,
            prompt=prompt,
            context=context,
            temperature=runtime_temperature,
        )
    return None


def _normalize_provider_name(provider: str | None) -> str:
    value = (provider or "").strip().lower()
    if value == "anthropic":
        return "claude"
    return value


def _normalize_temperature(temperature: float | None) -> float:
    if temperature is None:
        return 0.2
    try:
        value = float(temperature)
    except (TypeError, ValueError):
        return 0.2
    if value < 0:
        return 0.0
    if value > 2:
        return 2.0
    return value


def _resolve_api_key(provider: str, api_key: str | None) -> str:
    key = (api_key or "").strip()
    if key:
        return key
    provider_env_map = {
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "kimi": "MOONSHOT_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "doubao": "ARK_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "glm": "ZHIPUAI_API_KEY",
    }
    env_key = provider_env_map.get(provider)
    if env_key:
        return os.getenv(env_key, "").strip()
    return ""


def _normalize_openai_compat_base_url(provider: str, base_url: str | None) -> str:
    defaults = {
        "openai": "https://api.openai.com/v1",
        "kimi": "https://api.moonshot.cn/v1",
        "minimax": "https://api.minimax.chat/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "doubao": "https://ark.cn-beijing.volces.com/api/v3",
        "deepseek": "https://api.deepseek.com/v1",
        "glm": "https://open.bigmodel.cn/api/paas/v4",
    }
    raw = (base_url or defaults.get(provider, "https://api.openai.com/v1")).rstrip("/")
    if not raw.endswith("/v1") and provider in {"openai", "kimi", "minimax", "qwen", "deepseek"}:
        return f"{raw}/v1"
    return raw


def _build_messages(prompt: str, context: dict[str, Any]) -> list[dict[str, str]]:
    schema = context.get("_output_schema")
    schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
    system = (
        "You are a strict JSON generator. "
        "Return only one valid JSON object and no markdown. "
        f"JSON schema: {schema_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _read_image_payloads(context: dict[str, Any]) -> list[dict[str, str]]:
    """
    Read local images and return payloads with mime type + base64 body.
    Used by multimodal providers in drawing/visual nodes.
    """
    preloaded = context.get("_image_payloads")
    if isinstance(preloaded, list):
        normalized_preloaded: list[dict[str, str]] = []
        for item in preloaded:
            if not isinstance(item, dict):
                continue
            mime = item.get("mime_type")
            b64 = item.get("b64")
            if isinstance(mime, str) and isinstance(b64, str) and mime.startswith("image/") and b64:
                normalized_preloaded.append({"mime_type": mime, "b64": b64})
        if normalized_preloaded:
            return normalized_preloaded

    merged_paths: list[str] = []
    for key in ("image_paths", "application_image_paths", "prior_art_image_paths"):
        paths = context.get(key)
        if isinstance(paths, list):
            for item in paths:
                if isinstance(item, str) and item.strip():
                    merged_paths.append(item)
    if len(merged_paths) == 0:
        return []

    image_mime_types = context.get("image_mime_types")

    max_images = int(os.getenv("LLM_MAX_VISION_IMAGES", "4"))
    max_image_bytes = int(os.getenv("LLM_MAX_VISION_IMAGE_BYTES", str(4 * 1024 * 1024)))

    payloads: list[dict[str, str]] = []
    for idx, raw_path in enumerate(merged_paths):
        if len(payloads) >= max_images:
            break
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        try:
            with open(raw_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        if len(raw) == 0 or len(raw) > max_image_bytes:
            continue

        mime_type = "image/png"
        if isinstance(image_mime_types, list) and idx < len(image_mime_types):
            mt = image_mime_types[idx]
            if isinstance(mt, str) and mt.startswith("image/"):
                mime_type = mt
        else:
            guessed, _ = mimetypes.guess_type(raw_path)
            if isinstance(guessed, str) and guessed.startswith("image/"):
                mime_type = guessed
        normalized = _normalize_image_for_vision(raw=raw, mime_type=mime_type)
        if normalized is None:
            # Skip images that cannot be normalized/decoded by current runtime.
            continue
        normalized_mime, normalized_raw = normalized
        b64 = base64.b64encode(normalized_raw).decode("ascii")
        payloads.append({"mime_type": normalized_mime, "b64": b64})
    return payloads


def _normalize_image_for_vision(*, raw: bytes, mime_type: str) -> tuple[str, bytes] | None:
    """
    Normalize image bytes to formats commonly accepted by vision endpoints.
    Priority: keep PNG/JPEG as-is; convert others to PNG via Pillow when available.
    """
    mt = (mime_type or "").lower().strip()
    if mt in {"image/png", "image/jpeg"}:
        return mt, raw
    if mt == "image/jpg":
        return "image/jpeg", raw

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        with Image.open(BytesIO(raw)) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
            out = BytesIO()
            # Use PNG to preserve widest compatibility for converted sources.
            img.save(out, format="PNG")
            return "image/png", out.getvalue()
    except Exception:
        return None


def _pick_model_for_context(*, text_model: str, vision_model: str | None, context: dict[str, Any]) -> str:
    image_payloads = _read_image_payloads(context)
    if image_payloads and vision_model and vision_model.strip():
        return vision_model.strip()
    return text_model


def _call_openai_compatible(
    *,
    endpoint: str,
    text_model: str,
    vision_model: str,
    api_key: str,
    prompt: str,
    context: dict[str, Any],
    provider: str,
    temperature: float,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    image_payloads = _read_image_payloads(context)
    selected_model = _pick_model_for_context(text_model=text_model, vision_model=vision_model, context=context)
    messages = _build_messages(prompt, context)
    if image_payloads:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in image_payloads:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['b64']}"},
                }
            )
        messages = [
            messages[0],
            {"role": "user", "content": user_content},
        ]

    body: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
    }
    # Some providers support response_format, but with multimodal payloads this can trigger 400
    # on certain compatible endpoints/models. Keep it for text-only calls.
    if provider in {"openai", "qwen", "doubao", "kimi"} and not image_payloads:
        body["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=_httpx_timeout()) as client:
        native_error_detail: str | None = None
        resp = client.post(f"{endpoint}/chat/completions", headers=headers, json=body)
        if not resp.is_success and image_payloads:
            # Fallback for some OpenAI-compatible gateways that expect image_url as plain string.
            alt_messages = [messages[0]]
            alt_user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in image_payloads:
                alt_user_content.append(
                    {
                        "type": "image_url",
                        "image_url": f"data:{img['mime_type']};base64,{img['b64']}",
                    }
                )
            alt_messages.append({"role": "user", "content": alt_user_content})
            alt_body = {
                "model": selected_model,
                "messages": alt_messages,
                "temperature": temperature,
                "max_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
            }
            resp = client.post(f"{endpoint}/chat/completions", headers=headers, json=alt_body)
        if not resp.is_success and image_payloads and provider == "qwen":
            # Qwen-compatible fallback: avoid system role and inline schema in user text.
            schema = context.get("_output_schema")
            schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
            qwen_user_content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "你是严格JSON生成器。只返回一个JSON对象，不要markdown。\n"
                        f"JSON schema: {schema_text}\n\n"
                        f"任务:\n{prompt}"
                    ),
                }
            ]
            for img in image_payloads:
                qwen_user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['mime_type']};base64,{img['b64']}"},
                    }
                )
            qwen_body = {
                "model": selected_model,
                "messages": [{"role": "user", "content": qwen_user_content}],
                "temperature": temperature,
                "max_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
            }
            resp = client.post(f"{endpoint}/chat/completions", headers=headers, json=qwen_body)
        if not resp.is_success and image_payloads and provider == "qwen":
            # Qwen fallback: some gateways prefer base64:// over data URLs.
            schema = context.get("_output_schema")
            schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
            qwen_b64_user_content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "你是严格JSON生成器。只返回一个JSON对象，不要markdown。\n"
                        f"JSON schema: {schema_text}\n\n"
                        f"任务:\n{prompt}"
                    ),
                }
            ]
            for img in image_payloads:
                qwen_b64_user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"base64://{img['b64']}"},
                    }
                )
            qwen_b64_body = {
                "model": selected_model,
                "messages": [{"role": "user", "content": qwen_b64_user_content}],
                "temperature": temperature,
                "max_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
            }
            resp = client.post(f"{endpoint}/chat/completions", headers=headers, json=qwen_b64_body)
        if not resp.is_success and image_payloads and provider == "qwen" and len(image_payloads) > 1:
            # Isolate bad image: retry with only one image to avoid full-request failure.
            schema = context.get("_output_schema")
            schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
            single = image_payloads[0]
            qwen_single_body = {
                "model": selected_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "你是严格JSON生成器。只返回一个JSON对象，不要markdown。\n"
                                    f"JSON schema: {schema_text}\n\n"
                                    f"任务:\n{prompt}"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"base64://{single['b64']}"},
                            },
                        ],
                    }
                ],
                "temperature": temperature,
                "max_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
            }
            resp = client.post(f"{endpoint}/chat/completions", headers=headers, json=qwen_single_body)
        if not resp.is_success and image_payloads and provider == "qwen":
            native, native_error_detail = _call_qwen_native_multimodal(
                compatible_endpoint=endpoint,
                model=selected_model,
                api_key=api_key,
                prompt=prompt,
                context=context,
                image_payloads=image_payloads,
            )
            if native is not None:
                return native

        if not resp.is_success:
            detail = (resp.text or "").strip()
            detail = detail[:1200] if detail else ""
            if native_error_detail:
                detail = f"{detail}; {native_error_detail}" if detail else native_error_detail
            raise ValueError(
                f"OpenAI-compatible call failed: HTTP {resp.status_code}; "
                f"provider={provider}; model={selected_model}; body={detail or '<empty>'}"
            )
        payload = resp.json()
    content = payload["choices"][0]["message"]["content"]
    return _coerce_json_object(content)


def _derive_qwen_native_base(compatible_endpoint: str) -> str:
    """
    Convert DashScope OpenAI-compatible endpoint to native API base.
    Example:
    https://dashscope.aliyuncs.com/compatible-mode/v1 -> https://dashscope.aliyuncs.com/api/v1
    """
    endpoint = compatible_endpoint.rstrip("/")
    if "/compatible-mode/v1" in endpoint:
        return endpoint.replace("/compatible-mode/v1", "/api/v1")
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/api/v1"


def _call_qwen_native_multimodal(
    *,
    compatible_endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    context: dict[str, Any],
    image_payloads: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    """
    DashScope native multimodal fallback for Qwen.
    Uses data URI images via native endpoint when compatible-mode rejects image_url formats.
    """
    native_base = _derive_qwen_native_base(compatible_endpoint)
    if not native_base:
        return None, "qwen_native_endpoint_unavailable"
    schema = context.get("_output_schema")
    schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
    content_items: list[dict[str, str]] = []
    for img in image_payloads:
        content_items.append({"image": f"data:{img['mime_type']};base64,{img['b64']}"})
    content_items.append(
        {
            "text": (
                "你是严格JSON生成器。只返回一个JSON对象，不要markdown。\n"
                f"JSON schema: {schema_text}\n\n"
                f"任务:\n{prompt}"
            )
        }
    )
    body = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content_items,
                }
            ]
        },
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=_httpx_timeout()) as client:
        resp = client.post(
            f"{native_base}/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=body,
        )
        if not resp.is_success:
            detail = (resp.text or "").strip()
            detail = detail[:1200] if detail else "<empty>"
            return None, f"qwen_native_failed_http_{resp.status_code}: {detail}"
        payload = resp.json()

    try:
        content = payload["output"]["choices"][0]["message"]["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    return _coerce_json_object(item["text"]), None
        if isinstance(content, str):
            return _coerce_json_object(content), None
    except Exception as exc:
        return None, f"qwen_native_parse_failed: {exc}"
    return None, "qwen_native_empty_content"


def _call_claude(
    *,
    endpoint: str,
    text_model: str,
    vision_model: str,
    api_key: str,
    prompt: str,
    context: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    schema = context.get("_output_schema")
    schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    image_payloads = _read_image_payloads(context)
    selected_model = _pick_model_for_context(text_model=text_model, vision_model=vision_model, context=context)
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in image_payloads:
        user_content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["mime_type"],
                    "data": img["b64"],
                },
            }
        )

    body = {
        "model": selected_model,
        "max_tokens": 4096,
        "temperature": temperature,
        "system": (
            "Return only one valid JSON object and no markdown. "
            f"JSON schema: {schema_text}"
        ),
        "messages": [{"role": "user", "content": user_content}],
    }
    with httpx.Client(timeout=_httpx_timeout()) as client:
        resp = client.post(f"{endpoint}/v1/messages", headers=headers, json=body)
        resp.raise_for_status()
        payload = resp.json()
    text_blocks = payload.get("content", [])
    text = ""
    for block in text_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text", ""))
            break
    return _coerce_json_object(text)


def _call_gemini(
    *,
    endpoint: str,
    text_model: str,
    vision_model: str,
    api_key: str,
    prompt: str,
    context: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    schema = context.get("_output_schema")
    schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, dict) else "{}"
    image_payloads = _read_image_payloads(context)
    selected_model = _pick_model_for_context(text_model=text_model, vision_model=vision_model, context=context)
    parts: list[dict[str, Any]] = [{"text": f"Return JSON only.\nSchema:\n{schema_text}\n\nPrompt:\n{prompt}"}]
    for img in image_payloads:
        parts.append(
            {
                "inline_data": {
                    "mime_type": img["mime_type"],
                    "data": img["b64"],
                }
            }
        )
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": temperature},
    }
    with httpx.Client(timeout=_httpx_timeout()) as client:
        resp = client.post(
            f"{endpoint}/models/{selected_model}:generateContent",
            params={"key": api_key},
            json=body,
        )
        resp.raise_for_status()
        payload = resp.json()
    candidates = payload.get("candidates", [])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            text = str(parts[0].get("text", ""))
    return _coerce_json_object(text)


def _coerce_json_object(content: str) -> dict[str, Any]:
    """
    Parse JSON object from model text, tolerating fenced code blocks.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_json_text(text)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            # Fallback: keep only the outermost JSON object span, then repair again.
            start_idx = repaired.find("{")
            end_idx = repaired.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                narrowed = _repair_json_text(repaired[start_idx : end_idx + 1])
                parsed = json.loads(narrowed)
            else:
                raise
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def _repair_json_text(text: str) -> str:
    """
    Best-effort repair for common model JSON glitches:
    - raw newlines/tabs inside strings
    - unterminated braces caused by truncated tail
    """
    buf: list[str] = []
    in_string = False
    escaped = False
    brace_stack: list[str] = []

    for ch in text:
        if in_string:
            if escaped:
                buf.append(ch)
                escaped = False
                continue
            if ch == "\\":
                buf.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                buf.append(ch)
                continue
            if ch == "\n":
                buf.append("\\n")
                continue
            if ch == "\r":
                continue
            if ch == "\t":
                buf.append("\\t")
                continue
            buf.append(ch)
            continue

        if ch == '"':
            in_string = True
            buf.append(ch)
            continue
        if ch in "{[":
            brace_stack.append("}" if ch == "{" else "]")
            buf.append(ch)
            continue
        if ch in "}]":
            if brace_stack and brace_stack[-1] == ch:
                brace_stack.pop()
            buf.append(ch)
            continue
        buf.append(ch)

    if in_string:
        buf.append('"')
    while brace_stack:
        buf.append(brace_stack.pop())
    repaired = "".join(buf)
    # Normalize common non-JSON punctuations outside strings, e.g. full-width colon/comma.
    repaired = _normalize_json_punctuation(repaired)
    # Convert single-quoted keys/values to JSON-compatible double quotes (best-effort).
    repaired = _normalize_single_quoted_json(repaired)
    # Quote bare keys in objects: {foo: 1} -> {"foo": 1}
    repaired = _quote_bare_object_keys(repaired)
    # Remove trailing commas before object/array end, e.g. {"a":1,} or [1,2,]
    repaired = _strip_trailing_commas(repaired)
    # Fix common missing-colon cases: {"k" "v"} / {"k" { ... }} / {"k" 123}
    repaired = _insert_missing_colons(repaired)
    # Fix common missing-comma cases between adjacent members/values.
    repaired = _insert_missing_commas(repaired)
    # Fix missing value cases: {"k": } / {"k":,}
    repaired = _insert_missing_values(repaired)
    return repaired


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            continue

        if ch in "}]":
            j = len(out) - 1
            while j >= 0 and out[j].isspace():
                j -= 1
            if j >= 0 and out[j] == ",":
                del out[j]
            out.append(ch)
            continue

        out.append(ch)
    return "".join(out)


def _insert_missing_colons(text: str) -> str:
    # Only heuristic for JSON object keys; safe enough for model-generated content.
    # Match: "key" <spaces> <value-start>, where ":" is missing.
    return re.sub(
        r'("([^"\\]|\\.)*")\s+(?=(\{|\[|"|true|false|null|-?\d))',
        r"\1: ",
        text,
    )


def _insert_missing_values(text: str) -> str:
    # Replace key with empty value before comma/object end -> null
    # {"k": , ...} / {"k":}
    text = re.sub(r'("([^"\\]|\\.)*"\s*:\s*)(?=,|\})', r"\1 null", text)
    return text


def _normalize_json_punctuation(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch == "：":
            out.append(":")
            continue
        if ch == "，":
            out.append(",")
            continue
        out.append(ch)
    return "".join(out)


def _normalize_single_quoted_json(text: str) -> str:
    # {'k': 'v'} -> {"k": "v"}
    text = re.sub(r"(?<=\{|,)\s*'([^'\\]+)'\s*:", lambda m: f' "{m.group(1)}":', text)
    text = re.sub(r":\s*'([^'\\]*)'(?=\s*[,}\]])", lambda m: f': "{m.group(1)}"', text)
    return text


def _quote_bare_object_keys(text: str) -> str:
    # {foo:1, bar_baz:2} -> {"foo":1, "bar_baz":2}
    return re.sub(r'(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r' "\1":', text)


def _insert_missing_commas(text: str) -> str:
    # Insert commas between adjacent object members: {"a":1 "b":2} -> {"a":1, "b":2}
    text = re.sub(r'([}\]"0-9eE])\s+(?="([^"\\]|\\.)*"\s*:)', r"\1, ", text)
    # Insert commas between adjacent array values: [1 2] / ["a" {"b":1}]
    text = re.sub(r'([}\]"0-9eE])\s+(?=(\{|\[|"(?:[^"\\]|\\.)*"|-?\d|true|false|null))', r"\1, ", text)
    return text

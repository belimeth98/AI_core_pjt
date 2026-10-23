"""GMS 채팅 호출·환경변수·응답 완료 검사를 한곳에서 관리합니다."""
import json
import os
from openai import OpenAI


def chat_completion(messages, *, json_mode=False, max_completion_tokens=2000):
    api_key = os.getenv("GMS_API_KEY", "").strip()
    base_url = os.getenv("GMS_BASE_URL", "").strip()
    model = os.getenv("GMS_CHAT_MODEL", "").strip()
    if not all([api_key, base_url, model]):
        raise ValueError("GMS 환경변수 설정을 확인하세요.")
    options = {"response_format": {"type": "json_object"}} if json_mode else {}
    with OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=2) as client:
        response = client.chat.completions.create(
            model=model, messages=messages,
            max_completion_tokens=max_completion_tokens, **options,
        )
    if not response.choices:
        raise RuntimeError("모델 응답이 비어 있습니다.")
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise RuntimeError("모델이 생성을 거절했습니다.")
    if choice.finish_reason != "stop":
        raise RuntimeError("모델 응답이 완성되지 않았습니다.")
    content = choice.message.content
    if not content or not content.strip():
        raise RuntimeError("모델 응답 내용이 비어 있습니다.")
    return content


def request_chat(system_prompt, payload, *, json_mode=False):
    return chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], json_mode=json_mode)

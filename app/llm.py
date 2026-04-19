import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()


class LLMConfigError(Exception):
    pass


class LLMRuntimeError(Exception):
    pass


def validate_llm_config():
    missing = []

    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_DEPLOYMENT:
        missing.append("AZURE_OPENAI_DEPLOYMENT")

    if missing:
        raise LLMConfigError(f"Missing config: {', '.join(missing)}")


def get_client():
    validate_llm_config()

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def check_openai_health():
    try:
        client = get_client()
        res = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        return {"status": "ok", "response": res.choices[0].message.content}
    except Exception as e:
        raise LLMRuntimeError(str(e))


def generate_response(user_message: str) -> Dict[str, Any]:
    print("DEBUG ENDPOINT:", AZURE_OPENAI_ENDPOINT)
    print("DEBUG DEPLOYMENT:", AZURE_OPENAI_DEPLOYMENT)
    if not user_message:
        raise ValueError("Message required")

    try:
        client = get_client()

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a smart banking assistant."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=300
        )

        reply = response.choices[0].message.content

        return {
            "success": True,
            "reply": reply
        }

    except Exception as e:
        logger.exception("LLM error")
        raise LLMRuntimeError(str(e))
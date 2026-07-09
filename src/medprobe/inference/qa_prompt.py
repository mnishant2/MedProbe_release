"""Builds the 'Question / Answer / Is this medically correct?' prompt the LLM sees."""
from __future__ import annotations

from typing import Any

DEFAULT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Is this answer medically correct? Respond with Yes or No."
)


def format_qa_prompt(
    tokenizer: Any,
    question: str,
    answer: str,
    template: str = DEFAULT_TEMPLATE,
) -> str:
    body = template.format(question=question, answer=answer)
    messages = [{"role": "user", "content": body}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Tokenizer has no chat template — fall back to raw body
        return body

from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from app import config

client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


def build_prompt(question: str, chunks: List[Dict]) -> str:
    context_blocks = "\n\n".join(
        f"[{i + 1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the context below. Cite sources inline using "
        "[1], [2], etc. If the answer isn't in the context, say you don't know rather "
        "than guessing.\n\n"
        f"Context:\n{context_blocks}\n\nQuestion: {question}\nAnswer:"
    )


def generate_answer(question: str, chunks: List[Dict]) -> Tuple[str, Optional[int]]:
    prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise assistant that only answers from the provided "
                "context and always cites sources.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else None
    return answer, tokens_used

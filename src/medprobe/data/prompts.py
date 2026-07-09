"""Load register prompt YAMLs and build the final few-shot prompt string."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..paths import repo_root


@dataclass
class RegisterPrompt:
    name: str
    display: str
    vignette: str
    few_shot: list[dict[str, Any]]
    instruction: str

    @classmethod
    def load(cls, register: str, prompts_dir: Path | None = None) -> "RegisterPrompt":
        pdir = prompts_dir or (repo_root() / "configs" / "prompts")
        path = pdir / f"{register}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No prompt YAML for register {register!r} at {path}")
        raw = OmegaConf.load(path)
        return cls(
            name=str(raw.name),
            display=str(raw.display),
            vignette=str(raw.vignette).strip(),
            few_shot=OmegaConf.to_container(raw.few_shot, resolve=True),  # type: ignore[arg-type]
            instruction=str(raw.instruction).strip(),
        )

    def render(self, question: str, answer: str, correct: bool) -> str:
        """Build the full prompt for the register rewriter."""
        shots = []
        for i, ex in enumerate(self.few_shot, start=1):
            shots.append(
                f"EXAMPLE {i} ({ex['topic']}, correct={ex['correct']}):\n"
                f"  ORIGINAL QUESTION: {ex['question_in']}\n"
                f'  ORIGINAL ANSWER: {ex["answer_in"]}\n'
                f"  REWRITE:\n"
                f'  {{"question": {_json_str(ex["question_out"])}, "answer": {_json_str(ex["answer_out"])}}}'
            )
        shot_block = "\n\n".join(shots)
        return (
            f"You are helping create a medical NLP dataset for studying how clinical "
            f"language register affects AI safety tools.\n\n"
            f"TARGET REGISTER: {self.display}\n\n"
            f"REGISTER STYLE GUIDE:\n{self.vignette}\n\n"
            f"{shot_block}\n\n"
            f"--- YOUR TURN ---\n"
            f"ORIGINAL QUESTION: {question}\n"
            f"ORIGINAL ANSWER: {answer}\n"
            f"CORRECT: {correct}\n\n"
            f"{self.instruction}\n\nOUTPUT:\n"
        )


def _json_str(s: str) -> str:
    import json as _json

    return _json.dumps(s)

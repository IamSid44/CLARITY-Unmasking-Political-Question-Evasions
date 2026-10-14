"""M3's prompt: a fixed preamble (task, definitions, confusion guide), retrieved examples, the item, lettered candidates.

The preamble is identical for every item, so vLLM's prefix cache and the API's cached-input rate apply to it.
Definitions follow the taxonomy of Thomas et al. (2024); the confusion guide covers the largest error pairs of
M1 among deferred cross-fitted TRAIN rows (E15 registration), and was written without looking at dev.
"""

from __future__ import annotations

from dataclasses import dataclass

from qevasion.labels import EVASION_LABELS

PROMPT_VERSION = "m3-v1"
LETTERS = "ABCDEFGHI"
MAX_PROMPT_TOKENS = 12_000
CHARS_PER_TOKEN = 3.6  # conservative for English; only used for the 12k cap and cost estimates

DEFINITIONS: dict[str, str] = {
    "Explicit": "The requested information is stated explicitly, in the terms the question asks for.",
    "Implicit": "The requested information is given, but not stated explicitly: it has to be inferred from "
                "what the answer says.",
    "Dodging": "The answer ignores the question altogether and does not engage with what was asked.",
    "General": "The answer addresses the question but is too general: it lacks the specificity the question "
               "asks for, so the requested information cannot be recovered from it.",
    "Deflection": "The answer starts on the question's topic but shifts the focus and makes a different point "
                  "from the one asked.",
    "Partial/half-answer": "The answer gives only one component of the requested information and leaves the "
                           "rest unanswered.",
    "Declining to answer": "The speaker acknowledges the question but refuses, directly or indirectly, to "
                           "answer it.",
    "Claims ignorance": "The speaker says they do not know the answer.",
    "Clarification": "The speaker does not answer and instead asks for the question to be clarified.",
}

CONFUSION_GUIDE: tuple[str, ...] = (
    "Explicit vs Implicit: choose Explicit only if some sentence of the answer directly gives the requested "
    "information (a yes/no, a position, a number, a name). If the information follows from what is said but is "
    "never stated, choose Implicit.",
    "Implicit vs General: Implicit lets a listener infer the specific information asked for. General stays at "
    "the level of values, goals or broad statements from which the specific answer cannot be inferred.",
    "Explicit vs General: a clear, on-topic statement that still lacks the specificity asked for (for example "
    "'we will do whatever it takes' when asked how many troops) is General, not Explicit.",
    "Deflection vs Implicit/General: in a Deflection the answer moves on to another point (an opponent, a "
    "different policy, past achievements) and the requested information is not provided. If the requested "
    "information can still be inferred, prefer Implicit.",
    "Dodging vs Deflection: Dodging never engages the question's topic; Deflection engages it first and then "
    "moves to a different point.",
)


def preamble() -> str:
    defs = "\n".join(f"- {name}: {DEFINITIONS[name]}" for name in EVASION_LABELS)
    guide = "\n".join(f"- {rule}" for rule in CONFUSION_GUIDE)
    return (
        "You are an expert annotator of political interviews. A journalist's turn may contain several "
        "questions; it has been split into sub-questions, and each sub-question is paired with the "
        "politician's full answer. Your task is to label how the answer responds to ONE given sub-question. "
        "The same answer can respond explicitly to one sub-question and evade another, so judge only the "
        "given sub-question.\n\n"
        f"Labels:\n{defs}\n\n"
        f"How to separate the labels that are most often confused:\n{guide}\n\n"
        "You will be shown labelled examples from other interviews, then the item to label and a short list "
        "of allowed labels, each with a letter. Reply with the letter of the correct label only."
    )


@dataclass(frozen=True)
class Item:
    sub_question: str
    full_question: str
    answer: str


def item_from_row(row) -> Item:
    return Item(str(row["question"]).strip(), str(row["interview_question"]).strip(),
                str(row["interview_answer"]).strip())


def render_item(it: Item) -> str:
    return f"Sub-question: {it.sub_question}\nFull question: {it.full_question}\nAnswer: {it.answer}"


def approx_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def build_messages(item: Item, cand: list[int], examples: list[tuple[Item, int]]) -> tuple[list[dict], list[int]]:
    """Chat messages for one item.

    `cand` are label indices (any order; shown in taxonomy order with letters A, B, ...). `examples` are
    (item, gold label index) pairs, most similar LAST. Examples are dropped from the least similar end
    (the front) until the prompt fits MAX_PROMPT_TOKENS; none is ever cut. Returns (messages, kept example
    positions in `examples`).
    """
    cand = sorted(cand)
    options = "\n".join(f"{LETTERS[j]}. {EVASION_LABELS[c]}" for j, c in enumerate(cand))
    tail = (f"Item to label:\n{render_item(item)}\n\n"
            f"Which label describes how the answer responds to the sub-question? Allowed labels:\n{options}\n\n"
            f"Reply with one letter ({', '.join(LETTERS[: len(cand)])}).")
    blocks = [f"Example {j + 1}:\n{render_item(ex)}\nLabel: {EVASION_LABELS[y]}" for j, (ex, y) in enumerate(examples)]
    keep = list(range(len(examples)))
    sys_msg = preamble()
    while keep and approx_tokens(sys_msg) + sum(approx_tokens(blocks[j]) for j in keep) + approx_tokens(tail) > MAX_PROMPT_TOKENS:
        keep.pop(0)
    if keep:
        shown = [f"Example {n + 1}:" + blocks[j].split(":", 1)[1] for n, j in enumerate(keep)]
        user = "Labelled examples from other interviews:\n\n" + "\n\n".join(shown) + "\n\n" + tail
    else:
        user = tail
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], keep


def letter_map(cand: list[int]) -> dict[str, int]:
    """Option letter -> label index, in the order build_messages shows them."""
    return {LETTERS[j]: c for j, c in enumerate(sorted(cand))}

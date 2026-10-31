#!/usr/bin/env python3
"""Curate the Weights & Biases project: tags, notes and unambiguous names. Nothing is deleted.

    python -m utils.wandb_annotate            # dry run: prints what would change
    python -m utils.wandb_annotate --apply    # writes tags, notes and names to W&B

Why: the runs were logged as `<config>-s<seed>` in a group named after the config, which is exact but cryptic, and
restarted runs left several runs with the same name. This script
  * tags every run with its stage and experiment id and, for the final system, `final-system`;
  * writes a note with what the run is and links to its Hugging Face folder, the code and the experiment log;
  * keeps the most complete run of each (group, name) as is, and renames earlier attempts and crashed runs
    to `<name> [attempt k, superseded]` or `<name> [crashed]`, with a note pointing to the complete run.
Group names are left unchanged because they match the folders on Hugging Face and the names in the docs.
"""

from __future__ import annotations

import argparse
import collections
import os
import re

HF = "https://huggingface.co/siddarthg44/clarity-semeval26"
GH = "https://github.com/IamSid44/CLARITY-Unmasking-Political-Question-Evasions"

# group -> (stage tag, experiment id, role tag or None, description)
GROUPS = {
    "L0_large_base": ("stage-1-encoder", "E0", "deberta", "DeBERTa-v3-large baseline (sub-question and answer as input)."),
    "F0_oof": ("stage-1-encoder", "E6", "deberta", "Five-fold out-of-fold DeBERTa-v3-large models, the folds used by the E6 re-ranker."),
    "R1_rerank": ("stage-1-encoder", "E6", "deberta", "E6: DeBERTa re-ranker over the top-3 labels given the label definitions."),
    "E8_fullq_bal_focal": ("stage-1-encoder", "E8", "deberta", "E8: Balanced Softmax plus focal loss, full-question input."),
    "E8b_fullq_ce": ("stage-1-encoder", "E8", "deberta", "E8b: plain cross-entropy control for E8, full-question input."),
    "E9_hier": ("stage-1-encoder", "E9", "deberta", "E9: factorised hierarchical output head."),
    "E10_base_16ep": ("stage-1-encoder", "E10", "deberta", "E10 control: sub-question and answer input, 16 epochs."),
    "E10_fullq_16ep": ("stage-1-encoder", "E10", "deberta", "E10: full-question input, 16 epochs."),
    "E11_base_16ep": ("stage-1-encoder", "E11", "deberta", "E11 fresh-seed replication: sub-question and answer input, 16 epochs."),
    "E11_base_8ep": ("stage-1-encoder", "E11", "deberta", "E11 fresh-seed replication: sub-question and answer input, 8 epochs."),
    "E11_fullq_16ep": ("stage-1-encoder", "E11", "deberta", "E11 fresh-seed replication: full-question input, 16 epochs."),
    "E12a_alldata": ("stage-1-encoder", "E12", "deberta", "E12a: full question, 16 epochs, trained on all of train."),
    "E12b_gate": ("stage-1-encoder", "E12", "deberta", "E12b: binary Non-Reply gate, trained on all data."),
    "E12b_nr3": ("stage-1-encoder", "E12", "deberta", "E12b: three-way specialist on the Non-Reply rows."),
    "E12b_other6": ("stage-1-encoder", "E12", "deberta", "E12b: six-way specialist on the rows that are not Non-Reply."),
    "E12c_pair_expl_impl": ("stage-1-encoder", "E12", "deberta", "E12c: Explicit against Implicit pair expert."),
    "E12c_pair_gen_defl": ("stage-1-encoder", "E12", "deberta", "E12c: General against Deflection pair expert."),
    "E12c_pair_impl_dodg": ("stage-1-encoder", "E12", "deberta", "E12c: Implicit against Dodging pair expert."),
    "Q8_fullq_lora": ("stage-1-encoder", "E13", "qwen3-8b", "E13: Qwen3-8B-Base with LoRA, full-question input, 3 epochs."),
    "Q8_alldata": ("stage-1-encoder", "E13", "qwen3-8b", "E13c: Qwen3-8B-Base with LoRA trained on all of train, last epoch."),
    "Q8_fullq_lora_12ep": ("stage-2-cascade", "E14", "M1", "M1: Qwen3-8B-Base with LoRA, full question, 12 epochs. Ten seeds form the classifier of the final system."),
    "Q8_12ep_crossfit": ("stage-2-cascade", "E14", "M1", "Cross-fitted M1 models (five folds of train), used for candidate sets and distillation targets."),
    "Q8_distill_a05": ("stage-2-cascade", "E17", "M4", "M4: student trained on half gold and half M3 soft targets."),
    "Q8_distill_a05_m1": ("stage-3-controls", "E19", "M4-control", "Teacher control: student taught by M1's own cross-fitted probabilities."),
    "Q8_distill_a05_unif": ("stage-3-controls", "E19", "M4-control", "Teacher control: student taught by a uniform distribution over the candidates."),
    "cascade": ("stage-2-cascade", "E17", None, "Analysis run that logs the cascade results."),
}
FINAL = {"Q8_fullq_lora_12ep", "Q8_distill_a05"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    from utils.tracking import load_env
    load_env()
    import wandb

    api = wandb.Api()
    path = f"{api.default_entity}/{os.environ.get('WANDB_PROJECT', 'clarity-semeval26')}"
    runs = list(api.runs(path))
    by = collections.defaultdict(list)
    for r in runs:
        by[(r.group, re.sub(r" \[.*\]$", "", r.name))].append(r)  # names already annotated by an earlier pass
    plan = []
    for (group, name), rs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        rs.sort(key=lambda r: (r.state == "finished", r.lastHistoryStep, r.created_at))
        best = rs[-1]
        for k, r in enumerate(rs):
            stage, exp, role, desc = GROUPS.get(group, ("other", None, None, f"Run of group {group}."))
            seed = re.search(r"-(s\d+|fold\d+)$", name)
            if group == "cascade":
                exp = (re.match(r"(E\d+)", name) or [None, exp])[1]
            tags = [stage] + ([exp] if exp else []) + ([role] if role else [])
            if group in FINAL:
                tags.append("final-system")
            new_name, note = name, desc
            if r is not best or r.state != "finished":
                tag = "crashed" if r.state != "finished" else "superseded"
                tags.append(tag)
                new_name = f"{name} [{'crashed' if r.state != 'finished' else f'attempt {k + 1}, superseded'}]"
                note += f" Earlier attempt, stopped before the end or restarted. The complete run is {best.id}."
            elif seed and group not in ("cascade", "F0_oof"):
                folder = f"seed{seed.group(1)[1:]}" if seed.group(1).startswith("s") else seed.group(1)
                note += f" Weights, probabilities and metrics: {HF}/tree/main/{group}/{folder}."
            note += f" Code and experiment log: {GH} (docs/experiment_log.md)."
            plan.append((r, new_name, tags, note))
    changed = 0
    for r, new_name, tags, note in plan:
        if (r.name, sorted(r.tags), r.notes) == (new_name, sorted(tags), note):
            continue
        changed += 1
        print(f"{r.id}  {r.name!r} -> {new_name!r}  tags={tags}")
        if a.apply:
            r.name, r.tags, r.notes = new_name, tags, note
            r.update()
    print(f"{len(runs)} runs, {changed} {'updated' if a.apply else 'to update'}")


if __name__ == "__main__":
    main()

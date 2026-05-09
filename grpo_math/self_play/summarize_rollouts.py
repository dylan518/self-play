from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class _Stats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = math.inf
    max: float = -math.inf

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std(),
            "min": (None if self.min == math.inf else self.min),
            "max": (None if self.max == -math.inf else self.max),
        }


def _fmt_float(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}"


def _get_nested(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _iter_floats(xs: Iterable[Any]) -> Iterable[float]:
    for x in xs:
        if x is None:
            continue
        try:
            yield float(x)
        except Exception:
            continue


def _oracle_pairwise_preference_accuracy(row: dict[str, Any]) -> dict[str, Any]:
    """
    For pairwise comparisons where exactly one solution is oracle_correct==True and the other is False,
    compute how often the judge's preference picks the oracle-correct solution.
    """
    sols = row.get("solutions")
    pairs = row.get("pairwise_comparisons")
    if not isinstance(sols, list) or not isinstance(pairs, list):
        return {
            "acc": None,
            "correct_votes": 0,
            "total_votes": 0,
            "informative_pairs": 0,
        }

    oracle_by_idx: dict[int, bool] = {}
    for s in sols:
        if not isinstance(s, dict):
            continue
        oc = s.get("oracle_correct")
        if not isinstance(oc, bool):
            continue
        try:
            idx = int(s.get("solution_index", -1))
        except Exception:
            continue
        if idx >= 0:
            oracle_by_idx[idx] = oc

    correct_votes = 0
    total_votes = 0
    informative_pairs = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        try:
            i = int(pair.get("i"))
            j = int(pair.get("j"))
        except Exception:
            continue
        if i not in oracle_by_idx or j not in oracle_by_idx:
            continue
        oi = oracle_by_idx[i]
        oj = oracle_by_idx[j]
        if oi == oj:
            continue  # not informative
        informative_pairs += 1
        correct_idx = i if oi else j
        prefs = pair.get("prefs", [])
        if not isinstance(prefs, list):
            continue
        for pref in prefs:
            if pref == "A":
                winner = i
            elif pref == "B":
                winner = j
            else:
                continue
            total_votes += 1
            if winner == correct_idx:
                correct_votes += 1

    acc = (correct_votes / total_votes) if total_votes > 0 else None
    return {
        "acc": acc,
        "correct_votes": correct_votes,
        "total_votes": total_votes,
        "informative_pairs": informative_pairs,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rsep_elo = _Stats()
    rsep_win = _Stats()
    pref_stab = _Stats()
    num_pairs = _Stats()

    group0_elo = _Stats()
    group1_elo = _Stats()

    oracle_any_solution_acc = _Stats()
    oracle_group0_acc = _Stats()
    oracle_group1_acc = _Stats()
    oracle_best_by_elo_acc = _Stats()
    oracle_pref_acc_macro = _Stats()
    oracle_pref_informative_pairs = _Stats()
    oracle_pref_votes_total = 0
    oracle_pref_votes_correct = 0

    used_rows = 0
    for row in rows:
        rel = row.get("reliability", {}) if isinstance(row.get("reliability"), dict) else {}
        used_rows += 1

        pref_stab.update(float(rel.get("preference_stability", 0.0)))
        try:
            num_pairs.update(float(rel.get("num_pairs", 0)))
        except Exception:
            pass

        rsep_elo_val = rel.get("r_sep_elo")
        if rsep_elo_val is not None:
            rsep_elo.update(float(rsep_elo_val))

        rsep_win_val = rel.get("r_sep_pairwise_winrate")
        if rsep_win_val is not None:
            rsep_win.update(float(rsep_win_val))

        group_means = rel.get("group_elo_means")
        if isinstance(group_means, list) and len(group_means) >= 2:
            g0, g1 = group_means[0], group_means[1]
            for x in _iter_floats([g0]):
                group0_elo.update(x)
            for x in _iter_floats([g1]):
                group1_elo.update(x)

        # Oracle accuracy (only present when strong_verifier.enabled=true during rollout).
        sols = row.get("solutions")
        if isinstance(sols, list) and sols:
            oracle_correct_vals: list[tuple[int, bool]] = []
            group0_vals: list[bool] = []
            group1_vals: list[bool] = []
            best_by_elo: tuple[float, bool] | None = None
            for s in sols:
                if not isinstance(s, dict):
                    continue
                oc = s.get("oracle_correct")
                if not isinstance(oc, bool):
                    continue
                grp = s.get("sampling_group")
                try:
                    grp_i = int(grp)
                except Exception:
                    grp_i = -1
                oracle_correct_vals.append((grp_i, oc))
                if grp_i == 0:
                    group0_vals.append(oc)
                if grp_i == 1:
                    group1_vals.append(oc)
                try:
                    elo = float(s.get("elo_rating", 0.0))
                except Exception:
                    elo = 0.0
                if best_by_elo is None or elo > best_by_elo[0]:
                    best_by_elo = (elo, oc)

            if oracle_correct_vals:
                any_ok = any(ok for _, ok in oracle_correct_vals)
                oracle_any_solution_acc.update(1.0 if any_ok else 0.0)
            if group0_vals:
                oracle_group0_acc.update(sum(1.0 for x in group0_vals if x) / len(group0_vals))
            if group1_vals:
                oracle_group1_acc.update(sum(1.0 for x in group1_vals if x) / len(group1_vals))
            if best_by_elo is not None:
                oracle_best_by_elo_acc.update(1.0 if best_by_elo[1] else 0.0)

        pref_acc = _oracle_pairwise_preference_accuracy(row)
        if isinstance(pref_acc.get("acc"), float):
            oracle_pref_acc_macro.update(float(pref_acc["acc"]))
        if pref_acc.get("informative_pairs") is not None:
            try:
                oracle_pref_informative_pairs.update(float(pref_acc["informative_pairs"]))
            except Exception:
                pass
        oracle_pref_votes_total += int(pref_acc.get("total_votes", 0) or 0)
        oracle_pref_votes_correct += int(pref_acc.get("correct_votes", 0) or 0)

    return {
        "rows": used_rows,
        "preference_stability": pref_stab.as_dict(),
        "num_pairs": num_pairs.as_dict(),
        "r_sep_elo": rsep_elo.as_dict(),
        "r_sep_pairwise_winrate": rsep_win.as_dict(),
        "group0_elo_mean": group0_elo.as_dict(),
        "group1_elo_mean": group1_elo.as_dict(),
        "oracle_any_solution_acc": oracle_any_solution_acc.as_dict(),
        "oracle_group0_acc": oracle_group0_acc.as_dict(),
        "oracle_group1_acc": oracle_group1_acc.as_dict(),
        "oracle_best_by_elo_acc": oracle_best_by_elo_acc.as_dict(),
        "oracle_pref_acc_macro": oracle_pref_acc_macro.as_dict(),
        "oracle_pref_informative_pairs": oracle_pref_informative_pairs.as_dict(),
        "oracle_pref_acc_micro": (
            (oracle_pref_votes_correct / oracle_pref_votes_total) if oracle_pref_votes_total > 0 else None
        ),
        "oracle_pref_votes_total": oracle_pref_votes_total,
        "oracle_pref_votes_correct": oracle_pref_votes_correct,
    }


def _build_report_md(
    *,
    rows: list[dict[str, Any]],
    jsonl_path: Path,
    config_path: Path | None,
    readme_export_dir: Path | None,
) -> str:
    summary = _summarize(rows)
    run_id = rows[0].get("run_id") if rows else None
    method = _get_nested(rows[0], "ranking", "method") if rows else None

    out: list[str] = []
    out.append("# Pairwise Rollout Report")
    out.append("")
    out.append(f"- JSONL: `{jsonl_path}`")
    if config_path is not None:
        out.append(f"- Config: `{config_path}`")
    if readme_export_dir is not None:
        out.append(f"- READMEs: `{readme_export_dir}`")
    if run_id:
        out.append(f"- run_id: `{run_id}`")
    if method:
        out.append(f"- ranking.method: `{method}`")
    out.append("")

    def _add_stats_block(title: str, d: dict[str, Any], nd: int = 3) -> None:
        out.append(f"## {title}")
        out.append("")
        out.append(
            f"- n: `{d.get('n')}`  "
            f"mean: `{_fmt_float(d.get('mean'), nd)}`  "
            f"std: `{_fmt_float(d.get('std'), nd)}`  "
            f"min: `{_fmt_float(d.get('min'), nd)}`  "
            f"max: `{_fmt_float(d.get('max'), nd)}`"
        )
        out.append("")

    _add_stats_block("Preference stability", summary["preference_stability"], nd=3)
    _add_stats_block("Pairs per question", summary["num_pairs"], nd=0)
    _add_stats_block("R_sep (Elo)", summary["r_sep_elo"], nd=2)
    _add_stats_block("R_sep (cross-group win-rate)", summary["r_sep_pairwise_winrate"], nd=3)
    _add_stats_block("Group 0 mean Elo", summary["group0_elo_mean"], nd=2)
    _add_stats_block("Group 1 mean Elo", summary["group1_elo_mean"], nd=2)
    _add_stats_block("Oracle acc: any solution correct", summary["oracle_any_solution_acc"], nd=3)
    _add_stats_block("Oracle acc: group 0 (mean)", summary["oracle_group0_acc"], nd=3)
    _add_stats_block("Oracle acc: group 1 (mean)", summary["oracle_group1_acc"], nd=3)
    _add_stats_block("Oracle acc: best-by-Elo correct", summary["oracle_best_by_elo_acc"], nd=3)
    _add_stats_block("Oracle pref acc (macro over questions)", summary["oracle_pref_acc_macro"], nd=3)
    _add_stats_block("Oracle-informative pairs per question", summary["oracle_pref_informative_pairs"], nd=2)
    out.append("## Oracle preference accuracy (micro)")
    out.append("")
    out.append(
        f"- acc_micro: `{_fmt_float(summary.get('oracle_pref_acc_micro'), 3)}`  "
        f"correct_votes: `{summary.get('oracle_pref_votes_correct')}`  "
        f"total_votes: `{summary.get('oracle_pref_votes_total')}`"
    )
    out.append("")

    out.append("## Per-question breakdown")
    out.append("")
    out.append(
        "| q_idx | r_sep_elo | g0_elo | g1_elo | winrate_g0_vs_g1 | pref_stability | num_pairs | "
        "oracle_any_ok | oracle_best_by_elo_ok | oracle_g0_acc | oracle_g1_acc | oracle_pref_acc | oracle_pref_votes |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        qi = row.get("question_index")
        rel = row.get("reliability", {}) if isinstance(row.get("reliability"), dict) else {}
        gmeans = rel.get("group_elo_means") if isinstance(rel.get("group_elo_means"), list) else []
        g0 = gmeans[0] if len(gmeans) >= 1 else None
        g1 = gmeans[1] if len(gmeans) >= 2 else None

        sols = row.get("solutions") if isinstance(row.get("solutions"), list) else []
        oracle_any_ok: bool | None = None
        oracle_best_by_elo_ok: bool | None = None
        oracle_g0_acc: float | None = None
        oracle_g1_acc: float | None = None
        if sols:
            vals: list[tuple[int, bool]] = []
            best: tuple[float, bool] | None = None
            g0_vals: list[bool] = []
            g1_vals: list[bool] = []
            for s in sols:
                if not isinstance(s, dict):
                    continue
                oc = s.get("oracle_correct")
                if not isinstance(oc, bool):
                    continue
                try:
                    grp_i = int(s.get("sampling_group", -1))
                except Exception:
                    grp_i = -1
                vals.append((grp_i, oc))
                if grp_i == 0:
                    g0_vals.append(oc)
                if grp_i == 1:
                    g1_vals.append(oc)
                try:
                    elo = float(s.get("elo_rating", 0.0))
                except Exception:
                    elo = 0.0
                if best is None or elo > best[0]:
                    best = (elo, oc)
            if vals:
                oracle_any_ok = any(ok for _, ok in vals)
            if best is not None:
                oracle_best_by_elo_ok = bool(best[1])
            if g0_vals:
                oracle_g0_acc = sum(1.0 for x in g0_vals if x) / len(g0_vals)
            if g1_vals:
                oracle_g1_acc = sum(1.0 for x in g1_vals if x) / len(g1_vals)

        pref_acc = _oracle_pairwise_preference_accuracy(row)
        pref_acc_val = pref_acc.get("acc")
        pref_votes = pref_acc.get("total_votes", 0)

        out.append(
            "| "
            + " | ".join(
                [
                    str(qi),
                    _fmt_float(rel.get("r_sep_elo"), 2),
                    _fmt_float(g0, 2),
                    _fmt_float(g1, 2),
                    _fmt_float(rel.get("r_sep_pairwise_winrate"), 3),
                    _fmt_float(rel.get("preference_stability"), 3),
                    str(rel.get("num_pairs")),
                    ("NA" if oracle_any_ok is None else ("1" if oracle_any_ok else "0")),
                    ("NA" if oracle_best_by_elo_ok is None else ("1" if oracle_best_by_elo_ok else "0")),
                    _fmt_float(oracle_g0_acc, 3),
                    _fmt_float(oracle_g1_acc, 3),
                    _fmt_float(pref_acc_val if isinstance(pref_acc_val, float) else None, 3),
                    str(pref_votes),
                ]
            )
            + " |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to rollout JSONL.")
    ap.add_argument("--out_md", type=str, required=True, help="Path to write markdown report.")
    ap.add_argument("--config", type=str, default="", help="Optional config YAML path to include in report.")
    ap.add_argument(
        "--readme_export_dir",
        type=str,
        default="",
        help="Optional README export dir (to link in report).",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    rows = _read_jsonl(jsonl_path)
    config_path = Path(args.config).resolve() if args.config else None
    readme_export_dir = Path(args.readme_export_dir).resolve() if args.readme_export_dir else None

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        _build_report_md(
            rows=rows,
            jsonl_path=jsonl_path.resolve(),
            config_path=config_path,
            readme_export_dir=readme_export_dir,
        ),
        encoding="utf-8",
    )
    print(f"Wrote report to: {out_md}")


if __name__ == "__main__":
    main()


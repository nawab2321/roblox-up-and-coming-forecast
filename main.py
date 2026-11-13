#!/usr/bin/env python3
"""Forecast Roblox Up-and-Coming chart places using historical transitions."""

import datetime as dt
import json
import math
import os
from pathlib import Path


PLACES = 10
UNKNOWN_RANK = 110.0


def parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def rows(record, key):
    value = record.get(key)
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def valid_universe_id(value):
    # JSON booleans are Python integers, so use an exact type check.
    return type(value) is int and value > 0


def detail_map(record):
    value = record.get("details")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(x.get("universe_id")): x for x in value if isinstance(x, dict)}
    return {}


def snapshot_features(history):
    """Extract per-game signals available at each read, without assuming fields exist."""
    first_seen = {}
    prior = {}
    snapshots = []
    for record in history:
        at = parse_time(record.get("at"))
        chart = rows(record, "up_and_coming")
        details = detail_map(record)
        trending = {str(g.get("universe_id")): g.get("rank") for g in rows(record, "top_trending")}
        features = {}
        for fallback_rank, game in enumerate(chart, 1):
            uid = game.get("universe_id")
            if not valid_universe_id(uid):
                continue
            key = str(uid)
            # Duplicate rows in one chart read must not become fake zero-hour
            # transitions for this game's momentum and player-count trend.
            if key in features:
                continue
            rank = number(game.get("rank")) or float(fallback_rank)
            old = prior.get(key)
            elapsed = None
            momentum = None
            player_growth = None
            if old and at and old["at"]:
                elapsed = max(0.01, (at - old["at"]).total_seconds() / 3600.0)
                # Positive momentum means the game has been climbing (rank number falling).
                momentum = max(-18.0, min(18.0, (old["rank"] - rank) * 12.0 / elapsed))
                now_players, old_players = number(game.get("playing")), old["playing"]
                if now_players is not None and old_players is not None:
                    now_players, old_players = max(0.0, now_players), max(0.0, old_players)
                    player_growth = max(-2.0, min(2.0, math.log((now_players + 1) / (old_players + 1)) * 12.0 / elapsed))
            if key not in first_seen:
                first_seen[key] = at
            detail = details.get(key, {}) if isinstance(details, dict) else {}
            if not isinstance(detail, dict):
                detail = {}
            created = parse_time(detail.get("created"))
            age_days = max(0.0, (at - created).total_seconds() / 86400.0) if at and created else None
            if age_days is None and at and first_seen.get(key):
                # Chart first-seen age is a conservative lower bound, not a hard-coded date.
                age_days = max(0.0, (at - first_seen[key]).total_seconds() / 86400.0)
            up, down = number(game.get("up_votes")), number(game.get("down_votes"))
            denominator = up + down + 2.0 if up is not None and down is not None else None
            approval = (up + 1.0) / denominator if denominator is not None and denominator > 0 else None
            players = number(game.get("playing"))
            features[key] = {
                "rank": rank,
                "momentum": momentum,
                "players": math.log1p(max(0.0, players)) if players is not None else None,
                "player_growth": player_growth,
                "approval": approval,
                "age": age_days,
                "genre": game.get("genre") or detail.get("genre"),
                "trending_rank": number(trending.get(key)),
                "universe_id": uid,
            }
            prior[key] = {"rank": rank, "at": at, "playing": players}
        snapshots.append({"at": at, "record": record, "features": features})
    return snapshots


def similarity(a, b):
    """Smooth feature kernel. Missing optional fields simply contribute no penalty."""
    weight = 1.0
    specs = (("rank", 11.0), ("momentum", 7.0), ("players", 2.2),
             ("player_growth", 1.1), ("approval", 0.16), ("age", 8.0))
    for field, scale in specs:
        x, y = a.get(field), b.get(field)
        if x is not None and y is not None:
            weight *= math.exp(-abs(x - y) / scale)
    ga, gb = a.get("genre"), b.get("genre")
    if ga and gb and ga != gb:
        weight *= 0.9
    # Presence on Top Trending is a small useful boost, while its absence is not a penalty.
    if b.get("trending_rank") and not a.get("trending_rank"):
        weight *= 0.96
    return weight


def make_examples(snapshots, cutoff=None):
    """Build 20–60 hour rank transitions, matching the multi-day scored window."""
    examples = []
    for i, source in enumerate(snapshots):
        if not source["at"] or (cutoff and source["at"] >= cutoff):
            continue
        targets = []
        for target in snapshots[i + 1:]:
            if not target["at"]:
                continue
            hours = (target["at"] - source["at"]).total_seconds() / 3600.0
            if hours > 60.0:
                break
            if 20.0 <= hours <= 60.0:
                targets.append((target, hours))
        if not targets:
            continue
        # Choose one representative future read per source to avoid over-weighting dense periods.
        target, hours = min(targets, key=lambda pair: abs(pair[1] - 38.0))
        dest = target["features"]
        for key, feature in source["features"].items():
            future = dest.get(key)
            outcome = future["rank"] if future else UNKNOWN_RANK
            examples.append((feature, outcome, target["at"]))
    return examples


def predict_distribution(feature, examples):
    if not examples:
        # Safe fallback for very short or malformed histories.
        center = feature.get("rank", 50.0)
        return [math.exp(-abs(place - center) / 4.0) for place in range(1, PLACES + 1)]
    weighted = []
    for old, outcome, _ in examples:
        w = similarity(feature, old)
        if w > 1e-12:
            weighted.append((w, outcome))
    weighted.sort(key=lambda item: item[0], reverse=True)
    weighted = weighted[:500]
    if not weighted:
        weighted = [(1.0, UNKNOWN_RANK)]
    # Each historic transition contributes a soft rank kernel. Exits and lower chart rows
    # naturally contribute almost no mass to the top ten.
    probs = []
    for place in range(1, PLACES + 1):
        mass = sum(w * math.exp(-abs(outcome - place) / 2.15) for w, outcome in weighted)
        probs.append(mass)
    return probs


def forecast(history, places=PLACES):
    snapshots = snapshot_features(history)
    all_ids = {}
    for snapshot in snapshots:
        for key, feature in snapshot["features"].items():
            all_ids[key] = feature
    if len(all_ids) < PLACES:
        raise ValueError(
            f"cannot produce the required top 10: only {len(all_ids)} distinct valid universe IDs "
            f"exist across the usable history; need at least {PLACES}"
        )
    current = snapshots[-1]["features"] if snapshots else {}
    examples = make_examples(snapshots)
    raw = {key: predict_distribution(feature, examples) for key, feature in current.items()}

    probabilities = []
    # Reserve some probability for newly published games that cannot be named from history.
    named_share = 0.86
    for place in range(1, places + 1):
        total = sum(v[place - 1] for v in raw.values())
        if total <= 0:
            continue
        place_rows = []
        for key, feature in current.items():
            p = named_share * raw[key][place - 1] / total
            if p >= 0.0005:
                place_rows.append({"universe_id": feature["universe_id"], "place": place, "p": round(p, 8)})
        # The rules cap each place at 100 named rows.
        place_rows.sort(key=lambda row: (-row["p"], row["universe_id"]))
        probabilities.extend(place_rows[:100])

    # Use full observed transition outcomes for the ordered top-10 guess. The place
    # probabilities above intentionally focus on places 1–10, but conditioning this
    # ranking guess on those ten places would reward games that often leave the chart.
    expected = []
    ranking_pool = dict(current)
    if len(ranking_pool) < PLACES:
        # Only bring older games into the ordered guess when the last read cannot
        # supply ten candidates. The probability forecast remains based on the last chart.
        ranking_pool.update({key: feature for key, feature in all_ids.items() if key not in ranking_pool})
    for key, feature in ranking_pool.items():
        weighted = [(similarity(feature, old), outcome) for old, outcome, _ in examples]
        weighted = [(w, outcome) for w, outcome in weighted if w > 1e-12]
        weighted.sort(key=lambda item: item[0], reverse=True)
        weighted = weighted[:500]
        z = sum(w for w, _ in weighted)
        transition_rank = sum(w * outcome for w, outcome in weighted) / z if z else feature["rank"]
        # Retain some current-rank inertia and modestly extend recent momentum.
        score = 0.78 * transition_rank + 0.22 * feature["rank"] - 0.18 * (feature.get("momentum") or 0.0)
        expected.append((score, feature["rank"], feature["universe_id"]))
    expected.sort()
    top = [x[2] for x in expected[:10]]
    result = {"probabilities": probabilities, "top10": top}
    validate_output(result, places)
    return result


def validate_output(result, places):
    if set(result) != {"probabilities", "top10"}:
        raise ValueError("internal error: forecast has invalid top-level keys")
    top = result["top10"]
    if len(top) != PLACES or len(set(top)) != PLACES or any(not valid_universe_id(uid) for uid in top):
        raise ValueError("internal error: forecast must contain exactly ten distinct valid universe IDs")
    if not isinstance(result["probabilities"], list):
        raise ValueError("internal error: probabilities must be a list")
    sums = [0.0] * places
    counts = [0] * places
    seen = set()
    for row in result["probabilities"]:
        if set(row) != {"universe_id", "place", "p"}:
            raise ValueError("internal error: probability row has invalid keys")
        uid, place, p = row["universe_id"], row["place"], row["p"]
        if not valid_universe_id(uid) or type(place) is not int or not 1 <= place <= places:
            raise ValueError("internal error: probability row has invalid universe_id or place")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("internal error: probability must be a finite number from 0 to 1")
        key = (uid, place)
        if key in seen:
            raise ValueError("internal error: duplicate game/place probability row")
        seen.add(key)
        sums[place - 1] += p
        counts[place - 1] += 1
    if any(total > 1.0 + 1e-12 for total in sums) or any(count > 100 for count in counts):
        raise ValueError("internal error: probability place sum or row count violates the forecast schema")


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    task_path = input_dir / "task.json"
    history_path = input_dir / "history.ndjson"
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read valid task.json at {task_path}: {exc}")
    if not isinstance(task, dict):
        raise SystemExit("error: task.json must contain a JSON object")
    task_places = task.get("places")
    if type(task_places) is not int or task_places != PLACES:
        raise SystemExit(f"error: task.json places must be {PLACES}, got {task.get('places')!r}")
    history = []
    try:
        with history_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict) and record.get("ok", True):
                            history.append(record)
                    except json.JSONDecodeError:
                        continue
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"error: cannot read history.ndjson at {history_path}: {exc}")
    try:
        result = forecast(history, PLACES)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "forecast.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

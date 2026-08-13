"""Durable, fail-closed API-attempt budgets for Phase-2 annotation.

Each attempt is reserved in an append-only journal *before* network I/O.  A
reservation is never refunded: if the process dies during a request, the
attempt still counts.  Replaying the journal therefore preserves both the
global and per-chunk ceilings across restarts.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "ph2-annotation-attempt-guard-2"
JOURNAL_SCHEMA_VERSION = "ph2-annotation-attempt-reservation-1"
MAX_PREAPPROVED_SPEND_USD = 275.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Hard project ceiling on the *theoretical* maximum charge of a single call.
#: The 2026-08-11 stop happened because the operational output allowance
#: (4,000 tokens) was chosen from an *expected* echo length, so no combination
#: of declared parameters actually bounded a call at $0.05.  Everything below
#: derives the bound from the frozen rate card instead of from an estimate.
MAX_COST_PER_ATTEMPT_HARD_USD = 0.05


class AnnotationAttemptLimitError(RuntimeError):
    """Raised before network I/O when an annotation attempt is not allowed."""


class AnnotationRetryLimitError(AnnotationAttemptLimitError):
    """This scope may remain unresolved, but untouched scopes may still run."""


class AnnotationScopeLimitError(AnnotationRetryLimitError):
    """The current chunk has exhausted its immutable per-scope allowance."""


class AnnotationCostLimitError(AnnotationAttemptLimitError):
    """The next call would violate the frozen cost/quota boundary."""


class AnnotationCostTelemetryError(AnnotationAttemptLimitError):
    """A successful proxy response omitted required cost/quota telemetry."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnnotationAttemptLimitError(
            f"invalid {name}: expected integer >= {minimum}, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class AnnotationRateCard:
    """Frozen per-token prices plus a conservative characters-to-tokens bound.

    ``tokens_per_char_upper`` must over-estimate the real tokenizer.  The
    pipeline's own chunker uses a 4-chars-per-token *estimate* (0.25 tok/char),
    which is optimistic on the mathematical and unicode-heavy text in this
    battery; the bound therefore uses a deliberately larger value so that the
    worst case is an upper bound rather than a central guess.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    tokens_per_char_upper: float

    #: Per-character token bound for non-ASCII text.  BPE tokenizers spend up
    #: to ~3 tokens on a rare symbol (observed live 2026-08-11: two calls at
    #: $0.047493 exceeded the flat-0.4 "bound" of $0.047118 on symbol-bearing
    #: chunks — the flat rate under-estimates ◆◇/√-dense input).
    NONASCII_TOKENS_PER_CHAR = 3.0

    def estimate_input_tokens_upper(self, prompt: str) -> int:
        """Char-class-aware upper bound on the prompt's token count."""
        non_ascii = sum(1 for c in prompt if ord(c) > 127)
        ascii_chars = len(prompt) - non_ascii
        return math.ceil(ascii_chars * self.tokens_per_char_upper
                         + non_ascii * self.NONASCII_TOKENS_PER_CHAR)

    def worst_case_cost_usd(self, prompt_chars: int, max_output_tokens: int) -> float:
        """Static maximum charge for an all-ASCII prompt of ``prompt_chars``.

        The per-call check (:meth:`call_cost_upper_usd`) re-prices the ACTUAL
        prompt with the char-class-aware estimate, so unicode-dense prompts are
        bounded per call rather than assumed away here.
        """
        input_tokens = math.ceil(prompt_chars * self.tokens_per_char_upper)
        return (
            input_tokens * self.input_usd_per_mtok / 1_000_000.0
            + int(max_output_tokens) * self.output_usd_per_mtok / 1_000_000.0
        )

    def call_cost_upper_usd(self, prompt: str, max_output_tokens: int) -> float:
        """Upper bound on one actual call: char-class-aware input + full output."""
        return (
            self.estimate_input_tokens_upper(prompt)
            * self.input_usd_per_mtok / 1_000_000.0
            + int(max_output_tokens) * self.output_usd_per_mtok / 1_000_000.0
        )


def _require_rate_card(doc: dict) -> AnnotationRateCard:
    card = doc.get("rate_card")
    if not isinstance(card, dict) or set(card) != {
        "input_usd_per_mtok", "output_usd_per_mtok", "tokens_per_char_upper"
    }:
        raise AnnotationAttemptLimitError(
            "rate_card must declare exactly input_usd_per_mtok, "
            "output_usd_per_mtok and tokens_per_char_upper"
        )
    values = {}
    for name, value in card.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value <= 0):
            raise AnnotationAttemptLimitError(f"invalid rate_card {name}")
        values[name] = float(value)
    return AnnotationRateCard(**values)


@dataclass(frozen=True)
class AnnotationAttemptPolicy:
    manifest_file_sha256: str
    model: str
    annotation_window_tokens: int
    planned_initial_requests: int
    global_retry_capacity: int
    max_total_attempts: int
    max_attempts_per_scope: int
    approved_spend_ceiling_usd: float
    max_cost_per_attempt_usd: float
    remaining_quota_floor_usd: float
    source_paths: tuple[str, ...]
    bound_paths: tuple[str, ...]
    rate_card: AnnotationRateCard
    max_output_tokens: int
    max_prompt_chars: int
    prior_committed_spend_usd: float
    coverage_rule_version: str

    @property
    def worst_case_attempt_cost_usd(self) -> float:
        """Upper bound on any single call this policy can authorise."""
        return self.rate_card.worst_case_cost_usd(
            self.max_prompt_chars, self.max_output_tokens
        )


class AnnotationAttemptGuard:
    """Reserve attempts durably before calls and enforce immutable ceilings."""

    def __init__(self, policy: AnnotationAttemptPolicy, journal_path: Path):
        self.policy = policy
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        expected_file_sha256: str,
        journal_path: Path,
        *,
        root: Path,
        expected_model: str,
        expected_annotation_window_tokens: int,
        expected_bound_paths: tuple[str, ...] = (),
        expected_max_output_tokens: int,
        expected_max_prompt_chars: int,
        expected_coverage_rule_version: str,
    ) -> "AnnotationAttemptGuard":
        manifest_path = Path(manifest_path)
        root = Path(root).resolve()
        if not _SHA256_RE.fullmatch(expected_file_sha256):
            raise AnnotationAttemptLimitError(
                "annotation guard manifest SHA-256 must be 64 lowercase hex characters"
            )
        if not manifest_path.is_file():
            raise AnnotationAttemptLimitError(
                f"annotation guard manifest missing: {manifest_path}"
            )
        actual_sha = sha256_file(manifest_path)
        if actual_sha != expected_file_sha256:
            raise AnnotationAttemptLimitError(
                "annotation guard manifest SHA-256 mismatch; refusing all API calls"
            )
        try:
            doc = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnotationAttemptLimitError(
                f"annotation guard manifest unreadable: {exc}"
            ) from exc
        if not isinstance(doc, dict):
            raise AnnotationAttemptLimitError("annotation guard manifest must be an object")
        if doc.get("schema") != SCHEMA_VERSION or doc.get("status") != "authorised":
            raise AnnotationAttemptLimitError(
                "annotation guard manifest is not an authorised supported manifest"
            )
        if doc.get("model") != expected_model:
            raise AnnotationAttemptLimitError("annotation guard model mismatch")
        window = _require_int(
            doc.get("annotation_window_tokens"), "annotation_window_tokens", 1
        )
        if window != expected_annotation_window_tokens:
            raise AnnotationAttemptLimitError("annotation guard window mismatch")

        guards = doc.get("guards")
        if not isinstance(guards, dict):
            raise AnnotationAttemptLimitError("annotation guard limits are missing")
        planned = _require_int(
            guards.get("planned_initial_requests"), "planned_initial_requests", 1
        )
        retry_capacity = _require_int(
            guards.get("global_retry_capacity"), "global_retry_capacity", 0
        )
        total = _require_int(guards.get("max_total_attempts"), "max_total_attempts", 1)
        per_scope = _require_int(
            guards.get("max_attempts_per_scope"), "max_attempts_per_scope", 1
        )
        if per_scope > 3:
            raise AnnotationAttemptLimitError(
                "max_attempts_per_scope exceeds the hard project limit of 3"
            )
        if total != planned + retry_capacity:
            raise AnnotationAttemptLimitError(
                "max_total_attempts must equal planned_initial_requests + "
                "global_retry_capacity"
            )
        if total > planned * per_scope:
            raise AnnotationAttemptLimitError(
                "global attempt ceiling exceeds the per-scope ceiling product"
            )
        spend = guards.get("approved_spend_ceiling_usd")
        if (isinstance(spend, bool) or not isinstance(spend, (int, float))
                or not math.isfinite(float(spend)) or spend <= 0):
            raise AnnotationAttemptLimitError("invalid approved_spend_ceiling_usd")
        spend = float(spend)
        if spend > MAX_PREAPPROVED_SPEND_USD:
            raise AnnotationAttemptLimitError(
                f"spend ceiling exceeds pre-approved ${MAX_PREAPPROVED_SPEND_USD:.2f}"
            )
        max_call_cost = guards.get("max_cost_per_attempt_usd")
        if (isinstance(max_call_cost, bool)
                or not isinstance(max_call_cost, (int, float))
                or not math.isfinite(float(max_call_cost))
                or max_call_cost <= 0 or max_call_cost > spend):
            raise AnnotationAttemptLimitError("invalid max_cost_per_attempt_usd")
        quota_floor = guards.get("remaining_quota_floor_usd")
        if (isinstance(quota_floor, bool)
                or not isinstance(quota_floor, (int, float))
                or not math.isfinite(float(quota_floor))
                or quota_floor < 0):
            raise AnnotationAttemptLimitError("invalid remaining_quota_floor_usd")

        # ── provable per-call bound (2026-08-11 correction) ──────────────────
        # The operational output allowance and the largest prompt the plan can
        # build are BOTH declared here, so the theoretical maximum charge is a
        # property of the manifest rather than of an empirical echo estimate.
        rate_card = _require_rate_card(doc)
        max_output_tokens = _require_int(
            guards.get("max_output_tokens"), "max_output_tokens", 1
        )
        max_prompt_chars = _require_int(
            guards.get("max_prompt_chars"), "max_prompt_chars", 1
        )
        if max_output_tokens != expected_max_output_tokens:
            raise AnnotationAttemptLimitError(
                "annotation guard max_output_tokens does not match the "
                "executing output allowance"
            )
        if max_prompt_chars != expected_max_prompt_chars:
            raise AnnotationAttemptLimitError(
                "annotation guard max_prompt_chars does not match the "
                "executing chunk plan"
            )
        if float(max_call_cost) > MAX_COST_PER_ATTEMPT_HARD_USD:
            raise AnnotationAttemptLimitError(
                "max_cost_per_attempt_usd exceeds the hard project limit of "
                f"${MAX_COST_PER_ATTEMPT_HARD_USD:.2f}"
            )
        worst_case = rate_card.worst_case_cost_usd(max_prompt_chars, max_output_tokens)
        if worst_case > float(max_call_cost):
            raise AnnotationAttemptLimitError(
                "theoretical worst-case per-call cost "
                f"${worst_case:.6f} exceeds the authorised per-call maximum "
                f"${float(max_call_cost):.6f}; refusing all API calls"
            )

        prior_spend = guards.get("prior_committed_spend_usd", 0.0)
        if (isinstance(prior_spend, bool)
                or not isinstance(prior_spend, (int, float))
                or not math.isfinite(float(prior_spend)) or prior_spend < 0):
            raise AnnotationAttemptLimitError("invalid prior_committed_spend_usd")
        if float(prior_spend) > spend:
            raise AnnotationAttemptLimitError(
                "prior committed spend already exceeds the approved ceiling"
            )
        coverage_rule = doc.get("coverage_rule_version")
        if coverage_rule != expected_coverage_rule_version:
            raise AnnotationAttemptLimitError(
                "annotation guard coverage_rule_version does not match the "
                "executing coverage validator"
            )

        def validate_artifacts(field: str) -> set[str]:
            items = doc.get(field)
            if not isinstance(items, list) or not items:
                raise AnnotationAttemptLimitError(f"{field} must be a non-empty list")
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                    raise AnnotationAttemptLimitError(
                        f"each {field} item must contain exactly path and sha256"
                    )
                rel = item["path"]
                expected = item["sha256"]
                if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
                    raise AnnotationAttemptLimitError(
                        f"{field} path must be repository-relative"
                    )
                if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
                    raise AnnotationAttemptLimitError(f"invalid {field} SHA-256")
                candidate = (root / rel).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise AnnotationAttemptLimitError(
                        f"{field} path escapes repository root"
                    ) from exc
                if rel in seen:
                    raise AnnotationAttemptLimitError(f"duplicate {field} path")
                seen.add(rel)
                if not candidate.is_file() or sha256_file(candidate) != expected:
                    raise AnnotationAttemptLimitError(
                        f"{field} missing or hash-mismatched: {rel}"
                    )
            return seen

        seen_paths = validate_artifacts("source_artifacts")
        bound_paths = validate_artifacts("bound_artifacts")
        if tuple(sorted(bound_paths)) != tuple(sorted(expected_bound_paths)):
            raise AnnotationAttemptLimitError(
                "bound_artifacts set does not match the executing protocol/code set"
            )

        policy = AnnotationAttemptPolicy(
            manifest_file_sha256=actual_sha,
            model=expected_model,
            annotation_window_tokens=window,
            planned_initial_requests=planned,
            global_retry_capacity=retry_capacity,
            max_total_attempts=total,
            max_attempts_per_scope=per_scope,
            approved_spend_ceiling_usd=spend,
            max_cost_per_attempt_usd=float(max_call_cost),
            remaining_quota_floor_usd=float(quota_floor),
            source_paths=tuple(sorted(seen_paths)),
            bound_paths=tuple(sorted(bound_paths)),
            rate_card=rate_card,
            max_output_tokens=max_output_tokens,
            max_prompt_chars=max_prompt_chars,
            prior_committed_spend_usd=float(prior_spend),
            coverage_rule_version=str(coverage_rule),
        )
        return cls(policy, journal_path)

    def assert_attempt_cost_bound(self, prompt: str, max_output_tokens: int) -> None:
        """Refuse, before any network I/O, a call that could exceed its ceiling.

        This is the fail-closed complement to :meth:`record_response`, which can
        only observe a breach after it has already been billed.  Both the actual
        prompt length and the actual output allowance handed to the transport
        are checked against the frozen rate card, so a caller cannot widen the
        bound by passing a larger allowance than the manifest authorised.
        """
        allowance = _require_int(max_output_tokens, "max_output_tokens", 1)
        if allowance > self.policy.max_output_tokens:
            raise AnnotationCostLimitError(
                f"requested output allowance {allowance} exceeds the authorised "
                f"{self.policy.max_output_tokens}; no call made"
            )
        prompt_chars = len(prompt)
        if prompt_chars > self.policy.max_prompt_chars:
            raise AnnotationCostLimitError(
                f"prompt of {prompt_chars} chars exceeds the authorised "
                f"{self.policy.max_prompt_chars}; no call made"
            )
        worst_case = self.policy.rate_card.call_cost_upper_usd(prompt, allowance)
        if worst_case > self.policy.max_cost_per_attempt_usd:
            raise AnnotationCostLimitError(
                f"theoretical maximum charge ${worst_case:.6f} exceeds the "
                f"authorised per-call maximum "
                f"${self.policy.max_cost_per_attempt_usd:.6f}; no call made"
            )

    def assert_planned_initial_requests(self, observed: int) -> None:
        observed = _require_int(observed, "observed planned initial requests", 1)
        if observed != self.policy.planned_initial_requests:
            raise AnnotationAttemptLimitError(
                "annotation request-plan mismatch: "
                f"manifest={self.policy.planned_initial_requests}, observed={observed}"
            )

    def _replay_locked(self, handle) -> dict:
        handle.seek(0)
        state = {
            "total": 0,
            "initial": 0,
            "retries": 0,
            "per_scope": {},
            "reservations": {},
            "outcomes": {},
            "reported_cost_usd": 0.0,
            "last_remaining_quota_usd": None,
            "response_telemetry_missing": False,
        }
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AnnotationAttemptLimitError(
                    f"corrupt annotation attempt journal at line {line_number}"
                ) from exc
            if (
                not isinstance(event, dict)
                or event.get("schema") != JOURNAL_SCHEMA_VERSION
                or event.get("manifest_file_sha256")
                != self.policy.manifest_file_sha256
                or event.get("event") not in {"reservation", "outcome"}
            ):
                raise AnnotationAttemptLimitError(
                    f"invalid or foreign annotation attempt journal event at line {line_number}"
                )

            if event["event"] == "reservation":
                if (
                    not _SHA256_RE.fullmatch(str(event.get("scope_sha256", "")))
                    or not isinstance(event.get("ordinal"), int)
                    or not isinstance(event.get("scope_ordinal"), int)
                    or event.get("attempt_kind") not in {"initial", "retry"}
                    or not isinstance(event.get("reservation_id"), str)
                    or not event["reservation_id"]
                    or event["reservation_id"] in state["reservations"]
                ):
                    raise AnnotationAttemptLimitError(
                        f"invalid reservation event at journal line {line_number}"
                    )
                scope = event["scope_sha256"]
                state["total"] += 1
                per_scope = state["per_scope"]
                per_scope[scope] = per_scope.get(scope, 0) + 1
                expected_kind = "initial" if per_scope[scope] == 1 else "retry"
                state["initial"] += int(expected_kind == "initial")
                state["retries"] += int(expected_kind == "retry")
                if (event["ordinal"] != state["total"]
                        or event["scope_ordinal"] != per_scope[scope]
                        or event["attempt_kind"] != expected_kind):
                    raise AnnotationAttemptLimitError(
                        f"non-contiguous reservation at journal line {line_number}"
                    )
                state["reservations"][event["reservation_id"]] = event
                continue

            reservation_id = event.get("reservation_id")
            if (
                reservation_id not in state["reservations"]
                or reservation_id in state["outcomes"]
                or event.get("outcome") not in {"response", "transport_failure"}
            ):
                raise AnnotationAttemptLimitError(
                    f"invalid outcome event at journal line {line_number}"
                )
            cost = event.get("reported_cost_usd")
            quota = event.get("remaining_quota_usd")
            if cost is not None and (
                isinstance(cost, bool) or not isinstance(cost, (int, float))
                or not math.isfinite(float(cost)) or cost < 0
            ):
                raise AnnotationAttemptLimitError(
                    f"invalid reported cost at journal line {line_number}"
                )
            if quota is not None and (
                isinstance(quota, bool) or not isinstance(quota, (int, float))
                or not math.isfinite(float(quota)) or quota < 0
            ):
                raise AnnotationAttemptLimitError(
                    f"invalid remaining quota at journal line {line_number}"
                )
            if event["outcome"] == "response" and (cost is None or quota is None):
                state["response_telemetry_missing"] = True
            if cost is not None:
                state["reported_cost_usd"] += float(cost)
            if quota is not None:
                state["last_remaining_quota_usd"] = float(quota)
            state["outcomes"][reservation_id] = event

        unresolved_reservations = set(state["reservations"]) - set(state["outcomes"])
        # An interrupted/failed call with no outcome is charged at the approved
        # per-call maximum. This is deliberately conservative and never refunds.
        # Spend already committed under a SUPERSEDED manifest (e.g. the halted
        # 2026-08-11 run) is carried forward here, so replacing the manifest —
        # which necessarily starts a fresh journal — can never reset the
        # cumulative ceiling back to zero.
        state["committed_cost_usd"] = (
            self.policy.prior_committed_spend_usd
            + state["reported_cost_usd"]
            + len(unresolved_reservations) * self.policy.max_cost_per_attempt_usd
            + sum(
                self.policy.max_cost_per_attempt_usd
                for event in state["outcomes"].values()
                if event.get("reported_cost_usd") is None
            )
        )
        return state

    def reserve(self, scope_sha256: str) -> dict:
        """Persist one attempt reservation and return its sanitized event."""
        if not _SHA256_RE.fullmatch(scope_sha256):
            raise AnnotationAttemptLimitError("attempt scope must be a SHA-256 digest")
        with self.journal_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state = self._replay_locked(handle)
            total = state["total"]
            initial = state["initial"]
            retries = state["retries"]
            scope_count = state["per_scope"].get(scope_sha256, 0)
            attempt_kind = "initial" if scope_count == 0 else "retry"
            if state["response_telemetry_missing"]:
                raise AnnotationCostTelemetryError(
                    "successful proxy response omitted cost/quota telemetry; no call made"
                )
            last_quota = state["last_remaining_quota_usd"]
            if (last_quota is not None
                    and last_quota < self.policy.remaining_quota_floor_usd):
                raise AnnotationCostLimitError(
                    "remaining proxy quota is below the frozen floor; no call made"
                )
            if (state["committed_cost_usd"] + self.policy.max_cost_per_attempt_usd
                    > self.policy.approved_spend_ceiling_usd):
                raise AnnotationCostLimitError(
                    "next annotation call could exceed the spend ceiling; no call made"
                )
            if scope_count >= self.policy.max_attempts_per_scope:
                raise AnnotationScopeLimitError(
                    "per-chunk annotation API-attempt ceiling reached; no call made"
                )
            if attempt_kind == "initial" and initial >= self.policy.planned_initial_requests:
                raise AnnotationAttemptLimitError(
                    "planned initial annotation API-attempt ceiling reached; no call made"
                )
            if attempt_kind == "retry" and retries >= self.policy.global_retry_capacity:
                raise AnnotationRetryLimitError(
                    "global annotation retry ceiling reached; no call made"
                )
            if total >= self.policy.max_total_attempts:
                raise AnnotationAttemptLimitError(
                    "global annotation API-attempt ceiling reached; no call made"
                )
            event = {
                "schema": JOURNAL_SCHEMA_VERSION,
                "event": "reservation",
                "reservation_id": uuid.uuid4().hex,
                "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "manifest_file_sha256": self.policy.manifest_file_sha256,
                "scope_sha256": scope_sha256,
                "ordinal": total + 1,
                "scope_ordinal": scope_count + 1,
                "attempt_kind": attempt_kind,
            }
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return event

    def _record_outcome(
        self,
        reservation_id: str,
        *,
        outcome: str,
        reported_cost_usd: float | None,
        remaining_quota_usd: float | None,
        error_class: str | None = None,
    ) -> None:
        with self.journal_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state = self._replay_locked(handle)
            if reservation_id not in state["reservations"]:
                raise AnnotationAttemptLimitError("unknown attempt reservation outcome")
            if reservation_id in state["outcomes"]:
                raise AnnotationAttemptLimitError("duplicate attempt outcome")
            event = {
                "schema": JOURNAL_SCHEMA_VERSION,
                "event": "outcome",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_file_sha256": self.policy.manifest_file_sha256,
                "reservation_id": reservation_id,
                "outcome": outcome,
                "reported_cost_usd": reported_cost_usd,
                "remaining_quota_usd": remaining_quota_usd,
                "error_class": error_class,
            }
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_response(
        self,
        reservation_id: str,
        reported_cost_usd: object,
        remaining_quota_usd: object,
    ) -> None:
        def number_or_none(value: object) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            return number if math.isfinite(number) and number >= 0 else None

        cost = number_or_none(reported_cost_usd)
        quota = number_or_none(remaining_quota_usd)
        self._record_outcome(
            reservation_id,
            outcome="response",
            reported_cost_usd=cost,
            remaining_quota_usd=quota,
        )
        if cost is None or quota is None:
            raise AnnotationCostTelemetryError(
                "successful proxy response omitted numeric cost/quota telemetry"
            )
        if cost > self.policy.max_cost_per_attempt_usd:
            raise AnnotationCostLimitError(
                "reported call cost exceeded the approved per-call maximum"
            )

    def record_transport_failure(self, reservation_id: str, error_class: str) -> None:
        self._record_outcome(
            reservation_id,
            outcome="transport_failure",
            reported_cost_usd=None,
            remaining_quota_usd=None,
            error_class=error_class,
        )

    def summary(self) -> dict:
        with self.journal_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            state = self._replay_locked(handle)
        return {
            "manifest_file_sha256": self.policy.manifest_file_sha256,
            "journal_path": str(self.journal_path),
            "attempts_reserved": state["total"],
            "initial_attempts_reserved": state["initial"],
            "retry_attempts_reserved": state["retries"],
            "scopes_attempted": len(state["per_scope"]),
            "max_scope_attempts_observed": max(state["per_scope"].values(), default=0),
            "max_total_attempts": self.policy.max_total_attempts,
            "max_attempts_per_scope": self.policy.max_attempts_per_scope,
            "planned_initial_requests": self.policy.planned_initial_requests,
            "global_retry_capacity": self.policy.global_retry_capacity,
            "approved_spend_ceiling_usd": self.policy.approved_spend_ceiling_usd,
            "max_cost_per_attempt_usd": self.policy.max_cost_per_attempt_usd,
            "remaining_quota_floor_usd": self.policy.remaining_quota_floor_usd,
            "max_output_tokens": self.policy.max_output_tokens,
            "max_prompt_chars": self.policy.max_prompt_chars,
            "worst_case_attempt_cost_usd": self.policy.worst_case_attempt_cost_usd,
            "coverage_rule_version": self.policy.coverage_rule_version,
            "prior_committed_spend_usd": self.policy.prior_committed_spend_usd,
            "reported_cost_usd": state["reported_cost_usd"],
            "committed_cost_usd": state["committed_cost_usd"],
            "last_remaining_quota_usd": state["last_remaining_quota_usd"],
            "response_telemetry_missing": state["response_telemetry_missing"],
        }

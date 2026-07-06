"""LocalEnvironment subclass that injects case fields into agent Jinja templates."""
from __future__ import annotations

from typing import Any

from minisweagent.environments.local import LocalEnvironment
from minisweagent.utils.serialize import recursive_merge

from retrieval.case_api_client import CaseApiClient


class AlqacEnv(LocalEnvironment):
    """Wraps LocalEnvironment with ALQAC case context + per-case SQLite reset."""

    def __init__(
        self,
        *,
        case: dict,
        runs_dir: str = "runs",
        reset_case_log: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.case = case
        if reset_case_log:
            try:
                CaseApiClient(runs_dir=runs_dir).reset_case(case["case_id"])
            except RuntimeError:
                pass  # ALQAC_API_KEY unset: skip reset (smoke mode)

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        case = self.case
        return recursive_merge(
            super().get_template_vars(),
            {
                "case_id": case.get("case_id", ""),
                "court": case.get("court", ""),
                "case_type": case.get("case_type", ""),
                "A_role": case.get("A_role", ""),
                "B_role": case.get("B_role", ""),
                "case_fact": case.get("case_fact", ""),
                "case_query": case.get("case_query", ""),
            },
            kwargs,
        )

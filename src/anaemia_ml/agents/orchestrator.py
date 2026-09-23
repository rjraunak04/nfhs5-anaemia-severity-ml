"""Evidence-grounded orchestration for the public research copilot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from anaemia_ml.agents.planner import HybridPlanner, Planner
from anaemia_ml.agents.policies import (
    enforce_disclosure_boundary,
    enforce_public_intent,
)
from anaemia_ml.agents.schemas import (
    AgentIntent,
    AgentRequest,
    AgentResponse,
    EvidenceItem,
    PlanDecision,
    TraceStep,
)
from anaemia_ml.agents.tools import (
    calibration_evidence,
    explainability_evidence,
    final_test_gate_names,
    load_public_evidence,
    model_comparison,
    next_experiment_plan,
    release_readiness,
    selected_model,
)


class IntentRoutingError(ValueError):
    """Raised when a user request cannot be mapped to a safe supported intent."""


_INTENT_PATTERNS: tuple[tuple[AgentIntent, tuple[str, ...]], ...] = (
    (
        AgentIntent.RELEASE_READINESS,
        (
            "portfolio release",
            "release readiness",
            "software release",
            "deploy ready",
            "deployment ready",
            "portfolio ready",
        ),
    ),
    (
        AgentIntent.NEXT_EXPERIMENT,
        (
            "next experiment",
            "what should i run next",
            "next research step",
            "what to do next",
            "next validation",
        ),
    ),
    (
        AgentIntent.CHECK_FINAL_TEST_READINESS,
        ("final test", "locked test", "final evaluation", "confirmatory ready"),
    ),
    (
        AgentIntent.CALIBRATION_STATUS,
        ("calibration", "brier", "log loss", "ece", "temperature"),
    ),
    (
        AgentIntent.EXPLAIN_FEATURES,
        ("feature", "shap", "important predictor", "explainability"),
    ),
    (
        AgentIntent.COMPARE_MODELS,
        ("compare model", "model comparison", "which models", "macro f1"),
    ),
    (
        AgentIntent.EXPLAIN_SELECTION,
        ("selected model", "why random forest", "why selected", "best model"),
    ),
    (
        AgentIntent.PROJECT_STATUS,
        ("status", "project", "pipeline", "what is complete", "summary"),
    ),
)


def route_intent(query: str) -> AgentIntent:
    """Map plain-language requests to a small auditable intent set."""
    normalized = re.sub(r"\s+", " ", query.casefold()).strip()
    for intent, phrases in _INTENT_PATTERNS:
        if any(phrase in normalized for phrase in phrases):
            return intent
    raise IntentRoutingError(
        "I can explain project status, compare models, explain model selection, summarize "
        "calibration/SHAP, audit portfolio release readiness, plan the next experiment, "
        "or check locked-test readiness."
    )


_TOOL_NAMES: dict[AgentIntent, str] = {
    AgentIntent.PROJECT_STATUS: "read_project_status",
    AgentIntent.COMPARE_MODELS: "compare_development_models",
    AgentIntent.EXPLAIN_SELECTION: "explain_model_selection",
    AgentIntent.CALIBRATION_STATUS: "read_calibration_evidence",
    AgentIntent.EXPLAIN_FEATURES: "read_aggregate_shap",
    AgentIntent.CHECK_FINAL_TEST_READINESS: "read_final_test_gates",
    AgentIntent.RELEASE_READINESS: "audit_release_readiness",
    AgentIntent.NEXT_EXPERIMENT: "build_next_experiment_plan",
}


class ResearchCopilot:
    """Policy-gated agent that reasons only over approved aggregate evidence."""

    def __init__(
        self,
        *,
        summary_path: str | Path,
        validation_path: str | Path,
        planner: Planner | None = None,
    ) -> None:
        self.summary_path = Path(summary_path)
        self.validation_path = Path(validation_path)
        self.planner = planner or HybridPlanner()

    def run(self, request: AgentRequest) -> AgentResponse:
        """Plan, enforce policy, execute tools, and return a traced response."""
        if request.intent is None:
            decision = self.planner.plan(request.query)
        else:
            decision = PlanDecision(
                intent=request.intent,
                planner="explicit_intent",
                confidence=1.0,
                reason="Caller supplied an explicit approved intent.",
            )

        intent = decision.intent
        trace = [
            TraceStep(
                stage="plan",
                name=decision.planner,
                status="ok",
                detail=(
                    f"Selected {intent.value} with confidence "
                    f"{decision.confidence:.2f}. {decision.reason}"
                ),
            )
        ]

        enforce_public_intent(intent)
        trace.append(
            TraceStep(
                stage="policy",
                name="public_intent_allowlist",
                status="ok",
                detail="Requested action is in the closed public intent set.",
            )
        )

        summary = load_public_evidence(self.summary_path)
        enforce_disclosure_boundary(summary)
        trace.append(
            TraceStep(
                stage="policy",
                name="aggregate_disclosure_boundary",
                status="ok",
                detail="Evidence is aggregate-only and the locked final test remains untouched.",
            )
        )

        handlers = {
            AgentIntent.PROJECT_STATUS: self._project_status,
            AgentIntent.COMPARE_MODELS: self._compare_models,
            AgentIntent.EXPLAIN_SELECTION: self._explain_selection,
            AgentIntent.CALIBRATION_STATUS: self._calibration_status,
            AgentIntent.EXPLAIN_FEATURES: self._explain_features,
            AgentIntent.CHECK_FINAL_TEST_READINESS: self._final_test_readiness,
            AgentIntent.RELEASE_READINESS: self._release_readiness,
            AgentIntent.NEXT_EXPERIMENT: self._next_experiment,
        }

        response = handlers[intent](summary)
        trace.append(
            TraceStep(
                stage="tool",
                name=_TOOL_NAMES[intent],
                status="ok",
                detail="Deterministic project evidence/tooling produced the response inputs.",
            )
        )
        trace.append(
            TraceStep(
                stage="response",
                name="grounded_response",
                status=response.status,
                detail="Response contains explicit evidence sources and preserved research boundaries.",
            )
        )
        response.trace = trace
        response.metadata.update(
            {
                "planner": decision.planner,
                "planner_confidence": decision.confidence,
                "public_mode": True,
                "evidence_only": True,
                "tool": _TOOL_NAMES[intent],
            }
        )
        return response

    def _project_status(self, summary: dict[str, Any]) -> AgentResponse:
        workflow = summary["workflow"]
        selected = selected_model(summary)
        rows = workflow["partition_rows"]
        return AgentResponse(
            intent=AgentIntent.PROJECT_STATUS,
            status="ok",
            title="Project status",
            summary=(
                "The public development pipeline is complete through grouped model selection, "
                "calibration, aggregate explainability, testing and deployment. The protocol-defined "
                "final test remains intentionally locked."
            ),
            evidence=[
                EvidenceItem(
                    label="Selected development model",
                    value=str(selected["model_name"]),
                    source="aggregate portfolio summary",
                ),
                EvidenceItem(
                    label="Development / calibration / locked-test rows",
                    value=(
                        f'{rows["development"]:,} / {rows["calibration"]:,} / '
                        f'{rows["locked_test"]:,}'
                    ),
                    source="aggregate portfolio summary",
                ),
            ],
            warnings=["Development evidence is not final-test or clinical performance."],
        )

    def _compare_models(self, summary: dict[str, Any]) -> AgentResponse:
        rows = model_comparison(summary)
        evidence = [
            EvidenceItem(
                label=str(row["display_name"]),
                value=(
                    f'Macro-F1 {float(row["macro_f1_mean"]):.3f}; '
                    f'balanced accuracy {float(row["balanced_accuracy_mean"]):.3f}; '
                    f'severe recall {float(row["severe_recall_mean"]):.3f}'
                ),
                source="PSU-grouped development nested CV",
            )
            for row in rows
        ]
        leader = rows[0]
        return AgentResponse(
            intent=AgentIntent.COMPARE_MODELS,
            status="ok",
            title="Development model comparison",
            summary=(
                f'{leader["display_name"]} has the highest development macro-F1 among the '
                "reported candidates. This ranking applies only to grouped development estimates."
            ),
            evidence=evidence,
            warnings=["Do not interpret this as locked-test or external-validation performance."],
        )

    def _explain_selection(self, summary: dict[str, Any]) -> AgentResponse:
        selection = selected_model(summary)
        rows = {row["model_name"]: row for row in model_comparison(summary)}
        chosen = rows[str(selection["model_name"])]
        severe_recall = float(chosen["severe_recall_mean"])
        return AgentResponse(
            intent=AgentIntent.EXPLAIN_SELECTION,
            status="needs_review" if severe_recall < 0.20 else "ok",
            title="Why this model was selected",
            summary=(
                f'{chosen["display_name"]} was selected because macro-F1 is the prespecified '
                "primary development metric. Selection does not imply that every clinically "
                "important class performs well."
            ),
            evidence=[
                EvidenceItem(
                    label="Primary metric",
                    value=f'Macro-F1 {float(chosen["macro_f1_mean"]):.3f}',
                    source="PSU-grouped development nested CV",
                ),
                EvidenceItem(
                    label="Severe-class recall",
                    value=f"{severe_recall:.3f}",
                    source="PSU-grouped development nested CV",
                ),
            ],
            warnings=[
                "Severe-class recall is weak; the selected model should not be presented "
                "as a screening tool."
            ],
        )

    def _calibration_status(self, summary: dict[str, Any]) -> AgentResponse:
        calibration = calibration_evidence(summary)
        before = calibration["metrics_before"]
        after = calibration["deployed_metrics_estimate"]
        return AgentResponse(
            intent=AgentIntent.CALIBRATION_STATUS,
            status="ok",
            title="Calibration status",
            summary=(
                "Temperature scaling was selected on the separate calibration partition using "
                "PSU-disjoint cross-fitting."
            ),
            evidence=[
                EvidenceItem(
                    label="Temperature",
                    value=f'{float(calibration["temperature"]):.3f}',
                    source="calibration partition",
                ),
                EvidenceItem(
                    label="Weighted log loss",
                    value=f'{float(before["log_loss"]):.3f} → {float(after["log_loss"]):.3f}',
                    source="cross-fitted calibration estimate",
                ),
                EvidenceItem(
                    label="ECE",
                    value=(
                        f'{float(before["expected_calibration_error"]):.3f} → '
                        f'{float(after["expected_calibration_error"]):.3f}'
                    ),
                    source="cross-fitted calibration estimate",
                ),
            ],
            warnings=["Calibration evidence is separate from final-test generalization."],
        )

    def _explain_features(self, summary: dict[str, Any]) -> AgentResponse:
        explanation = explainability_evidence(summary)
        features = explanation["global_features"][:5]
        evidence = [
            EvidenceItem(
                label=str(row["feature"]),
                value=f'Mean |SHAP| {float(row["mean_abs_shap"]):.4f}',
                source="aggregate calibration SHAP",
            )
            for row in features
        ]
        return AgentResponse(
            intent=AgentIntent.EXPLAIN_FEATURES,
            status="ok",
            title="Aggregate feature associations",
            summary=(
                "The leading features below are aggregate model-attribution signals on calibration "
                "observations. They describe model behavior, not causal or clinical importance."
            ),
            evidence=evidence,
            warnings=["SHAP associations must not be interpreted causally."],
        )

    def _final_test_readiness(self, summary: dict[str, Any]) -> AgentResponse:
        gates = final_test_gate_names(self.validation_path)
        return AgentResponse(
            intent=AgentIntent.CHECK_FINAL_TEST_READINESS,
            status="blocked",
            title="Locked final-test readiness",
            summary=(
                "The public copilot cannot unlock or evaluate the final test. The confirmatory "
                "release remains blocked until every protocol gate is independently completed and "
                "the single-use evaluation is explicitly authorized."
            ),
            evidence=[
                EvidenceItem(
                    label="Locked test evaluated",
                    value="No",
                    source="public disclosure metadata",
                ),
                EvidenceItem(
                    label="Required protocol gates",
                    value=", ".join(gates),
                    source=str(self.validation_path),
                ),
            ],
            warnings=[
                "The agent is intentionally incapable of bypassing the final-test unlock policy."
            ],
            metadata={"required_gates": gates},
        )

    def _release_readiness(self, summary: dict[str, Any]) -> AgentResponse:
        audit = release_readiness(summary)
        checks = audit["checks"]
        passed = sum(bool(check["passed"]) for check in checks)
        total = len(checks)
        return AgentResponse(
            intent=AgentIntent.RELEASE_READINESS,
            status="ok" if audit["portfolio_release_ready"] else "needs_review",
            title="Portfolio release readiness",
            summary=(
                f"{passed}/{total} public engineering gates pass. "
                "The portfolio/software release is ready, while confirmatory scientific "
                "release remains a separate protocol-gated milestone."
                if audit["portfolio_release_ready"]
                else f"{passed}/{total} public engineering gates pass; failed checks need review."
            ),
            evidence=[
                EvidenceItem(
                    label=str(check["name"]),
                    value="PASS" if check["passed"] else "FAIL",
                    source=str(check["detail"]),
                )
                for check in checks
            ],
            warnings=[
                "Portfolio release readiness does not imply final-test, clinical, or external "
                "validation readiness."
            ],
            metadata={
                "portfolio_release_ready": audit["portfolio_release_ready"],
                "confirmatory_release_ready": audit["confirmatory_release_ready"],
                "next_action": audit["next_action"],
            },
        )

    def _next_experiment(self, summary: dict[str, Any]) -> AgentResponse:
        plan = next_experiment_plan(summary, self.validation_path)
        return AgentResponse(
            intent=AgentIntent.NEXT_EXPERIMENT,
            status="needs_review",
            title="Next research milestones",
            summary=(
                "The next work should progress from geographic validation to frozen confirmatory "
                "evaluation. The plan below is ordered to avoid leakage and post-hoc tuning."
            ),
            evidence=[
                EvidenceItem(
                    label=f'Step {item["step"]}: {item["action"]}',
                    value=item["why"],
                    source="protocol-safe experiment planner",
                )
                for item in plan
            ],
            warnings=[
                "This is an execution plan, not authorization to open the locked final test."
            ],
            metadata={"planned_steps": len(plan)},
        )

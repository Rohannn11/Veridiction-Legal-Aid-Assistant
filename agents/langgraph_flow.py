"""Step 3+: Veridiction LangGraph flow (Retriever -> Structured Advisor -> Safety).

Implements:
- Claim classification + retrieval
- Structured legal response sections
- Grok-backed advisor generation with deterministic fallback
- Safety flags and mandatory disclaimer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legal.knowledge_base import LegalKnowledgeBase
from nlp.classifier import ClaimClassifier
from rag.retriever import LegalRetriever


LOGGER = logging.getLogger(__name__)

MANDATORY_DISCLAIMER = (
    "⚠️ This is NOT legal advice. This is an AI research prototype. "
    "Please consult a qualified lawyer immediately."
)


def _read_env_value(key: str) -> str | None:
    value = os.getenv(key)
    if value and value.strip():
        return value.strip()

    env_path = Path(".env")
    if not env_path.exists():
        return None

    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


class FlowchartStep(BaseModel):
    step: int
    title: str
    details: str


class CaseScenario(BaseModel):
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    missing_details: list[str] = Field(default_factory=list)


class PossibleSteps(BaseModel):
    immediate_actions: list[str] = Field(default_factory=list)
    legal_actions: list[str] = Field(default_factory=list)
    next_48_hours: list[str] = Field(default_factory=list)


class RequiredDocumentation(BaseModel):
    mandatory: list[str] = Field(default_factory=list)
    supporting: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class CourtsAndProcess(BaseModel):
    state: str
    courts_forum: list[str] = Field(default_factory=list)
    application_process: list[str] = Field(default_factory=list)
    jurisdiction_note: str


class SeverityAssessment(BaseModel):
    level: str
    rationale: str
    time_sensitivity: str


class HelplineInfo(BaseModel):
    name: str
    number: str
    applicability: str
    availability: str


class StructuredLegalOutput(BaseModel):
    case_scenario: CaseScenario
    possible_steps: PossibleSteps
    required_documentation: RequiredDocumentation
    courts_and_filing_process: CourtsAndProcess
    severity_assessment: SeverityAssessment
    helplines_india: list[HelplineInfo] = Field(default_factory=list)
    flowchart: list[FlowchartStep] = Field(default_factory=list)
    tts_summary: str


class AdvisorOutput(BaseModel):
    issue_summary: str
    action_steps: list[str] = Field(default_factory=list)
    legal_basis: list[str] = Field(default_factory=list)
    documents_to_collect: list[str] = Field(default_factory=list)
    escalation_guidance: str


class SafetyOutput(BaseModel):
    risk_flags: list[str] = Field(default_factory=list)
    safe_next_steps: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


class VeridictionState(TypedDict, total=False):
    user_query: str
    claim: dict[str, Any]
    retrieved_passages: list[dict[str, Any]]
    retrieval_route: str
    retrieval_query_variants: list[str]
    advisor_output: dict[str, Any]
    structured_output: dict[str, Any]
    safety_output: dict[str, Any]
    node_latencies_ms: dict[str, float]
    final_response: dict[str, Any]
    error: str


class GrokClient:
    """Minimal Grok chat-completions client with schema-driven JSON response."""

    def __init__(self) -> None:
        self.api_key = _read_env_value("GROK_API_KEY")
        self.base_url = _read_env_value("GROK_BASE_URL") or "https://api.x.ai/v1"
        self.model = _read_env_value("GROK_MODEL") or "grok-2-latest"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def generate_structured(
        self,
        query: str,
        claim: dict[str, Any],
        passages: list[dict[str, Any]],
        mapping: dict[str, Any],
        state_name: str,
        helplines: list[dict[str, Any]],
        max_output_tokens: int | None = None,
        low_context_mode: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Grok client is not enabled")

        top_passages = []
        for item in passages[:4]:
            text = str(item.get("passage", "")).replace("\n", " ")[:600]
            top_passages.append({"score": item.get("score"), "passage": text})

        prompt = {
            "query": query,
            "claim": claim,
            "state": state_name,
            "mapping": mapping,
            "helplines": helplines,
            "retrieved_passages": top_passages,
            "output_requirements": {
                "language": "English",
                "json_only": True,
                "answer_mode": "low_context" if low_context_mode else "grounded",
                "sections": [
                    "case_scenario",
                    "possible_steps",
                    "required_documentation",
                    "courts_and_filing_process",
                    "severity_assessment",
                    "helplines_india",
                    "flowchart",
                    "tts_summary",
                ],
            },
        }

        body = {
            "model": self.model,
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an Indian legal triage assistant for Maharashtra. Return STRICT JSON only. "
                        "Use retrieved evidence and mapping. Keep it practical and safety-aware. "
                        "If evidence is sparse, avoid legal certainty and ask concise follow-up details."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=True),
                },
            ],
        }
        if max_output_tokens is not None:
            body["max_tokens"] = int(max_output_tokens)

        req = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Grok HTTP error: {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Grok connection error: {exc.reason}") from exc

        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Grok returned empty content")

        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise RuntimeError("Grok returned non-JSON content")

        return json.loads(match.group(0))


class StructuredAdvisor:
    """Builds structured legal response via Grok (preferred) or deterministic fallback."""

    def __init__(self, knowledge: LegalKnowledgeBase, provider: str = "auto") -> None:
        self.knowledge = knowledge
        self.provider = provider
        self.grok = GrokClient()

    def generate(self, query: str, claim: dict[str, Any], passages: list[dict[str, Any]]) -> StructuredLegalOutput:
        claim_type = claim.get("claim_type", "other")
        mapping = self.knowledge.claim_mapping(claim_type)
        top_score = max((float(p.get("score", 0.0)) for p in passages), default=0.0)
        low_grounding = not passages or top_score < 0.84

        if self._can_use_grok():
            try:
                raw = self.grok.generate_structured(
                    query=query,
                    claim=claim,
                    passages=passages,
                    mapping=mapping,
                    state_name=self.knowledge.state,
                    helplines=self.knowledge.national_helplines,
                    max_output_tokens=420 if low_grounding else None,
                    low_context_mode=low_grounding,
                )
                return StructuredLegalOutput.model_validate(raw)
            except Exception as exc:
                LOGGER.warning("Grok structured generation failed, using deterministic fallback: %s", exc)

        response = self._deterministic_response(query=query, claim=claim, passages=passages, mapping=mapping)
        if low_grounding:
            response.case_scenario.missing_details.append(
                "Retrieved evidence was limited. Please share more specific facts/documents for higher-confidence guidance."
            )
        return response

    def _can_use_grok(self) -> bool:
        if self.provider == "fallback":
            return False
        if self.provider == "grok":
            return self.grok.enabled
        return self.grok.enabled

    def _deterministic_response(
        self,
        query: str,
        claim: dict[str, Any],
        passages: list[dict[str, Any]],
        mapping: dict[str, Any],
    ) -> StructuredLegalOutput:
        claim_type = claim.get("claim_type", "other")
        urgency = claim.get("urgency", "medium")

        key_facts = [
            f"Detected claim type: {claim_type}",
            f"Classifier urgency: {urgency}",
            f"User statement: {query[:220]}",
        ]
        if passages:
            key_facts.append("Retrieved legal references support this category.")

        missing_details = [
            "Exact incident timeline (date/time/place)",
            "Identity details of involved persons",
            "Whether any prior complaint/FIR/legal notice was filed",
        ]

        process_steps = list(mapping.get("application_process", []))
        courts = list(mapping.get("courts_forum", []))
        documents = list(mapping.get("documents_required", []))

        immediate_actions = process_steps[:2] or [
            "Ensure immediate personal safety and collect essential evidence.",
            "Document incident details with timeline.",
        ]
        legal_actions = process_steps[2:] or [
            "Approach the proper authority/court with a written complaint.",
            "Consult legal aid for representation and filing strategy.",
        ]

        severity_level = self._severity_from_claim(claim_type=claim_type, urgency=urgency, query=query)
        severity_rationale = (
            f"Severity marked as {severity_level} due to claim category '{claim_type}', urgency '{urgency}', "
            "and reported risk context."
        )
        time_sensitivity = "Immediate action advised (within hours)." if severity_level in {"high", "critical"} else "Action advised within 24-48 hours."

        flowchart = self._build_flowchart(
            immediate_actions=immediate_actions,
            legal_actions=legal_actions,
            process_steps=process_steps,
        )

        summary = self._tts_summary(
            query=query,
            claim_type=claim_type,
            urgency=urgency,
            severity_level=severity_level,
            immediate_actions=immediate_actions,
            legal_actions=legal_actions,
            process_steps=process_steps,
            mandatory_documents=documents,
            courts=courts,
        )

        return StructuredLegalOutput(
            case_scenario=CaseScenario(
                summary=f"This appears to be a {claim_type.replace('_', ' ')} scenario in Maharashtra requiring structured legal action.",
                key_facts=key_facts,
                missing_details=missing_details,
            ),
            possible_steps=PossibleSteps(
                immediate_actions=immediate_actions,
                legal_actions=legal_actions,
                next_48_hours=[
                    "Consult District Legal Services Authority for guided filing support.",
                    "Prepare documentary bundle and maintain a dated chronology.",
                ],
            ),
            required_documentation=RequiredDocumentation(
                mandatory=documents[:4] if documents else ["Identity proof", "Primary incident evidence"],
                supporting=documents[4:] if len(documents) > 4 else ["Witness details", "Additional communication records"],
                optional=["Any prior legal notices", "Affidavit-style chronology"],
            ),
            courts_and_filing_process=CourtsAndProcess(
                state=self.knowledge.state,
                courts_forum=courts,
                application_process=process_steps,
                jurisdiction_note=(
                    f"Jurisdiction generally depends on where the cause of action arose in {self.knowledge.state} "
                    "or where parties reside/work."
                ),
            ),
            severity_assessment=SeverityAssessment(
                level=severity_level,
                rationale=severity_rationale,
                time_sensitivity=time_sensitivity,
            ),
            helplines_india=[HelplineInfo(**h) for h in self.knowledge.national_helplines],
            flowchart=flowchart,
            tts_summary=summary,
        )

    def _severity_from_claim(self, claim_type: str, urgency: str, query: str) -> str:
        lowered = query.lower()
        if any(x in lowered for x in ("kill", "weapon", "strang", "suicide", "immediate danger")):
            return "critical"
        if claim_type in {"domestic_violence", "police_harassment"}:
            return "high"
        if urgency == "high":
            return "high"
        if urgency == "medium":
            return "medium"
        return "low"

    def _build_flowchart(
        self,
        immediate_actions: list[str],
        legal_actions: list[str],
        process_steps: list[str],
    ) -> list[FlowchartStep]:
        items = immediate_actions[:1] + legal_actions[:2] + process_steps[:2]
        if not items:
            items = [
                "Document facts and evidence",
                "Approach proper authority",
                "File application/complaint",
                "Attend hearings and comply with orders",
            ]

        steps: list[FlowchartStep] = []
        for idx, text in enumerate(items, start=1):
            steps.append(
                FlowchartStep(
                    step=idx,
                    title=f"Step {idx}",
                    details=text,
                )
            )
        return steps

    def _tts_summary(
        self,
        query: str,
        claim_type: str,
        urgency: str,
        severity_level: str,
        immediate_actions: list[str],
        legal_actions: list[str],
        process_steps: list[str],
        mandatory_documents: list[str],
        courts: list[str],
    ) -> str:
        del query, urgency, severity_level

        next_steps = (immediate_actions + legal_actions)[:3]
        steps_text = "; ".join(next_steps) if next_steps else (
            "Collect evidence, prepare a dated timeline, and seek legal aid support"
        )

        docs_text = ", ".join(mandatory_documents[:4]) if mandatory_documents else "identity proof and incident-related records"

        if process_steps:
            process_text = " then ".join(process_steps[:2])
        elif courts:
            process_text = f"Approach {courts[0]} and submit a written complaint with acknowledgement"
        else:
            process_text = "Submit a written complaint to the appropriate authority and track acknowledgement"

        emergency_text = "Call 112 immediately if there is physical danger, and use legal aid helplines for urgent support"

        return (
            f"For your {claim_type.replace('_', ' ')} case, here is what to do now. "
            f"Next steps: {steps_text}. "
            f"Keep these documents ready: {docs_text}. "
            f"Process to follow: {process_text}. "
            f"Emergency support: {emergency_text}."
        )


class VeridictionGraph:
    """End-to-end graph orchestration with structured legal response."""

    def __init__(self, top_k: int = 5, advisor_provider: str = "auto") -> None:
        self.top_k = top_k
        self.classifier = ClaimClassifier()
        self.retriever = LegalRetriever()
        self.knowledge = LegalKnowledgeBase()
        self.structured_advisor = StructuredAdvisor(self.knowledge, provider=advisor_provider)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(VeridictionState)
        builder.add_node("retriever", self.retriever_node)
        builder.add_node("advisor", self.advisor_node)
        builder.add_node("safety", self.safety_node)
        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "advisor")
        builder.add_edge("advisor", "safety")
        builder.add_edge("safety", END)
        return builder.compile()

    def retriever_node(self, state: VeridictionState) -> VeridictionState:
        node_start = time.perf_counter()
        query = state.get("user_query", "")
        classify_start = time.perf_counter()
        claim = self.classifier.classify(query)
        classify_ms = (time.perf_counter() - classify_start) * 1000
        retrieve_start = time.perf_counter()
        passages = self.retriever.query(query, top_k=self.top_k)
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000
        retrieval_route = "judgment_priority"
        query_variants: list[str] = []
        if passages:
            metadata = passages[0].get("metadata", {}) or {}
            retrieval_route = str(metadata.get("retrieval_route", retrieval_route))
            variants_raw = str(metadata.get("query_variants", "")).strip()
            if variants_raw:
                query_variants = [v.strip() for v in variants_raw.split(";") if v.strip()]

        timings = dict(state.get("node_latencies_ms", {}) or {})
        timings.update(
            {
                "classifier_ms": round(classify_ms, 2),
                "retrieval_ms": round(retrieve_ms, 2),
                "retriever_node_ms": round((time.perf_counter() - node_start) * 1000, 2),
            }
        )

        return {
            "claim": claim,
            "retrieved_passages": passages,
            "retrieval_route": retrieval_route,
            "retrieval_query_variants": query_variants,
            "node_latencies_ms": timings,
        }

    def advisor_node(self, state: VeridictionState) -> VeridictionState:
        node_start = time.perf_counter()
        query = state.get("user_query", "")
        claim = state.get("claim", {})
        passages = state.get("retrieved_passages", [])
        retrieval_route = str(state.get("retrieval_route", "judgment_priority"))
        query_variants = list(state.get("retrieval_query_variants", []) or [])
        top_score = max((float(p.get("score", 0.0)) for p in passages), default=0.0)
        grounding_status = "grounded" if top_score >= 0.84 else "low_grounding"

        advisor_start = time.perf_counter()
        structured = self.structured_advisor.generate(query=query, claim=claim, passages=passages)
        advisor_ms = (time.perf_counter() - advisor_start) * 1000
        advisor_output = self._legacy_advisor_output(structured, passages)
        structured_output = structured.model_dump()
        structured_output["missing_facts_followups"] = self._missing_facts_followups(query=query, claim=claim)
        structured_output["section_citations"] = self._section_citations(passages=passages)
        structured_output["retrieval_context"] = {
            "route": retrieval_route,
            "query_variants": query_variants,
            "top_score": round(top_score, 4),
            "grounding_status": grounding_status,
        }

        timings = dict(state.get("node_latencies_ms", {}) or {})
        timings.update(
            {
                "advisor_generation_ms": round(advisor_ms, 2),
                "advisor_node_ms": round((time.perf_counter() - node_start) * 1000, 2),
            }
        )

        return {
            "structured_output": structured_output,
            "advisor_output": advisor_output,
            "node_latencies_ms": timings,
        }

    def safety_node(self, state: VeridictionState) -> VeridictionState:
        node_start = time.perf_counter()
        query = state.get("user_query", "").lower()
        claim = state.get("claim", {})
        advisor_output = state.get("advisor_output", {})
        structured = dict(state.get("structured_output", {}) or {})
        risk_flags = self._risk_flags(query=query, claim=claim)
        retrieval_context = structured.get("retrieval_context", {}) or {}
        if retrieval_context.get("grounding_status") == "low_grounding":
            risk_flags.append("limited_grounding")

        safe_steps: list[str] = []
        if "immediate_danger" in risk_flags:
            safe_steps.append("Seek immediate physical safety and call 112 emergency services.")
        if "domestic_violence_risk" in risk_flags:
            safe_steps.append("Use women safety helplines 181/1091 and approach nearest police station.")
        if "police_misconduct" in risk_flags:
            safe_steps.append("Escalate complaint to SP/Commissioner and seek legal aid support.")
        if not safe_steps:
            safe_steps.append("Follow documented legal steps and seek legal aid for filing strategy.")

        safety_output = SafetyOutput(
            risk_flags=risk_flags,
            safe_next_steps=safe_steps,
            disclaimer=MANDATORY_DISCLAIMER,
        )

        if structured:
            sev = structured.get("severity_assessment", {}) or {}
            if "immediate_danger" in risk_flags:
                sev["level"] = "critical"
                sev["time_sensitivity"] = "Immediate action required within minutes/hours."
            elif claim.get("urgency") == "high" and sev.get("level") not in {"critical", "high"}:
                sev["level"] = "high"
            structured["severity_assessment"] = sev

        final_response = {
            "claim_type": claim.get("claim_type", "other"),
            "urgency": claim.get("urgency", "low"),
            "confidence": claim.get("confidence", 0.0),
            "intent_labels": claim.get("intent_labels", []),
            "intent_scores": claim.get("intent_scores", {}),
            "retrieval_route": state.get("retrieval_route", "judgment_priority"),
            "retrieval_query_variants": state.get("retrieval_query_variants", []),
            "retrieved_passages": state.get("retrieved_passages", []),
            "advisor": advisor_output,
            "structured_response": structured,
            "missing_facts_followups": structured.get("missing_facts_followups", []),
            "section_citations": structured.get("section_citations", {}),
            "safety": safety_output.model_dump(),
            "tts_summary": structured.get("tts_summary", ""),
            "node_latencies_ms": state.get("node_latencies_ms", {}),
            "final_text": self._compose_final_text(structured=structured, safety=safety_output),
        }

        timings = dict(state.get("node_latencies_ms", {}) or {})
        timings["safety_node_ms"] = round((time.perf_counter() - node_start) * 1000, 2)
        final_response["node_latencies_ms"] = timings

        return {
            "safety_output": safety_output.model_dump(),
            "node_latencies_ms": timings,
            "final_response": final_response,
        }

    def _legacy_advisor_output(self, structured: StructuredLegalOutput, passages: list[dict[str, Any]]) -> dict[str, Any]:
        legal_basis = [item.get("passage", "")[:220] for item in passages[:3] if item.get("passage")]
        return AdvisorOutput(
            issue_summary=structured.case_scenario.summary,
            action_steps=(
                structured.possible_steps.immediate_actions[:2]
                + structured.possible_steps.legal_actions[:3]
            ),
            legal_basis=legal_basis,
            documents_to_collect=structured.required_documentation.mandatory,
            escalation_guidance=structured.severity_assessment.time_sensitivity,
        ).model_dump()

    def _risk_flags(self, query: str, claim: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if any(token in query for token in ("danger", "assault", "violence", "threat", "urgent", "tonight")):
            flags.append("immediate_danger")
        if claim.get("claim_type") == "police_harassment":
            flags.append("police_misconduct")
        if claim.get("claim_type") == "domestic_violence":
            flags.append("domestic_violence_risk")
        if claim.get("urgency") == "high" and "high_urgency" not in flags:
            flags.append("high_urgency")
        return flags

    def _compose_final_text(self, structured: dict[str, Any], safety: SafetyOutput) -> str:
        scenario = structured.get("case_scenario", {}) or {}
        steps = structured.get("possible_steps", {}) or {}
        docs = structured.get("required_documentation", {}) or {}
        filing = structured.get("courts_and_filing_process", {}) or {}
        severity = structured.get("severity_assessment", {}) or {}
        helplines = structured.get("helplines_india", []) or []
        flowchart = structured.get("flowchart", []) or []

        lines: list[str] = []
        lines.append("Case Scenario:")
        lines.append(f"- {scenario.get('summary', '')}")

        key_facts = scenario.get("key_facts", []) or []
        if key_facts:
            lines.append("- Key facts:")
            for fact in key_facts:
                lines.append(f"  * {fact}")

        lines.append("Possible Steps:")
        for item in steps.get("immediate_actions", []) or []:
            lines.append(f"- Immediate: {item}")
        for item in steps.get("legal_actions", []) or []:
            lines.append(f"- Legal: {item}")

        lines.append("Required Documentation:")
        for item in docs.get("mandatory", []) or []:
            lines.append(f"- Mandatory: {item}")

        lines.append("Courts and Filing Process:")
        for forum in filing.get("courts_forum", []) or []:
            lines.append(f"- Forum: {forum}")
        for idx, proc in enumerate(filing.get("application_process", []) or [], start=1):
            lines.append(f"- Process {idx}: {proc}")

        lines.append("Severity:")
        lines.append(f"- Level: {severity.get('level', 'medium')}")
        lines.append(f"- Rationale: {severity.get('rationale', '')}")

        lines.append("Helplines (India):")
        for h in helplines[:4]:
            lines.append(f"- {h.get('name', '')}: {h.get('number', '')} ({h.get('availability', '')})")

        lines.append("Flowchart:")
        for step in flowchart:
            lines.append(f"- Step {step.get('step', '')}: {step.get('details', '')}")

        if safety.risk_flags:
            lines.append(f"Risk flags: {', '.join(safety.risk_flags)}")

        followups = structured.get("missing_facts_followups", []) or []
        if followups:
            lines.append("Follow-up questions:")
            for q in followups[:4]:
                lines.append(f"- {q}")

        lines.append(safety.disclaimer)
        return "\n".join(lines)

    def _missing_facts_followups(self, query: str, claim: dict[str, Any]) -> list[str]:
        claim_type = str(claim.get("claim_type", "other"))
        lowered = query.lower()

        generic = [
            "What exact date and location did the key incident occur?",
            "Who are the involved parties and their relationship to you?",
            "Do you already have any written complaint, notice, or FIR reference number?",
        ]

        per_claim: dict[str, list[str]] = {
            "domestic_violence": [
                "Are there recent threats or injuries requiring immediate police or medical help?",
                "Do you have medical reports, photos, chats, or witness details?",
                "Do you need immediate shelter or protection order support?",
            ],
            "police_harassment": [
                "Which police station and officers were involved?",
                "Was any written refusal to register FIR given?",
                "Do you have call logs, recordings, or witness names for misconduct?",
            ],
            "unpaid_wages": [
                "What is the exact salary due amount and period pending?",
                "Do you have salary slips, bank statements, and appointment proof?",
                "Did you send a written demand/notice to the employer?",
            ],
            "tenant_rights": [
                "Do you have rent agreement, rent receipts, and deposit payment proof?",
                "Was any eviction notice served and with what timeline?",
                "Are utilities access or lockout actions currently happening?",
            ],
        }

        followups = list(generic)
        followups.extend(per_claim.get(claim_type, []))

        if not any(token in lowered for token in ("date", "day", "month", "year")):
            followups.append("Please share the complete timeline with dates in chronological order.")
        if not any(token in lowered for token in ("document", "proof", "evidence", "record")):
            followups.append("What documents or digital evidence can you share right now?")

        # Preserve order while deduplicating and keep concise.
        unique: list[str] = []
        seen: set[str] = set()
        for item in followups:
            key = item.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:6]

    def _section_citations(self, passages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        def _cite(item: dict[str, Any]) -> dict[str, Any]:
            metadata = dict(item.get("metadata", {}) or {})
            dataset = str(metadata.get("dataset", ""))
            source_label = str(metadata.get("source_label", "Judgment Index"))
            snippet = str(item.get("passage", "")).replace("\n", " ").strip()
            return {
                "source_label": source_label,
                "dataset": dataset,
                "score": round(float(item.get("score", 0.0)), 4),
                "snippet": snippet[:320] + ("..." if len(snippet) > 320 else ""),
            }

        ranked = sorted(passages, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        top = ranked[:6]

        buckets: dict[str, list[dict[str, Any]]] = {
            "case_scenario": [],
            "possible_steps": [],
            "required_documentation": [],
            "courts_and_filing_process": [],
        }

        for idx, item in enumerate(top):
            citation = _cite(item)
            if idx < 2:
                buckets["case_scenario"].append(citation)

            text = str(item.get("passage", "")).lower()
            if any(token in text for token in ("step", "process", "file", "complaint", "petition")):
                buckets["possible_steps"].append(citation)
            if any(token in text for token in ("document", "evidence", "proof", "records", "receipt")):
                buckets["required_documentation"].append(citation)
            if any(token in text for token in ("court", "forum", "jurisdiction", "magistrate", "tribunal")):
                buckets["courts_and_filing_process"].append(citation)

        for key in buckets:
            if not buckets[key] and top:
                buckets[key] = [_cite(top[0])]
            buckets[key] = buckets[key][:2]

        return buckets

    def run(self, user_query: str) -> dict[str, Any]:
        run_start = time.perf_counter()
        initial_state: VeridictionState = {"user_query": user_query}
        result = self.graph.invoke(initial_state)
        final = result.get("final_response", {})
        total_ms = (time.perf_counter() - run_start) * 1000
        node_latencies = dict(final.get("node_latencies_ms", {}) or {})
        node_latencies["total_pipeline_ms"] = round(total_ms, 2)
        final["node_latencies_ms"] = node_latencies
        return final


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Veridiction LangGraph flow")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None, help="Optional audio file input for Step 4")
    parser.add_argument("--live-mic", action="store_true", help="Capture real-time microphone input")
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=0,
        help="If > 0, records microphone audio and transcribes before running flow",
    )
    parser.add_argument("--record-out", type=str, default="data/audio_recorded.wav")
    parser.add_argument("--audio-max-seconds", type=int, default=30)
    parser.add_argument("--audio-silence-threshold", type=float, default=0.01)
    parser.add_argument("--audio-silence-seconds", type=float, default=2.0)
    parser.add_argument("--audio-model-name", type=str, default="distil-large-v3")
    parser.add_argument("--audio-model-dir", type=str, default="data/models/faster-whisper")
    parser.add_argument("--audio-local-files-only", action="store_true")
    parser.add_argument("--audio-language", type=str, default="en")
    parser.add_argument("--audio-beam-size", type=int, default=5)
    parser.add_argument("--audio-no-vad", action="store_true")
    parser.add_argument("--advisor-provider", type=str, default="auto", choices=["auto", "grok", "fallback"])
    parser.add_argument("--enable-tts", action="store_true", help="Generate Step 5 TTS audio output")
    parser.add_argument("--tts-output", type=str, default="data/tts/final_response.mp3")
    parser.add_argument("--tts-engine", type=str, default="edge_tts", choices=["edge_tts", "pyttsx3"])
    parser.add_argument("--tts-fallback-engine", type=str, default="pyttsx3", choices=["pyttsx3", "edge_tts"])
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_cli().parse_args()

    query_text = args.query
    audio_metadata: dict[str, Any] | None = None

    if args.live_mic or args.audio_file or args.record_seconds > 0:
        from audio.transcriber import (
            AudioTranscriber,
            TranscriberConfig,
            record_microphone_live_to_wav,
            record_microphone_to_wav,
        )

        input_audio = args.audio_file
        if args.live_mic:
            recorded_file = record_microphone_live_to_wav(
                output_wav=args.record_out,
                sample_rate=16000,
                channels=1,
                max_seconds=args.audio_max_seconds,
                silence_threshold=args.audio_silence_threshold,
                silence_seconds=args.audio_silence_seconds,
                enable_enter_to_stop=True,
            )
            input_audio = str(recorded_file)
        elif args.record_seconds > 0:
            recorded_file = record_microphone_to_wav(
                output_wav=args.record_out,
                duration_seconds=args.record_seconds,
                sample_rate=16000,
                channels=1,
            )
            input_audio = str(recorded_file)

        if not input_audio:
            raise ValueError("Audio mode requires --live-mic, --audio-file, or --record-seconds > 0")

        transcriber = AudioTranscriber(
            config=TranscriberConfig(
                model_name=args.audio_model_name,
                model_dir=args.audio_model_dir,
                local_files_only=args.audio_local_files_only,
                language=args.audio_language,
                beam_size=args.audio_beam_size,
                vad_filter=not args.audio_no_vad,
            )
        )
        transcription = transcriber.transcribe_file(
            audio_path=input_audio,
            language=args.audio_language,
            beam_size=args.audio_beam_size,
            vad_filter=not args.audio_no_vad,
        )
        query_text = transcription.get("text", "").strip()
        audio_metadata = {
            "audio_file": input_audio,
            "language": transcription.get("language", args.audio_language),
            "language_probability": transcription.get("language_probability", 0.0),
            "duration": transcription.get("duration", 0.0),
            "segments": len(transcription.get("segments", [])),
        }

    if audio_metadata is not None and not query_text:
        raise ValueError("Transcription produced empty text. Try clearer audio or disable background noise.")

    if not query_text:
        query_text = "My employer has not paid my salary for 3 months."

    flow = VeridictionGraph(top_k=args.top_k, advisor_provider=args.advisor_provider)
    output = flow.run(query_text)

    if args.enable_tts:
        from tts.speak import TTSConfig, TTSGenerator

        tts_generator = TTSGenerator(
            config=TTSConfig(
                preferred_engine=args.tts_engine,
                fallback_engine=args.tts_fallback_engine,
                output_dir="data/tts",
            )
        )
        text_for_tts = output.get("tts_summary") or output.get("final_text", "")
        tts_result = tts_generator.speak_to_file(
            text=text_for_tts,
            output_path=args.tts_output,
            include_disclaimer=False,
        )
        output["tts"] = {
            "engine": tts_result["engine"],
            "audio_path": tts_result["audio_path"],
            "mime_type": tts_result["mime_type"],
            "size_bytes": tts_result["size_bytes"],
            "spoken_text": text_for_tts,
        }

    if audio_metadata is not None:
        output["input_mode"] = "audio"
        output["transcript"] = query_text
        output["audio_metadata"] = audio_metadata
    else:
        output["input_mode"] = "text"
        output["transcript"] = query_text

    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

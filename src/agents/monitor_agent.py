from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class MonitorState(TypedDict, total=False):
    session_id: str
    client_ip: str
    event_type: str | None
    severity: str
    detail: str

    # Persisted agent state
    initial_ip: str | None
    current_ip: str | None
    ip_history: list[str]
    ip_consistent: bool
    ip_change_count: int
    camera_status: str
    focus_status: str
    alert_count: int
    events: list[dict[str, Any]]
    last_seen: str | None
    last_ip_change: str | None

    # Current graph run
    observation: dict[str, Any]
    decision: str
    decision_reason: str
    action: str
    action_severity: str


@tool
def register_ip(session: dict[str, Any], client_ip: str) -> dict[str, Any]:
    """Register the first observed client IP for an assessment session."""
    if not client_ip or client_ip == "unknown":
        return session

    if session.get("initial_ip") is None:
        session["initial_ip"] = client_ip

    session["current_ip"] = client_ip
    history = session.setdefault("ip_history", [])
    if client_ip not in history:
        history.append(client_ip)
    return session


@tool
def record_integrity_event(
    session: dict[str, Any],
    event_type: str,
    severity: str,
    detail: str,
    client_ip: str,
    max_events: int = 100,
) -> dict[str, Any]:
    """Record one observable assessment-integrity event."""
    session.setdefault("events", []).append(
        {
            "id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "detail": detail,
            "client_ip": client_ip,
        }
    )
    session["events"] = session["events"][-max_events:]
    return session


@tool
def raise_integrity_flag(session: dict[str, Any]) -> dict[str, Any]:
    """Raise one assessment-integrity flag."""
    session["alert_count"] = session.get("alert_count", 0) + 1
    return session


@tool
def update_monitor_state(session: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply a validated monitoring-state update."""
    session.update(updates)
    return session


class MonitorRun(TypedDict):
    agent: str
    graph: list[str]
    observation: dict[str, Any]
    decision: str
    decision_reason: str
    action: str
    severity: str


class MonitorAgent:
    """Stateful LangGraph agent for assessment/session monitoring.

    One graph execution is a complete agent cycle:

        OBSERVE -> DECIDE -> ACT -> UPDATE STATE -> END

    LangChain tools perform the state-changing actions. LangGraph controls the
    agent state and node transitions. The agent deliberately uses deterministic
    guardrails for IP/session integrity because those decisions should not be
    delegated to an LLM.
    """

    def __init__(self, max_events: int = 100) -> None:
        self.max_events = max_events
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MonitorState)
        workflow.add_node("observe", self._observe)
        workflow.add_node("decide", self._decide)
        workflow.add_node("act", self._act)
        workflow.add_node("update_state", self._update_state)

        workflow.add_edge(START, "observe")
        workflow.add_edge("observe", "decide")
        workflow.add_edge("decide", "act")
        workflow.add_edge("act", "update_state")
        workflow.add_edge("update_state", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ---------------- LangGraph nodes ----------------
    def _observe(self, state: MonitorState) -> dict[str, Any]:
        observation = {
            "session_id": state["session_id"],
            "incoming_ip": state.get("client_ip"),
            "previous_ip": state.get("current_ip"),
            "initial_ip": state.get("initial_ip"),
            "event_type": state.get("event_type"),
            "event_severity": state.get("severity", "info"),
            "detail": state.get("detail", ""),
            "camera_status": state.get("camera_status", "NOT_STARTED"),
            "focus_status": state.get("focus_status", "ACTIVE"),
            "alert_count": state.get("alert_count", 0),
            "ip_history": list(state.get("ip_history", [])),
        }
        return {"observation": observation}

    def _decide(self, state: MonitorState) -> dict[str, Any]:
        obs = state["observation"]
        incoming_ip = obs.get("incoming_ip")
        previous_ip = obs.get("previous_ip")
        event_type = obs.get("event_type")
        severity = obs.get("event_severity", "info")

        # Hard safety/consistency rules are the authoritative decision layer.
        if not incoming_ip or incoming_ip == "unknown":
            decision = "IP_UNAVAILABLE"
            reason = "The API could not determine the client network address."
            action = "REVIEW_REQUIRED"
            action_severity = "warning"
        elif previous_ip is None:
            decision = "INITIAL_IP"
            reason = "This is the first network address observed for this session."
            action = "REGISTER_IP"
            action_severity = "info"
        elif incoming_ip != previous_ip:
            decision = "IP_CHANGED"
            reason = "The observed client network address changed during the session."
            action = "FLAG_IP_CHANGE"
            action_severity = "warning"
        elif event_type and severity == "warning":
            decision = "INTEGRITY_WARNING"
            reason = obs.get("detail") or f"The event {event_type} requires attention."
            action = "FLAG_INTEGRITY_EVENT"
            action_severity = "warning"
        elif event_type:
            decision = "EVENT_OBSERVED"
            reason = obs.get("detail") or f"Recorded monitoring event {event_type}."
            action = "RECORD_EVENT"
            action_severity = "info"
        else:
            decision = "IP_STABLE"
            reason = "The observed client network address is unchanged."
            action = "CONTINUE_MONITORING"
            action_severity = "info"

        return {
            "decision": decision,
            "decision_reason": reason,
            "action": action,
            "action_severity": action_severity,
        }

    def _act(self, state: MonitorState) -> dict[str, Any]:
        # The graph's ACT node invokes actual LangChain tools. These are not
        # placeholders: they mutate the session state that the API exposes.
        session = {
            "initial_ip": state.get("initial_ip"),
            "current_ip": state.get("current_ip"),
            "ip_history": list(state.get("ip_history", [])),
            "ip_consistent": state.get("ip_consistent", False),
            "ip_change_count": state.get("ip_change_count", 0),
            "camera_status": state.get("camera_status", "NOT_STARTED"),
            "focus_status": state.get("focus_status", "ACTIVE"),
            "alert_count": state.get("alert_count", 0),
            "events": list(state.get("events", [])),
            "last_seen": state.get("last_seen"),
            "last_ip_change": state.get("last_ip_change"),
        }

        now = datetime.now(timezone.utc).isoformat()
        # LangChain tool invocation returns a validated state dictionary;
        # always capture that return value so scalar fields such as current_ip
        # are not lost while shared list references are updated.
        incoming_ip = state.get("client_ip", "unknown")
        event_type = state.get("event_type")
        severity = state.get("severity", "info")
        detail = state.get("detail", "")
        action = state["action"]

        if action == "REGISTER_IP":
            session = register_ip.invoke({"session": session, "client_ip": incoming_ip})
            session = update_monitor_state.invoke(
                {"session": session, "updates": {"ip_consistent": True, "last_seen": now}}
            )
        elif action == "FLAG_IP_CHANGE":
            session = register_ip.invoke({"session": session, "client_ip": incoming_ip})
            session = raise_integrity_flag.invoke({"session": session})
            session = record_integrity_event.invoke(
                {
                    "session": session,
                    "event_type": "ip_changed",
                    "severity": "warning",
                    "detail": "Client network address changed during the assessment session.",
                    "client_ip": incoming_ip,
                    "max_events": self.max_events,
                }
            )
            session = update_monitor_state.invoke(
                {
                    "session": session,
                    "updates": {
                        "ip_consistent": False,
                        "ip_change_count": session.get("ip_change_count", 0) + 1,
                        "last_ip_change": now,
                        "last_seen": now,
                    },
                }
            )
        elif action == "FLAG_INTEGRITY_EVENT":
            session = raise_integrity_flag.invoke({"session": session})
            if event_type:
                session = record_integrity_event.invoke(
                    {
                        "session": session,
                        "event_type": event_type,
                        "severity": severity,
                        "detail": detail,
                        "client_ip": incoming_ip,
                        "max_events": self.max_events,
                    }
                )
            session = update_monitor_state.invoke(
                {"session": session, "updates": {"last_seen": now}}
            )
        else:
            if event_type:
                session = record_integrity_event.invoke(
                    {
                        "session": session,
                        "event_type": event_type,
                        "severity": severity,
                        "detail": detail,
                        "client_ip": incoming_ip,
                        "max_events": self.max_events,
                    }
                )

            if incoming_ip and incoming_ip != "unknown":
                session = register_ip.invoke({"session": session, "client_ip": incoming_ip})

            session = update_monitor_state.invoke(
                {
                    "session": session,
                    "updates": {
                        "ip_consistent": bool(incoming_ip and incoming_ip != "unknown"),
                        "last_seen": now,
                    },
                }
            )

            status_updates: dict[str, Any] = {}
            if event_type == "camera_started":
                status_updates["camera_status"] = "ACTIVE"
            elif event_type in {"camera_stopped", "camera_unavailable"}:
                status_updates["camera_status"] = "INTERRUPTED"
            elif event_type == "page_focus_lost":
                status_updates["focus_status"] = "LOST"
            elif event_type == "page_focus_returned":
                status_updates["focus_status"] = "ACTIVE"

            if status_updates:
                session = update_monitor_state.invoke({"session": session, "updates": status_updates})

        return {
            "initial_ip": session.get("initial_ip"),
            "current_ip": session.get("current_ip"),
            "ip_history": session.get("ip_history", []),
            "ip_consistent": session.get("ip_consistent", False),
            "ip_change_count": session.get("ip_change_count", 0),
            "camera_status": session.get("camera_status", "NOT_STARTED"),
            "focus_status": session.get("focus_status", "ACTIVE"),
            "alert_count": session.get("alert_count", 0),
            "events": session.get("events", []),
            "last_seen": session.get("last_seen"),
            "last_ip_change": session.get("last_ip_change"),
        }

    def _update_state(self, state: MonitorState) -> dict[str, Any]:
        return {
            "graph_status": "completed",
            "agent_cycle": ["observe", "decide", "act", "update_state"],
        }

    def run(
        self,
        session: dict[str, Any],
        *,
        client_ip: str,
        event_type: str | None = None,
        severity: str = "info",
        detail: str = "",
        session_id: str = "unknown",
    ) -> MonitorRun:
        # Start from the persisted session state, then overwrite the
        # per-event inputs with the values supplied for this graph run.
        # The previous order let session["client_ip"] (often None) overwrite
        # the actual IP observed by FastAPI, which caused IP_UNAVAILABLE and
        # ultimately failed the LangChain tool validation.
        initial = {
            **session,
            "session_id": session_id,
            "client_ip": client_ip,
            "event_type": event_type,
            "severity": severity,
            "detail": detail,
        }

        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial, config)

        # The graph owns the decisions; the API session store receives its
        # resulting state so other endpoints/UI can observe it immediately.
        for key in (
            "initial_ip", "current_ip", "ip_history", "ip_consistent",
            "ip_change_count", "camera_status", "focus_status", "alert_count",
            "events", "last_seen", "last_ip_change",
        ):
            if key in result:
                session[key] = result[key]

        return {
            "agent": "MonitorAgent",
            "graph": result.get("agent_cycle", ["observe", "decide", "act", "update_state"]),
            "observation": result.get("observation", {}),
            "decision": result.get("decision", "UNKNOWN"),
            "decision_reason": result.get("decision_reason", ""),
            "action": result.get("action", "UNKNOWN"),
            "severity": result.get("action_severity", "info"),
        }

    def snapshot(self, session: dict[str, Any]) -> dict[str, Any]:
        client_ip = session.get("current_ip") or session.get("initial_ip")
        if client_ip is None:
            status = "WAITING_FOR_IP"
        elif session.get("ip_consistent"):
            status = "IP_STABLE"
        else:
            status = "REVIEW_REQUIRED"

        return {
            "agent": "MonitorAgent",
            "framework": "LangGraph + LangChain tools",
            "mode": "stateful_observe_decide_act",
            "status": status,
            "client_ip": client_ip,
            "initial_ip": session.get("initial_ip"),
            "ip_history": list(session.get("ip_history", [])),
            "ip_consistent": session.get("ip_consistent", False),
            "ip_change_count": session.get("ip_change_count", 0),
            "camera_status": session.get("camera_status", "NOT_STARTED"),
            "focus_status": session.get("focus_status", "ACTIVE"),
            "alert_count": session.get("alert_count", 0),
            "last_seen": session.get("last_seen"),
        }

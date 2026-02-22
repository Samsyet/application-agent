from __future__ import annotations

from typing import cast, Any, Dict, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from hive_hook import (
    EndpointEnum,
    HiveInboundBaseData,
    hive_data,
    hive_hook,
    send_to_hive,
    start_server,
)


# ── Inbound Models ─────────────────────────────────────


class IdentityAgentResponse(HiveInboundBaseData):
    num_ones: int


class CreditAgentResponse(HiveInboundBaseData):
    num_zeroes: int


class RiskAgentResponse(HiveInboundBaseData):
    num_bits: int


# ── Graph State ────────────────────────────────────────


class State(TypedDict):
    binary_number: str
    num_ones: Optional[int]
    num_zeroes: Optional[int]
    num_bits: Optional[int]
    x: Optional[int]
    y: Optional[int]
    decision: Optional[int]


# ── Nodes ──────────────────────────────────────────────


async def initiate_check(state: State) -> Dict[str, Any]:
    binary_number = state["binary_number"]

    for agent_id in ("identity_agent", "credit_agent", "risk_agent"):
        await send_to_hive(
            destination_agent_id=agent_id,
            destination_agent_endpoint=EndpointEnum.START,
            payload={"binary_number": binary_number},
        )

    return {"binary_number": binary_number}


@hive_hook(
    {
        "identity": IdentityAgentResponse,
        "credit": CreditAgentResponse,
        "risk": RiskAgentResponse,
    }
)
async def await_responses(state: State) -> Dict[str, Any]:
    identity = hive_data.get("identity", IdentityAgentResponse)
    credit = hive_data.get("credit", CreditAgentResponse)
    risk = hive_data.get("risk", RiskAgentResponse)

    return {
        "num_ones": identity.num_ones,
        "num_zeroes": credit.num_zeroes,
        "num_bits": risk.num_bits,
    }


async def decide(state: State) -> Dict[str, Any]:
    num_ones = cast(int, state["num_ones"])
    num_zeroes = cast(int, state["num_zeroes"])
    num_bits = cast(int, state["num_bits"])

    x = 1 if num_ones > num_zeroes else 0
    y = 1 if num_bits % 2 == 0 else 0
    decision = x & y

    await send_to_hive(
        destination_agent_id="endfront",
        destination_agent_endpoint=EndpointEnum.EXTERNAL,
        payload=dict(state),
    )

    return {
        "x": x,
        "y": y,
        "decision": decision,
    }


# ── Graph ──────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("initiate_check", initiate_check)
graph.add_node("await_responses", await_responses)
graph.add_node("decide", decide)

graph.set_entry_point("initiate_check")

graph.add_edge("initiate_check", "await_responses")
graph.add_edge("await_responses", "decide")
graph.add_edge("decide", END)

compiled = graph.compile()


# ── Run Server ─────────────────────────────────────────

if __name__ == "__main__":
    start_server(
        compiled,
        agent_id="application_agent",
    )

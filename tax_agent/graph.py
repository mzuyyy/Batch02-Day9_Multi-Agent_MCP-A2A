"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA.

Answer tax questions concisely. Limit the response to 120 words or fewer.
Focus only on the most relevant tax consequences, such as:
- civil vs. criminal penalties
- IRS / DOJ Tax Division / FinCEN involvement
- company liability vs. executive liability
- key FBAR, FATCA, transfer pricing, or tax fraud issues when relevant

Use 2-4 short bullet points. Avoid long background explanations and avoid
repeating facts from the user's question. End with one brief educational-use
disclaimer.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph

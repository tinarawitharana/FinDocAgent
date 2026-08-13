"""Regenerates agent_flow.png, the LangGraph agent diagram used in the README/dissertation."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.graph import build_graph

agent = build_graph()

# Try PNG first
try:
    png_bytes = agent.get_graph().draw_mermaid_png()
    with open("agent_flow.png", "wb") as f:
        f.write(png_bytes)
    print("Saved: agent_flow.png")

except Exception as e:
    print(f"PNG failed ({e}), printing Mermaid instead:")
    print(agent.get_graph().draw_mermaid())

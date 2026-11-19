"""Minimal stand-in for the part of `langgraph` that brain/agent/graph.py uses (StateGraph + END), for the in-browser demo (Pyodide).
The real langgraph needs compiled dependencies (orjson, ormsgpack, xxhash, uuid_utils) that Pyodide does not ship, so it cannot be installed there.
This shim has the same semantics for what the agent needs: nodes return partial state updates (merged, last value wins), edges are fixed or
conditional on the state, execution starts at the entry point and stops at END. `web/tests/test_static_demo.py` runs the agent on the real
langgraph and on this shim over a fixed command list and requires identical replies."""

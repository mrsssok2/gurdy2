from __future__ import annotations

END = "__end__"


class StateGraph:
    def __init__(self, state_schema=None):
        self.nodes: dict = {}
        self.edges: dict[str, str] = {}
        self.cond: dict[str, tuple] = {}
        self.entry: str | None = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, src, dst):
        self.edges[src] = dst

    def add_conditional_edges(self, src, chooser, mapping=None):
        self.cond[src] = (chooser, mapping)

    def compile(self):
        return _Compiled(self)


class _Compiled:
    MAX_STEPS = 100          # same idea as LangGraph's recursion limit: a graph that never reaches END is a bug, not an infinite loop

    def __init__(self, g: StateGraph):
        self.g = g

    def invoke(self, state: dict) -> dict:
        g, node, state = self.g, self.g.entry, dict(state)
        for _ in range(self.MAX_STEPS):
            if node == END:
                return state
            update = g.nodes[node](state)
            if update:
                state.update(update)
            if node in g.cond:
                chooser, mapping = g.cond[node]
                key = chooser(state)
                node = mapping[key] if mapping else key
            else:
                node = g.edges[node]
        raise RuntimeError("graph did not reach END within 100 steps")

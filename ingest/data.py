import re
from typing import List, Tuple
from attrs import define, field
import penman
from penman import surface
from penman.graph import Graph

BLOCK_NAME = {"red", "yellow", "green", "blue", "purple"}
PRONOUNS = {
    "it",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
}

BLOCK_NAME_PATTERN = rf"\b({'|'.join(BLOCK_NAME)})\b"
PRONOUN_PATTERN = rf"\b({'|'.join(PRONOUNS)})\b"


@define
class ActionComponent:
    verb: str = field(default=None)
    obj: str = field(default=None)
    prep: str = field(default=None)
    loc: str = field(default=None)
    other_text: str = field(default=None)
    raw_text: str = field(default=None)

    @classmethod
    def from_str(cls, action_str: str):
        # annotation errors will break this string parsing method
        parts = action_str.split(" ", 1)
        if len(parts) == 2:
            comp, other = parts
        else:
            comp = parts[0]
            other = None
        comp_lst = [c for c in re.split(r"\(|\)|,", comp.strip()) if c][:4]
        return cls(*comp_lst, other_text=other, raw_text=action_str)


@define
class Action:
    annotation_id: str
    tier_id: str
    start_ts: float
    end_ts: float
    component: ActionComponent


@define
class GAMR:
    annotation_id: str
    tier_id: str
    start_ts: float
    end_ts: float
    penman_str: str

    def parse(self):
        g = penman.decode(self.penman_str)
        instances = {ins.source: ins.target for ins in g.instances()}
        args = {e.role[1:]: instances[e.target] for e in g.edges()}

        return instances[g.top], args


@define
class Utterance:
    id: int
    speaker_id: int
    start: float
    end: float
    text: str
    actions: List[Action] = field(factory=list)
    gamrs: List[GAMR] = field(factory=list)

    @property
    def is_researcher(self) -> bool:
        return self.speaker_id == 4

    @property
    def contain_block_name(self) -> List[Tuple]:
        matches = re.finditer(BLOCK_NAME_PATTERN, self.text.lower())
        return [(m.group(), *m.span()) for m in matches]

    @property
    def contain_pronouns(self) -> List[Tuple]:
        matches = re.finditer(PRONOUN_PATTERN, self.text.lower())
        return [(m.group(), *m.span()) for m in matches]

    @property
    def contain_block_str(self) -> bool:
        return "block" in self.text.lower()


@define
class Dialogue:
    group_id: int
    utterances: List[Utterance]


@define(repr=False)
class AMR:
    amr_graph: Graph

    def add_node(self, form: str, prefix: str = None) -> str:
        num = 0
        if prefix is None:
            prefix = form[0]
        node_id = prefix
        while node_id in self.amr_graph.variables():
            num += 1
            node_id = prefix + str(num)

        self.amr_graph.triples.append((node_id, ":instance", form))
        return node_id

    def add_edge(self, source: str, label: str, target: str, is_attribute: bool = False):
        if source in self.amr_graph.variables():
            if is_attribute or target in self.amr_graph.variables():
                self.amr_graph.triples.append((source, label, target))
        else:
            raise ValueError(f"{source} or {target} are not valid variables in the graph!")

    @classmethod
    def from_penman_str(cls, penman_str: str):
        return cls(penman.decode(penman_str))

    def __repr__(self):
        return penman.encode(self.amr_graph)


if __name__ == "__main__":
    pass

import xml.etree.ElementTree as ET
from typing import List
from ingest.data import Action, ActionComponent
from ingest.file_path import ACTION_PATH


def ingest_action_xml(group_id: int) -> List[Action]:
    time_slots = dict()
    actions = []
    tree = ET.parse(ACTION_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_action_adjudicated.eaf"))
    root = tree.getroot()
    for st in root.find("TIME_ORDER").findall("TIME_SLOT"):
        st_id = st.get("TIME_SLOT_ID")
        st_value = st.get("TIME_VALUE")
        time_slots[st_id] = int(st_value) / 1000

    for tier in root.findall("TIER"):
        tier_id = tier.get("TIER_ID")
        for annotation in tier:
            aligned = annotation.find("ALIGNABLE_ANNOTATION")
            annotation_id = aligned.get("ANNOTATION_ID")
            ts_ref1 = aligned.get("TIME_SLOT_REF1")
            ts_ref2 = aligned.get("TIME_SLOT_REF2")
            text = aligned.find("ANNOTATION_VALUE").text
            if not text:
                # ANNOTATION_VALUE could be empty?
                continue
            action = Action(
                annotation_id,
                tier_id,
                time_slots[ts_ref1],
                time_slots[ts_ref2],
                ActionComponent.from_str(text.strip()),
            )
            actions.append(action)
    return actions


if __name__ == "__main__":
    actions = ingest_action_xml(2)

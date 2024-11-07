import xml.etree.ElementTree as ET
from ingest.file_path import GAMR_PATH
from ingest.data import GAMR


def ingest_gamr_xml(group_id: int):
    gamrs = []
    time_slots = dict()
    tree = ET.parse(GAMR_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_GAMR_gold.eaf"))
    root = tree.getroot()
    for st in root.find("TIME_ORDER").findall("TIME_SLOT"):
        st_id = st.get("TIME_SLOT_ID")
        st_value = st.get("TIME_VALUE")
        time_slots[st_id] = int(st_value) / 1000

    for tier in root.findall("TIER"):
        tier_id = tier.get("TIER_ID")
        if "gold" not in tier_id:
            continue
        for annotation in tier:
            aligned = annotation.find("ALIGNABLE_ANNOTATION")
            annotation_id = aligned.get("ANNOTATION_ID")
            ts_ref1 = aligned.get("TIME_SLOT_REF1")
            ts_ref2 = aligned.get("TIME_SLOT_REF2")
            text = aligned.find("ANNOTATION_VALUE").text
            if not text:
                # ANNOTATION_VALUE could be empty?
                continue
            gamr = GAMR(
                annotation_id,
                tier_id,
                time_slots[ts_ref1],
                time_slots[ts_ref2],
                text.strip(),
            )
            gamrs.append(gamr)
    return gamrs


if __name__ == '__main__':
    gamrs = ingest_gamr_xml(1)

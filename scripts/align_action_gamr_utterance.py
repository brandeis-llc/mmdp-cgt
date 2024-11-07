from typing import List

from ingest.data import Dialogue, Utterance, Action, GAMR
from ingest.ingest_action import ingest_action_xml
from ingest.ingest_gamr import ingest_gamr_xml
from ingest.ingest_dialogue import ingest_dialogue, ingest_dped_dialogue, PHASE1_END_UTTERANCE, PHASE1_START_UTTERANCE


def overlap_sec(start1, end1, start2, end2):
    if start1 <= end2 and end1 >= start2:
        return min(end1, end2) - max(start1, start2)
    else:
        return 0


def align_action_utterance(actions: List[Action], utterances: List[Utterance]) -> List[Utterance]:
    for action in actions:
        overlapped = -1
        for i, utt in enumerate(utterances):
            new_overlapped = overlap_sec(action.start_ts, action.end_ts, utt.start, utt.end)
            if new_overlapped >= overlapped:
                overlapped = new_overlapped
            else:
                utterances[i - 1].actions.append(action)
                # align the action to the utterance that overlaps the most with the action
                break
            if overlapped > 0 and i == len(utterances) - 1:
                utterances[-1].actions.append(action)

    return utterances


def align_gamr_utterance(gamrs: List[GAMR], utterances: List[Utterance]) -> List[Utterance]:
    for gamr in gamrs:
        overlapped = -1
        for i, utt in enumerate(utterances):
            new_overlapped = overlap_sec(gamr.start_ts, gamr.end_ts, utt.start, utt.end)
            if new_overlapped >= overlapped:
                overlapped = new_overlapped
            else:
                utterances[i - 1].gamrs.append(gamr)
                # align the gamr to the utterance that overlaps the most with the gamr
                break
            if overlapped > 0 and i == len(utterances) - 1:
                utterances[-1].gamrs.append(gamr)

    return utterances


def prepare_aligned_utterance(group_number: int, *, use_dp: bool) -> List[Utterance]:
    if use_dp:
        dialogue = ingest_dped_dialogue(group_number)
    else:
        dialogue = ingest_dialogue(group_number)

    actions = ingest_action_xml(group_number)
    gamrs = ingest_gamr_xml(group_number)
    utterances = dialogue.utterances
    aligned_utterances = align_action_utterance(actions, utterances)
    aligned_utterances = align_gamr_utterance(gamrs, aligned_utterances)
    return aligned_utterances


if __name__ == '__main__':
    uttrs = prepare_aligned_utterance(1, use_dp=False)
    for utt in uttrs:
        if utt.speaker_id < 4 and PHASE1_START_UTTERANCE[1] <= utt.id <= PHASE1_END_UTTERANCE[1]:
            print(utt.text, utt.speaker_id)
            print([a.component.raw_text for a in utt.actions])
            print([g.parse() for g in utt.gamrs])
            print("=====================================")



from collections import defaultdict

from ingest.generate_history_from_cga import generate_cg_history
from scripts.align_action_gamr_utterance import prepare_aligned_utterance
from bisect import bisect_left


def is_relevant_utterances(utterance, *, text_only: bool):
    if text_only:
        if utterance.contain_block_name or utterance.contain_block_str or utterance.contain_pronouns:
            return True
    else:
        if utterance.contain_block_name or utterance.contain_block_str or utterance.contain_pronouns:
            return True
        # if utterance.actions:
        #     return True
            # for action in utterance.actions:
            #     if "block" in action.component.raw_text:
            #         return True
            # for gamr in utterance.gamrs:
            #     if "block" in " ".join(gamr.parse()[1].values()):
            #         return True

    return False


def utterance_to_input(utterance, text_only: bool, with_action: bool, with_gamr: bool, with_text: bool):
    inputs = []
    text = utterance.text
    text_input = (f"Participant {utterance.speaker_id} utterance: {text}", utterance.start)
    if with_text:
        inputs.append(text_input)
        if text_only:
            return inputs
    if with_action:
        for action in utterance.actions:
            if action.tier_id[-1].isdigit():
                action_input = (f"Participant {action.tier_id[-1]} action: {action.component.raw_text}", action.start_ts)
            else:
                action_input = (f"Scale state: {action.component.raw_text}", action.start_ts)
            inputs.append(action_input)
    if with_gamr:
        for gamr in utterance.gamrs:
            gamr_act, pairs = gamr.parse()
            if pairs["ARG2"] == "group":
                pairs["ARG2"] = "other participants"
            if gamr_act == "deixis-GA":
                gamr_input = (
                    f"Participant {pairs['ARG0'][-1]} gesture: point({pairs['ARG1'], pairs['ARG2']})", gamr.start_ts)
                inputs.append(gamr_input)
            elif gamr_act == "emblem-GA":
                gamr_input = (
                    f"Participant {pairs['ARG0'][-1]} gesture: confirm({pairs['ARG1'], pairs['ARG2']})", gamr.start_ts)
                inputs.append(gamr_input)
    return sorted(inputs, key=lambda x: x[1])


def get_evidences_for_cg_statement(group_id: int, text_only: bool, use_dp: bool, use_new: bool):
    history = generate_cg_history(group_id, use_new=use_new)
    utterances = prepare_aligned_utterance(group_id, use_dp=use_dp)
    statemant_ts = []
    for ts in history:
        cg = history[ts]
        if cg.cg_type == "ACCEPT":
            statemant_ts.append(ts)
    max_ts = max(statemant_ts)
    evidences = defaultdict(list)
    for utt in utterances:
        if utt.is_researcher:
            continue
        ts = utt.end
        if ts > max_ts:
            break
        if not is_relevant_utterances(utt, text_only=text_only) and ts not in history:
            continue
        idx = bisect_left(statemant_ts, ts, hi=len(statemant_ts))
        evidences[statemant_ts[idx]].append(utt)
    return evidences, history


if __name__ == '__main__':
    evidence, history = get_evidences_for_cg_statement(1, use_dp=False, text_only=True, use_new=False)
    for e in evidence:
        print(e, history[e].fbank)
        for u in evidence[e]:
            for l in (utterance_to_input(u, text_only=True, with_action=False, with_gamr=False, with_text=True)):
                print(l[0])
            print("---------------------------------")
        print("====================================")

# -*- coding: utf-8 -*-
"""
Script to read CPS annotation file
"""
import collections
import csv

from ingest import file_path
from ingest import ingest_dialogue

CPS_hierarchy = """
- CONST
    - SharesU(nderstanding)
        - Situation
        - CorrectSolutions
        - IncorrectSolutions
    - EstablishesCG
        - Confirms
        - Interrupts
- NEG
    - Responds
        - Reasons
        - QuestionsOthers
        - Responds
    - MonitorsE(xecution)
        - Results
        - Strategizes
        - Save
        - GivingUp
- MAINTAIN
    - Initiative
        - Suggestions
        - Compliments
        - Criticizes
    - FulfillsR(ole)
        - Support
        - Apologizes
        - InitiatesOffTopic
        - JoinsOffTopic
"""

dialogue_amr_sa_hierarchy = """
- Information Transfer Functions 
    - Question 
    - Assertion 
- Action-Discussion Functions 
    - Commissive 
        - Offer
        - Promise
    - Directive 
        - Command
        - Open-Option
        - Request
- Expressive
    - Accept/Reject
    - Greeting
    - Gratitude
    - Regret
    - Judgment
    - Mistake
    - Hold-Floor
"""

cps_to_sa_mapping = {
    "CPS_CONST_SharesU_Situation": "",
    "CPS_CONST_SharesU_CorrectSolutions": "",
    "CPS_CONST_SharesU_IncorrectSolutions": "",
    "CPS_CONST_EstablishesCG_Confirms": "",
    "CPS_CONST_EstablishesCG_Interrupts": "",
    "CPS_NEG_Responds_Reasons": "",
    "CPS_NEG_Responds_QuestionsOthers": "",
    "CPS_NEG_Responds_Responds": "",
    "CPS_NEG_MonitorsE_Results": "",
    "CPS_NEG_MonitorsE_Strategizes": "",
    "CPS_NEG_MonitorsE_Save": "",
    "CPS_NEG_MonitorsE_GivingUp": "",
    "CPS_MAINTAIN_Initiative_Suggestions": "",
    "CPS_MAINTAIN_Initiative_Compliments": "",
    "CPS_MAINTAIN_Initiative_Criticizes": "",
    "CPS_MAINTAIN_FulfillsR_Support": "",
    "CPS_MAINTAIN_FulfillsR_Apologizes": "",
    "CPS_MAINTAIN_FulfillsR_InitiatesOffTopic": "",
    "CPS_MAINTAIN_FulfillsR_JoinsOffTopic": ""
}


def ingest_cps(group_id: int):
    cps = collections.defaultdict(list)
    with open(file_path.CPS_CSV_PATH, "r") as in_f:
        csv_reader = csv.DictReader(in_f)
        for row in csv_reader:
            gid, uid = list(map(int, row['utteranceID'].split('_')[1:3]))
            
            if gid == group_id:
                # do not skip "phase2" utterances
                # if ingest_dialogue.PHASE1_START_UTTERANCE[gid] <= uid <= ingest_dialogue.PHASE1_END_UTTERANCE[gid]:
                for cps_label in cps_to_sa_mapping:
                    if row[cps_label] == '1':
                        cps[uid].append(cps_label)
    return cps


if __name__ == '__main__':
    utts = {}
    for group_id in range(1, 11):
        dialogue = ingest_dialogue.ingest_dialogue(group_id)
        for utt in dialogue.utterances:
            utts[f'{group_id}.{utt.id}'] = utt.text

    # dict to store CPS facet to list of sents
    cps_to_utts = collections.defaultdict(list)
    # count all utterances in the phase 1
    total_utts = 0
    # count #labels for each utterance
    utt_to_cpss = collections.defaultdict(list)

    with open(file_path.CPS_CSV_PATH, "r") as in_f:
        csv_reader = csv.DictReader(in_f)
        for row in csv_reader:
            gid, uid = list(map(int, row['utteranceID'].split('_')[1:3]))
            if uid > ingest_dialogue.PHASE1_END_UTTERANCE[gid]:
                continue
            total_utts += 1
            for cps_label in cps_to_sa_mapping:
                if row[cps_label] == '1':
                    utt_to_cpss[f'{gid}.{uid}'].append(cps_label)
                    cps_to_utts[cps_label].append((gid, uid, utts[f'{gid}.{uid}']))
    for k, vs in cps_to_utts.items():
        for v in vs:
            print(f"{k}: {v}")
    for k, v in cps_to_utts.items():
        print(f'{k}: {len(v)}')
    print('Total utts:', total_utts)

    print('# multi-labeled utterances')
    for k, v in utt_to_cpss.items():
        if len(v) > 1:
            print(f'{k} - "{utts[k]}": {v}')


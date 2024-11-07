import os
from typing import Union
import pandas as pd
from ingest.data import Utterance, Dialogue
from ingest.file_path import ORACLE_PATH, DP_ORACLE_PATH

BLOCK_NAME = {"red", "yellow", "green", "blue", "purple", "brown", "mystery"}
PRONOUNS = {
    "it",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
}
PHASE1_START_UTTERANCE = {
    1: 23,
    2: 11,
    3: 11,
    4: 12,
    5: 12,
    6: 9,
    7: 14,
    8: 0,  # this one is cut off at the beginning, and it seems we lost the whole task explanation and some of the very beginning of the collaborative dialogue lines. 
    9: 10,
    10: 10,
}
PHASE1_END_UTTERANCE = {
    # group number: utterance_id
    1: 101,
    2: 80,
    3: 88,
    4: 63,
    5: 82,
    6: 74,
    7: 144,
    8: 118,
    9: 41,
    10: 209,
}

BLOCK_NAME_PATTERN = rf"\b({'|'.join(BLOCK_NAME)})\b"
PRONOUN_PATTERN = rf"\b({'|'.join(PRONOUNS)})\b"


def read_transcript_csv(csv_file: Union[str, os.PathLike]):
    csv_df = pd.read_csv(csv_file, header=0)
    return csv_df


def ingest_dialogue(group_id: int) -> Dialogue:
    csv_file = ORACLE_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_Oracle.csv")
    csv_df = read_transcript_csv(csv_file)
    utterances = []
    for _, row in csv_df.iterrows():
        utt = Utterance(
            row.Utterance,
            row.Participant,
            row.Start,
            row.End,
            row.Transcript.strip(),
        )
        utterances.append(utt)
    dialogue = Dialogue(group_id, utterances)
    return dialogue


def ingest_dped_dialogue(group_id: int) -> Dialogue:
    csv_file = DP_ORACLE_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_dped_Oracle.csv")
    csv_df = read_transcript_csv(csv_file)
    utterances = []
    for _, row in csv_df.iterrows():
        utt = Utterance(
            row.Utterance,
            row.Participant,
            row.Start,
            row.End,
            row.Transcript.strip(),
        )
        utterances.append(utt)
    dialogue = Dialogue(group_id, utterances)
    return dialogue


if __name__ == "__main__":
    dialogue = ingest_dialogue(1)

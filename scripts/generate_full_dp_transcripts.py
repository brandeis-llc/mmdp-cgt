from ingest.file_path import DP_PATH, ORACLE_PATH, DATA_FOLDER
import pandas as pd


def read_dp_transcript_csv(group_id: int):
    csv_file = DP_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_DPed.csv")
    csv_df = pd.read_csv(csv_file, header=0)
    row_dict = dict()
    for _, row in csv_df.iterrows():
        row_dict[row.Utterance] = row.DPed
    return row_dict


def write_full_dp2file(group_id: int):
    dp_row_dict = read_dp_transcript_csv(group_id)
    csv_df = pd.read_csv(ORACLE_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_Oracle.csv"))
    rows = []
    for i, row in csv_df.iterrows():
        if row.Utterance in dp_row_dict:
            row.Transcript = dp_row_dict[row.Utterance]
        rows.append(row.values.tolist())
    pd.DataFrame(rows, columns=["Utterance", "Start", "End", "Group", "Participant", "Transcript"]).to_csv(
        DATA_FOLDER.joinpath("dped-oracle").joinpath(f"Group_{str(group_id).zfill(2)}_dped_Oracle.csv"), index=False)


if __name__ == '__main__':
    for i in range(1, 11):
        write_full_dp2file(i)

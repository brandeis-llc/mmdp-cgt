import copy
import csv
import re
from ingest.common_ground import CommonGround
from ingest.file_path import CGA_NEW_PATH, CGA_ORI_PATH

"""
Given a CGA csv file, return a dictionary of {timestamp: CommonGround object}.
Timestamps are end times of utterances that update the common ground (STATEMENT
and ACCEPT).
For example, if history has keys 0.0, 26.487, 82.08, etc.:
history[0.0] = common ground from time 0.0 to 26.487
history[26.487] = common ground from time 26.487 to 82.08
etc.
CommonGround objects can be queried with cg.poss, cg.evidence_for,
cg.evidence_against, etc. (see common_ground.py for details).
"""


def generate_cg_history(group_id: int, use_new: bool):
    cg = CommonGround()
    # store statement contents (accepts refer only to statement ids)
    statement_dict = {}
    history = {}
    # initialize history with initial common ground
    history[0.] = copy.deepcopy(cg)
    if use_new:
        csv_file = CGA_NEW_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_CGA.csv")
    else:
        csv_file = CGA_ORI_PATH.joinpath(f"Group_{str(group_id).zfill(2)}_CGA.csv")

    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            end_time = float(row['End Time - ss.msec'])
            cga = row['Common Ground']
            if 'STATEMENT' in row['Common Ground']:
                cg.cg_type = 'STATEMENT'
                statement_match = re.match(r'(S\d+).*STATEMENT.*\((.*?)\)', cga)
                if statement_match:
                    i = statement_match[1]
                    content = statement_match[2]
                    # update statement_dict, cg and history
                    statement_dict[i] = content
                    cg.update('STATEMENT', content)
                    history[end_time] = copy.deepcopy(cg)
            if 'ACCEPT' in row['Common Ground']:
                cg.cg_type = 'ACCEPT'
                accept_match = re.match(r'ACCEPT\((S\d+)\)', cga)
                if accept_match:
                    i = accept_match[1]
                    content = statement_dict[i]
                    # update cg and history
                    cg.update('ACCEPT', content)
                    history[end_time] = copy.deepcopy(cg)
    return history


if __name__ == '__main__':

    history = generate_cg_history(1, use_new=True)
    for k in history:
        print(k)
        cg = history[k]
        cg.print()
        print("=====================================")

import re

from ingest.file_path import CGA_OUTPUT_PATH
import json


def postprocess_gpt_4o(setting_id: str, group_id: int):
    with open(CGA_OUTPUT_PATH.joinpath(
            f"setting{setting_id}/mmdp_gpt_output_gpt-3.5-0125_segmented_group_{group_id}.json"), "r") as f:
        json_dict = json.load(f)
    for ts in json_dict:
        chat = json_dict[ts]["gpt_chat"]
        pred = chat[-1]["output"]["text"]
        # print(gold)
        # print(pred)
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, pred, re.DOTALL)
        if not matches:
            json_str = "{}"
        else:
            json_str = matches[0].strip()
        json_dict[ts]["gpt_chat"][-1]["output"]["text"] = json_str
        # print(json_str)
        # print("=====================================")
    with open(CGA_OUTPUT_PATH.joinpath(
            f"setting{setting_id}_fixed/mmdp_gpt_output_gpt-3.5-0125_segmented_group_{group_id}.json"), "w") as f:
        json.dump(json_dict, f, indent=4)


if __name__ == '__main__':
    for i in range(1, 11):
        postprocess_gpt_4o("vision5-4omini", i)

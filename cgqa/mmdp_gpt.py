import ast
import json
import re
from typing import List, Dict, Optional
from ingest.file_path import CGA_OUTPUT_PATH

import tqdm

from cgqa.prepare_cg_input import get_evidences_for_cg_statement, utterance_to_input

from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = ("The weight task is about 3 participants using a scale to determine the weight (in grams) of blocks "
                 "with different colors (red, blue, yellow, purple and green). They know at the beginning that "
                 "red block weights 10 grams\nYou are provided with the utterance, actions and gestures from the "
                 "participants during a time segment. "
                 "You task is to generate descriptive sentences that describe who said what and who did what. ")

SYSTEM_NO_DP_PROMPT = (
    "The weight task is about 3 participants using a scale to determine the weight (in grams) of blocks "
    "with different colors (red, blue, yellow, purple and green). They know at the beginning that "
    "red block weights 10 grams\nYou are provided with the utterance, actions and gestures from the "
    "participants during a time segment. ")

CG_PROMPT = ("Do they update or reach a conclusion on the weight of the blocks that have been discussed so far? "
             "If yes, answer in the JSON format {block color: weight, ...}.")


def prepare_cg_prompt(last_cg_response: Optional[str], model):
    if not last_cg_response:
        return CG_PROMPT
    if model == "gpt-4o":
        try:
            pattern = r'```json(.*?)```'
            matches = re.findall(pattern, last_cg_response, re.DOTALL)
            json_str = matches[0].strip()
            facts = json.loads(json_str)
        except:
            return CG_PROMPT
    else:
        try:
            facts = ast.literal_eval(last_cg_response)
            if not facts:
                return CG_PROMPT
        except:
            return CG_PROMPT
    facts = ", ".join([f"{k} block is {v}" for k, v in facts.items()])
    return f"They concluded that {facts}. After the discussion, {CG_PROMPT}"


def run_gpt(model_name: str, history_msg: List[Dict], curr_msg: Dict[str, str], include_curr: bool):
    history_msg.append(curr_msg)
    response = client.chat.completions.create(
        model=model_name,
        messages=history_msg,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response_txt = response.choices[0].message.content
    response = {"text": response_txt, "model": response.model, "role": response.choices[0].message.role}
    if include_curr:
        history_msg.append({"role": "assistant", "content": response_txt})
    else:
        history_msg[-1] = {"role": "assistant", "content": response_txt}
    return history_msg, response


def gpt_res2file(group_id: int, setting_id: str, text_only: bool, use_dp: bool, with_action: bool, with_gamr: bool,
                 with_text: bool, use_new: bool, cut_off: bool, model):
    history_msg = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]

    evidence, history = get_evidences_for_cg_statement(group_id, text_only=text_only, use_dp=use_dp, use_new=use_new)
    out_dict = dict()

    out_f = open(
        CGA_OUTPUT_PATH.joinpath(f"setting{setting_id}/mmdp_gpt_output_gpt-3.5-0125_segmented_group_{group_id}.json"),
        "w")
    prev_cg_response = None
    for e in evidence:
        out_dict[e] = dict()
        out_dict[e]["gold"] = list(history[e].fbank)  # for json serialization
        out_dict[e]["gpt_chat"] = []
        for u in tqdm.tqdm(evidence[e], "Processing utterances"):
            model_input = utterance_to_input(u, text_only=text_only, with_action=with_action, with_gamr=with_gamr,
                                             with_text=with_text)
            curr_message = {
                "role": "user",
                "content": "\n".join(m[0] for m in model_input)
            }
            history_msg, response = run_gpt(model, history_msg=history_msg, curr_msg=curr_message, include_curr=True)
            out_dict[e]["gpt_chat"].append({"input": curr_message, "output": response})

        curr_message = {
            "role": "user",
            "content": prepare_cg_prompt(prev_cg_response, model)
        }
        history_msg, response = run_gpt(model, history_msg=history_msg, curr_msg=curr_message, include_curr=True)
        if cut_off:
            history_msg = history_msg[:1]
        prev_cg_response = response["text"]
        out_dict[e]["gpt_chat"].append({"input": curr_message, "output": response})
    out_f.write(json.dumps(out_dict, indent=4) + "\n")
    out_f.close()


def gpt_no_dp_res2file(group_id: int, setting_id: str, text_only: bool, use_dp: bool, with_action: bool,
                       with_gamr: bool,
                       with_text: bool, use_new: bool, cut_off: bool, model):
    history_msg = [{
        "role": "system",
        "content": SYSTEM_NO_DP_PROMPT
    }]

    evidence, history = get_evidences_for_cg_statement(group_id, text_only=text_only, use_dp=use_dp, use_new=use_new)
    out_dict = dict()

    out_f = open(
        CGA_OUTPUT_PATH.joinpath(f"setting{setting_id}/mmdp_gpt_output_gpt-3.5-0125_segmented_group_{group_id}.json"),
        "w")
    prev_cg_response = None
    for e in evidence:
        out_dict[e] = dict()
        out_dict[e]["gold"] = list(history[e].fbank)  # for json serialization
        out_dict[e]["gpt_chat"] = []
        curr_utterances = []
        for u in tqdm.tqdm(evidence[e], "Processing utterances"):
            model_input = utterance_to_input(u, text_only=text_only, with_action=with_action, with_gamr=with_gamr,
                                             with_text=with_text)
            curr_utterances.append("\n".join(m[0] for m in model_input))

        curr_utterances.append(prepare_cg_prompt(prev_cg_response, model))
        # print("\n\n".join(curr_utterances))

        curr_message = {
            "role": "user",
            "content": "\n\n".join(curr_utterances)
        }
        # print(history_msg)
        # print(curr_message)
        # print("=====================================")
        history_msg, response = run_gpt(model, history_msg=history_msg, curr_msg=curr_message, include_curr=True)
        if cut_off:
            history_msg = history_msg[:1]

        prev_cg_response = response["text"]
        out_dict[e]["gpt_chat"].append({"input": curr_message, "output": response})
    out_f.write(json.dumps(out_dict, indent=4) + "\n")
    out_f.close()


def gpt_dp_only_res2file(group_id: int, setting_id: str, text_only: bool, use_dp: bool, with_action: bool,
                         with_gamr: bool,
                         with_text: bool, use_new: bool, model):
    history_msg = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]

    evidence, history = get_evidences_for_cg_statement(group_id, text_only=text_only, use_dp=use_dp, use_new=use_new)
    out_dict = dict()

    out_f = open(
        CGA_OUTPUT_PATH.joinpath(f"setting{setting_id}/mmdp_gpt_output_gpt-3.5-0125_segmented_group_{group_id}.json"),
        "w")
    prev_cg_response = None
    for e in evidence:
        out_dict[e] = dict()
        out_dict[e]["gold"] = list(history[e].fbank)  # for json serialization
        out_dict[e]["gpt_chat"] = []
        for u in tqdm.tqdm(evidence[e], "Processing utterances"):
            model_input = utterance_to_input(u, text_only=text_only, with_action=with_action, with_gamr=with_gamr,
                                             with_text=with_text)
            curr_message = {
                "role": "user",
                "content": "\n".join(m[0] for m in model_input)
            }
            history_msg, response = run_gpt(model, history_msg=history_msg, curr_msg=curr_message, include_curr=False)
            out_dict[e]["gpt_chat"].append({"input": curr_message, "output": response})

        curr_message = {
            "role": "user",
            "content": prepare_cg_prompt(prev_cg_response, model)
        }
        history_msg, response = run_gpt(model, history_msg=history_msg, curr_msg=curr_message, include_curr=True)
        prev_cg_response = response["text"]
        out_dict[e]["gpt_chat"].append({"input": curr_message, "output": response})
    out_f.write(json.dumps(out_dict, indent=4) + "\n")
    out_f.close()


if __name__ == '__main__':
    print(SYSTEM_PROMPT)
    input("This script will generate GPT responses for the CGQA task. Press Enter to continue.")
    setting_id = "18-3"  # folder name
    text_only = True  # only use text input (overwrite with_action, with_gamr)
    with_action = False  # include action input
    with_gamr = False  # include gamr input
    with_text = True  # include text input
    use_dp = False  # use DP text
    use_new = False  # use new CGA data
    cut_off = True  # cut off the conversation context
    model = "gpt-4o"  # or gpt-3.5-turbo-0125

    for group_id in range(1, 11):
        # gpt_res2file(group_id, setting_id=setting_id, text_only=text_only, use_dp=use_dp, with_action=with_action,
        #              with_gamr=with_gamr, with_text=with_text, use_new=use_new, cut_off=cut_off, model=model)

        gpt_no_dp_res2file(group_id, setting_id=setting_id, text_only=text_only, use_dp=use_dp, with_action=with_action,
                           with_gamr=with_gamr, with_text=with_text, use_new=use_new, cut_off=cut_off, model=model)

        # gpt_dp_only_res2file(group_id, setting_id=setting_id, text_only=text_only, use_dp=use_dp, with_action=with_action,
        #                    with_gamr=with_gamr, with_text=with_text, use_new=use_new, model=model)

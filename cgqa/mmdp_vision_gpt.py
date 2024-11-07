import ast
import json
import re
import base64
from typing import List, Dict, Optional
from ingest.file_path import CGA_OUTPUT_PATH, FRAME_PATH

import tqdm

from cgqa.prepare_cg_input import get_evidences_for_cg_statement, utterance_to_input

from openai import OpenAI

client = OpenAI()

SYSTEM_NO_DP_VISION_PROMPT = (
    "The weight task is about 3 participants (numbered 1, 2, 3 from left to right) using a scale to determine the weight (in grams) of blocks "
    "with different colors (red, blue, yellow, purple and green). They know at the beginning that "
    "red block weights 10 grams\nYou are provided with the utterances during a time segment, and five corresponding video frames to each utterance.")

CG_PROMPT = ("Do they update or reach a conclusion on the weight of the blocks that have been discussed so far? "
             "If yes, answer in the JSON format {block color: weight, ...}.")


def prepare_cg_prompt(last_cg_response: Optional[str], model):
    if not last_cg_response:
        return CG_PROMPT
    if model == "gpt-4o" or model == "gpt-4o-mini":
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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def format_image_dict(image_path):
    base64_image = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "low",
        },
    }


def format_gpt_vision_input(image_path_lst):
    text_indicator = {
        "type": "text",
        "text": "Image frames corresponding to the utterance above:\n",
    }
    message_dict = {
        "role": "user",
        "content": [format_image_dict(image_path) for image_path in image_path_lst],
    }
    message_dict["content"].insert(0, text_indicator)
    return message_dict


def run_gpt(model_name: str, history_msg: List[Dict], curr_msgs: List[Dict[str, str]], include_curr: bool):
    history_msg.extend(curr_msgs)
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
        history_msg = [history_msg[0], {"role": "assistant", "content": response_txt}]
    return history_msg, response


def gpt_no_dp_res2file(group_id: int, setting_id: str, text_only: bool, use_dp: bool, with_action: bool,
                       with_gamr: bool,
                       with_text: bool, use_new: bool, cut_off: bool, model):
    history_msg = [{
        "role": "system",
        "content": SYSTEM_NO_DP_VISION_PROMPT
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
        print(f"Gold: {out_dict[e]['gold']}")
        out_dict[e]["gpt_chat"] = []
        curr_utterances = []
        curr_evi_messages = []
        for u in tqdm.tqdm(evidence[e], "Processing utterances"):
            model_input = utterance_to_input(u, text_only=text_only, with_action=with_action, with_gamr=with_gamr,
                                             with_text=with_text)
            curr_utterances.append("\n".join(m[0] for m in model_input))
            curr_utterance = "\n".join(m[0] for m in model_input)
            curr_text_message = {
                "role": "user",
                "content": curr_utterance + "\n"}
            curr_evi_messages.append(curr_text_message)

            frame_path_lst = [(FRAME_PATH / str(group_id) / f"{e}-{u.id}-{i}.jpg") for i in [0, 2, 4, 6, 8]]
            # print(frame_path_lst)
            curr_frame_message = format_gpt_vision_input(frame_path_lst)
            curr_evi_messages.append(curr_frame_message)

        cg_message = {
            "role": "user",
            "content": prepare_cg_prompt(prev_cg_response, model)
        }
        print(cg_message)
        curr_evi_messages.append(cg_message)

        # pretty_evi_messages = []
        # for message in curr_evi_messages:
        #     content = message["content"]
        #     if isinstance(content, str):
        #         pretty_evi_messages.append(message)
        #     else:
        #         for i, item in enumerate(content):
        #             if item.get("type") == "image_url":
        #                 content[i]["image_url"]["url"] = "image_url"
        #         message["content"] = content
        #         pretty_evi_messages.append(message)

        # pprint.pprint(pretty_evi_messages)
        # return

        # curr_utterances.append(prepare_cg_prompt(prev_cg_response, model))
        # print("\n\n".join(curr_utterances))

        # curr_message = {
        #     "role": "user",
        #     "content": "\n\n".join(curr_utterances)
        # }
        # print(history_msg)
        # print(curr_message)
        # print("=====================================")
        history_msg, response = run_gpt(model, history_msg=history_msg, curr_msgs=curr_evi_messages, include_curr=True)
        if cut_off:
            history_msg = history_msg[:1]

        prev_cg_response = response["text"]
        print(f"Predict: {prev_cg_response}")

        pretty_evi_messages = []
        for message in curr_evi_messages:
            content = message["content"]
            if isinstance(content, str):
                pretty_evi_messages.append(message)
            else:
                for i, item in enumerate(content):
                    if item.get("type") == "image_url":
                        content[i]["image_url"]["url"] = "image_url"
                message["content"] = content
                pretty_evi_messages.append(message)

        # pprint.pprint(pretty_evi_messages)

        out_dict[e]["gpt_chat"].append({"input": pretty_evi_messages, "output": response})
    out_f.write(json.dumps(out_dict, indent=4) + "\n")
    out_f.close()


if __name__ == '__main__':
    print(SYSTEM_NO_DP_VISION_PROMPT)
    input("This script will generate GPT responses for the CGQA task. Press Enter to continue.")
    setting_id = "vision5-4omini"  # folder name
    text_only = True  # only use text input (overwrite with_action, with_gamr)
    with_action = False  # include action input
    with_gamr = False  # include gamr input
    with_text = True  # include text input
    use_dp = False  # use DP text
    use_new = False  # use new CGA data
    cut_off = True  # cut off the conversation context
    model = "gpt-4o-mini"  # or gpt-4o

    for group_id in range(1, 11):
        print(f"Processing group {group_id}")
        gpt_no_dp_res2file(group_id, setting_id=setting_id, text_only=text_only, use_dp=use_dp, with_action=with_action,
                           with_gamr=with_gamr, with_text=with_text, use_new=use_new, cut_off=cut_off, model=model)

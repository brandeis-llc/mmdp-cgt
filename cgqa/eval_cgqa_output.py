import re
from statistics import mean
from typing import Dict, Set

from ingest.file_path import CGA_OUTPUT_PATH
import json

GOLD = {
    "red": "10",
    "blue": "10",
    "green": "20",
    "purple": "30",
    "yellow": "50"
}

GOLD_STR = {f"{k}={v}": v for k, v in GOLD.items()}


def load_cga_json(setting_id: str, group_id: int):
    with open(CGA_OUTPUT_PATH.joinpath(
            f"setting{setting_id}/mmdp_gpt_output_gpt-3.5-0125_segmented_group_{group_id}.json"), "r") as f:
        json_dict = json.load(f)
    golds = []
    preds = []
    for ts in json_dict:
        gold = json_dict[ts]["gold"]
        gold = {c.split("=")[0]: c.split("=")[1] for c in gold}
        chat = json_dict[ts]["gpt_chat"]
        pred = chat[-1]["output"]["text"]
        try:
            pred = json.loads(chat[-1]["output"]["text"])
            pred = {re.split(r"_|\s|block", k)[0].lower(): str(v) for k, v in pred.items()}
        except:
            pass
        golds.append(gold)
        preds.append(pred)
    return golds, preds


def dsc(golds: Set[str], preds: Set[str]):
    return 2 * len(golds & preds) / (len(golds) + len(preds))


def group_dsc(setting_id, group_id, strict: bool):
    all_golds = set()
    all_preds = set()
    all_dsc = []
    golds, preds = load_cga_json(setting_id, group_id)

    for i, (g, p) in enumerate(zip(golds, preds)):
        print(p)
        gs = {f"{i}-{k}={v}" for k, v in g.items()}
        ps = {f"{i}-{k}={v}" for k, v in p.items()}
        # print(g)
        # print(p)

        if strict:
            all_golds |= gs
            all_preds |= ps
        else:
            filtered_ps = set()
            for p in ps:
                if p not in gs:
                    part = p.split("-")[1]
                    if part in GOLD_STR:
                        continue
                filtered_ps.add(p)

            all_golds |= gs
            all_preds |= filtered_ps

        if not all_golds and not all_preds:
            continue
        dsc_score = dsc(all_golds, all_preds)
        # print(dsc_score)
        all_dsc.append(dsc_score)
    # print(mean(all_dsc))
    return mean(all_dsc), all_dsc


def prf1(golds: Dict[str, str], preds: Dict[str, str]):
    tp = 0
    fp = 0
    fn = 0
    for g in golds:
        pred = preds.get(g)
        if golds[g] == pred:
            tp += 1
        elif pred is None:
            fn += 1
        else:
            fp += 1
    for p in preds:
        if p not in golds and GOLD.get(p) != preds[p]:
            fp += 1

    if not tp:
        if not fp and not fn:
            return 1, 1, 1
        return 0, 0, 0

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def group_accumulative_prf1(setting_id, group_id):
    all_golds = dict()
    all_preds = dict()
    all_precision = []
    all_recall = []
    all_f1 = []
    golds, preds = load_cga_json(setting_id, group_id)

    for g, p in zip(golds, preds):
        # print(g)
        # print(p)
        all_golds = {**all_golds, **g}
        all_preds = {**all_preds, **p}
        if not all_golds and not all_preds:
            continue
        precision, recall, f1 = prf1(all_golds, all_preds)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        # print(precision, recall, f1)
    # print(mean(all_precision), mean(all_recall), mean(all_f1))
    return mean(all_f1), all_f1


if __name__ == '__main__':
    setting_id = "vision5-4omini_fixed"  # folder name for the model output
    all_group_dsc = []
    all_group_f1 = []
    dsc_strict = True

    for group_id in range(1, 11):
        dscs = group_dsc(setting_id, group_id, strict=dsc_strict)
        print(dscs)
        all_group_dsc.append(dscs[0])
        f1s = group_accumulative_prf1(setting_id, group_id)
        all_group_f1.append(f1s[0])
        print(f1s)
        print("====================================")
        golds, preds = load_cga_json(setting_id, group_id)
        # for g, p in zip(golds, preds):
        #     print(g)
        #     print(p)
        #     print("====================================")
        # break
    print("\t".join([str(round(s, 3)) for s in all_group_dsc]))
    print("\t".join([str(round(s, 3)) for s in all_group_f1]))
    print(" & ".join([str(round(s * 100, 1)) for s in all_group_dsc]))
    print(sum(all_group_dsc) / len(all_group_dsc))

import os
import json
import pandas as pd

DIR_LABELS = "/home/user/jverhoek/sct-ood-dataset/labels"
EXCEL_OVERVIEW_PELVIS = "/local/scratch/jverhoek/datasets/Task1/pelvis/overview/1_pelvis_train.xlsx"


def parse_index(s):
    if s.strip().lower() == "na":
        return None
    elif s.strip().isdigit():
        return int(s.strip())
    else:
        raise ValueError(f"Invalid index value: {s}")


def process_line(line, abnormal_ids, include_type=False):
    try:
        data = line.strip().split(",")

        if len(data) < 6:
            return None  # skip incomplete/normal cases

        if data[5] in abnormal_ids:
            id_ = data[0]
            mr_start = parse_index(data[1])
            mr_end = parse_index(data[2])
            ct_start = parse_index(data[3])
            ct_end = parse_index(data[4])

            if id_.lower().startswith("1b"):
                body_part = "brain"
            elif id_.lower().startswith("1p"):
                body_part = "pelvis"
            else:
                raise ValueError(f"Unknown body part for id: {id_}")

            entry = {
                id_: {
                    "mr_start": mr_start,
                    "mr_end": mr_end,
                    "ct_start": ct_start,
                    "ct_end": ct_end,
                    "body_part": body_part,
                }
            }

            if include_type:
                entry[id_]["type"] = data[5]

            return entry
    except Exception as e:
        print(f"Error processing line: {e}: {line}")


def process_labels(dir_labels=DIR_LABELS, abnormal_ids=["1"], include_type=False):
    labels = []
    for filename in os.listdir(dir_labels):
        if filename.endswith(".txt") and filename.startswith("ood"):
            file_path = os.path.join(dir_labels, filename)
            with open(file_path, 'r') as file:
                processed_labels = filter(
                    None, [process_line(line, abnormal_ids, include_type) for line in file]
                )
                labels.extend(processed_labels)
    return labels


def main():
    # Type 1 labels
    labels_1 = process_labels(abnormal_ids=["1"], include_type=False)
    labels_type1 = {"type1": labels_1}

    with open(os.path.join(DIR_LABELS, "labels_implant.json"), "w") as f:
        json.dump(labels_type1, f, indent=2)

    print(f"Number of type 1 labels: {len(labels_1)}")
    print("Sample labels (type 1):", labels_1[:3])

    # Types 2–7 labels
    labels_other = process_labels(abnormal_ids=[str(i) for i in range(2, 8)], include_type=True)
    labels_other_json = {"types_2_to_7": labels_other}

    with open(os.path.join(DIR_LABELS, "labels_others.json"), "w") as f:
        json.dump(labels_other_json, f, indent=2)

    print(f"Number of type 2–7 labels: {len(labels_other)}")
    print("Sample labels (2–7):", labels_other[:3])

    # Load Excel sheets just for info
    df_overview_pelvis_ct = pd.read_excel(EXCEL_OVERVIEW_PELVIS, sheet_name="CT")
    df_overview_pelvis_mr = pd.read_excel(EXCEL_OVERVIEW_PELVIS, sheet_name="MR")

    print("CT overview shape:", df_overview_pelvis_ct.shape)
    print("MR overview shape:", df_overview_pelvis_mr.shape)


if __name__ == "__main__":
    main()

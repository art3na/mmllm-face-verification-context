from typing import Dict, List, Tuple, Optional
from glob import glob
import os
import numpy as np
import pandas as pd


def _parse_filename(fn: str) -> Tuple[str, str, int]:
    dataset_errors = {
        'LeslieCaron': 'f', 'MarisaPavan': 'f', 'FrancesDee': 'f',
        'JoanLorring': 'f', 'RoyScheider': 'm', 'GladysCooper': 'f',
        'TomJones': 'm'
    }

    fn = fn.split('/')[-1]
    age = int(fn.split('_')[-2])
    name = fn.split('_')[-3]
    if name in dataset_errors:
        sex = dataset_errors[name]
    else:
        sex = fn.split('_')[-1].split('.')[0]

    return (name, sex, age)


"""
Args:
    root_img_dir (string): root directory to AgeDB dataset
    age_gap (int): age gap between pair images in probe and gallery lists
    min_probe_age (int): minimal age for person in probe list,
                         gallery list minimal age will be min_probe_age+age_gap accordingly
    max_probe_age (int): maximal age for person in probe list,
                         gallery list maximal age will be max_probe_age+age_gap accordingly
    sex_invariant (boolean): to use or not male-female pairs
"""
def make_pairs(
    root_img_dir: str,
    age_gap: int,
    min_probe_age: int = 1,
    max_probe_age: int = 101,
    sex_invariant: bool = False
) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    males_by_age: Dict[int, List[Tuple[str, int]]] = {}
    females_by_age: Dict[int, List[Tuple[str, int]]] = {}

    img_fns = glob(os.path.join(root_img_dir, '*.jpg'))
    names: List[str] = []

    for img_fn in img_fns:
        img_base_fn = os.path.basename(img_fn)
        name, sex, age = _parse_filename(img_fn)

        if name in names:
            label = names.index(name)
        else:
            label = len(names)
            names.append(name)

        if sex == 'm':
            males_by_age.setdefault(age, []).append((img_base_fn, label))
        else:
            females_by_age.setdefault(age, []).append((img_base_fn, label))

    # same-sex and (optionally) cross-sex pairs
    for age in males_by_age.keys():
        if age < min_probe_age or age > max_probe_age:
            continue

        if (age + age_gap) in males_by_age:
            for probe_male in males_by_age[age]:
                for gallery_male in males_by_age[age + age_gap]:
                    pairs.append((probe_male, gallery_male))

        if sex_invariant and (age + age_gap) in females_by_age:
            for probe_male in males_by_age[age]:
                for gallery_female in females_by_age[age + age_gap]:
                    pairs.append((probe_male, gallery_female))

    for age in females_by_age.keys():
        if age < min_probe_age or age > max_probe_age:
            continue

        if (age + age_gap) in females_by_age:
            for probe_female in females_by_age[age]:
                for gallery_female in females_by_age[age + age_gap]:
                    pairs.append((probe_female, gallery_female))

        if sex_invariant and (age + age_gap) in males_by_age:
            for probe_female in females_by_age[age]:
                for gallery_male in males_by_age[age + age_gap]:
                    pairs.append((probe_female, gallery_male))

    return pairs


def pairs_to_dataframe(
    pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]]
) -> pd.DataFrame:
    """
    Convert list of pairs into a pandas DataFrame.

    Columns:
        probe_fn, probe_label, gallery_fn, gallery_label, genuine
    """
    rows = []
    for (probe_fn, probe_label), (gallery_fn, gallery_label) in pairs:
        rows.append({
            "probe_fn": probe_fn,
            "probe_label": probe_label,
            "gallery_fn": gallery_fn,
            "gallery_label": gallery_label,
            "genuine": int(probe_label == gallery_label),  # 1 = same identity, 0 = impostor
        })
    return pd.DataFrame(rows)


def create_pairs_csv(
    pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
    save_dir: str,
    genuine_cnt: Optional[int] = None,
    impostor_cnt: Optional[int] = None,
    prefix: str = "AgeDB",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create CSV files that you can easily load with pandas.

    Returns:
        (df_all, df_genuine, df_impostor)
    """
    os.makedirs(save_dir, exist_ok=True)

    # full dataframe
    df_all = pairs_to_dataframe(pairs)

    # split genuine / impostor
    df_genuine = df_all[df_all["genuine"] == 1].copy()
    df_impostor = df_all[df_all["genuine"] == 0].copy()

    # subsample if requested
    if genuine_cnt is not None and genuine_cnt < len(df_genuine):
        df_genuine = df_genuine.sample(
            n=genuine_cnt, replace=False, random_state=random_state
        )

    if impostor_cnt is not None and impostor_cnt < len(df_impostor):
        df_impostor = df_impostor.sample(
            n=impostor_cnt, replace=False, random_state=random_state
        )

    # infer age_gap from filenames (same logic as before)
    probe_age = int(df_all.iloc[0]["probe_fn"].split("_")[-2])
    gallery_age = int(df_all.iloc[0]["gallery_fn"].split("_")[-2])
    age_gap = gallery_age - probe_age

    # file names
    all_fn = os.path.join(save_dir, f"{prefix}_{age_gap}_pairs.csv")
    genuine_fn = os.path.join(save_dir, f"{prefix}_{age_gap}_genuine.csv")
    impostor_fn = os.path.join(save_dir, f"{prefix}_{age_gap}_impostor.csv")

    # save as CSV
    df_all.to_csv(all_fn, index=False)
    df_genuine.to_csv(genuine_fn, index=False)
    df_impostor.to_csv(impostor_fn, index=False)

    return df_all, df_genuine, df_impostor


# (optional) keep your old txt writer if you still need the original format
def create_pairs_txt(
    pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
    save_dir: str,
    genuine_cnt: Optional[int] = None,
    impostor_cnt: Optional[int] = None
) -> None:
    genuine_pairs = []
    impostor_pairs = []
    for pair in pairs:
        if pair[0][1] == pair[1][1]:
            genuine_pairs.append(pair)
        else:
            impostor_pairs.append(pair)

    if genuine_cnt is None:
        genuine_cnt = len(genuine_pairs)
    if impostor_cnt is None:
        impostor_cnt = len(impostor_pairs)

    age_gap = int(pairs[0][1][0].split('_')[-2]) - int(pairs[0][0][0].split('_')[-2])
    matched_fn = os.path.join(save_dir, f'AgeDB_{age_gap}_gen.txt')
    mismatched_fn = os.path.join(save_dir, f'AgeDB_{age_gap}_imp.txt')

    with open(matched_fn, 'w') as matched, open(mismatched_fn, 'w') as mismatched:
        genuine_pairs_id = np.random.choice(len(genuine_pairs), genuine_cnt, replace=False)
        for genuine_pair_id in genuine_pairs_id:
            cur_pair = genuine_pairs[genuine_pair_id]
            matched.write(f"{cur_pair[0][0][:-4]}\t{cur_pair[1][0]}\n")

        impostor_pairs_id = np.random.choice(len(impostor_pairs), impostor_cnt, replace=False)
        for impostor_pair_id in impostor_pairs_id:
            cur_pair = impostor_pairs[impostor_pair_id]
            mismatched.write(f"{cur_pair[0][0][:-4]}\t{cur_pair[1][0]}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_img_dir', default='..', help='Root directory containing AgeDB images')
    parser.add_argument('--age_gap', type=int, default=30, help='Age gap')
    parser.add_argument('--genuine_cnt', type=int, default=3000, help='Genuine pair count')
    parser.add_argument('--impostor_cnt', type=int, default=3000, help='Impostor pair count')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    args = parser.parse_args()

    # generate pairs
    pairs = make_pairs(args.root_img_dir, args.age_gap)

    # NEW: create CSVs + work directly with pandas
    df_all, df_genuine, df_impostor = create_pairs_csv(
        pairs,
        save_dir=args.output_dir,
        genuine_cnt=args.genuine_cnt,
        impostor_cnt=args.impostor_cnt
    )


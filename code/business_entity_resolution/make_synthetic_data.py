"""Generates small synthetic train/test datasets matching the challenge
schema, purely to smoke-test that the pipeline code runs end-to-end without
errors. This is NOT meant to validate model quality -- swap in the real
challenge dataset for that.
"""
import random
import csv
import os

random.seed(0)

BASE_NAMES = [
    "Sharma Traders", "Bombay Textiles", "Green Leaf Cafe", "Om Sai Enterprises",
    "Kumar & Sons", "Nashik Auto Parts", "Sunrise Bakery", "Patel Hardware",
    "Delta Logistics Corp", "Riverside Motors Pvt Ltd", "Golden Gate Consulting Inc",
    "Metro Fresh Mart", "Silverline Electronics", "Everest Traders Company",
    "Blue Ocean Foods Limited",
]
CITIES_US = ["123 Main St, Springfield", "45 Market Rd, Austin", "9 Elm Ave, Denver"]
CITIES_IN = ["12 MG Road, Pune", "45 Gandhi Nagar, Nashik", "7 Sector 5, Delhi"]
CITIES_FR = ["10 Rue de Paris, Lyon", "5 Avenue Foch, Nice"]


def noisy_name(name):
    variants = [
        name,
        name.replace("Pvt Ltd", "Private Limited").replace("Inc", "Incorporated").replace("Corp", "Corporation"),
        name.replace("&", "and"),
        name.upper(),
        name + " ",
        " ".join(reversed(name.split())),
    ]
    return random.choice(variants)


def noisy_addr(addr):
    variants = [addr, addr.replace("St,", "Street,").replace("Rd,", "Road,").replace("Ave,", "Avenue,"),
                "Near Landmark, " + addr, addr.replace(",", "")]
    return random.choice(variants)


def build(n_entities, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
    for i in range(n_entities):
        eid1 = f"S1-{i:05d}"
        name = random.choice(BASE_NAMES) + f" {i}"
        country = random.choice(["US", "India"] if prefix == "train" else ["US", "India", "France"])
        addr = random.choice(CITIES_US if country == "US" else (CITIES_IN if country == "India" else CITIES_FR))
        s1_rows.append([eid1, name, addr, country])

        matched = []
        # 60% chance of having 1-2 matches spread across S2/S3, else singleton
        if random.random() < 0.6:
            n_match = random.choice([1, 1, 2])
            for _ in range(n_match):
                src = random.choice(["S2", "S3"])
                eid2 = f"{src}-{len(s2_rows) if src=='S2' else len(s3_rows):05d}"
                row = [eid2, noisy_name(name), noisy_addr(addr), country]
                (s2_rows if src == "S2" else s3_rows).append(row)
                matched.append(eid2)
        gt_rows.append([eid1, ",".join(matched)])

        # sprinkle some unrelated noise records too
        if random.random() < 0.3:
            noise_name = random.choice(BASE_NAMES) + f" noise{i}"
            noise_country = random.choice(["US", "India", "France"])
            noise_addr = random.choice(CITIES_US + CITIES_IN + CITIES_FR)
            src = random.choice(["S2", "S3"])
            eid_noise = f"{src}-{len(s2_rows) if src=='S2' else len(s3_rows):05d}"
            (s2_rows if src == "S2" else s3_rows).append([eid_noise, noise_name, noise_addr, noise_country])

    def write_tsv(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)

    header = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(f"{out_dir}/{prefix}_source1.tsv", header, s1_rows)
    write_tsv(f"{out_dir}/{prefix}_source2.tsv", header, s2_rows)
    write_tsv(f"{out_dir}/{prefix}_source3.tsv", header, s3_rows)
    if prefix == "train":
        write_tsv(f"{out_dir}/{prefix}_ground_truth.tsv",
                  ["source1_entity_id", "matched_entity_ids"], gt_rows)


if __name__ == "__main__":
    build(200, "dataset/train", "train")
    build(60, "dataset/test", "test")
    print("Synthetic data written to dataset/train and dataset/test")

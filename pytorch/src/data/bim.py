
from typing import Dict, List


def bim_reader(bim_file: str) -> List[Dict]:
    bim_list = []
    with open(bim_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts: list = line.split()
            bim_list.append({
                "chrom": parts[0],
                "snp_id": parts[1],
                "cM": parts[2],
                "position": parts[3],
                "allele1": parts[4],
                "allele2": parts[5]
            })
    return bim_list

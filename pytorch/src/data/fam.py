
from typing import Literal, overload


@overload
def fam_reader(
    fam_path: str, return_feature: Literal[True]) -> tuple[list, list]: ...


@overload
def fam_reader(fam_path: str, return_feature: Literal[False]) -> list: ...


def fam_reader(fam_path: str, return_feature: bool) -> list | tuple[list, list]:
    fam_list_id: list = []
    fam_list_feature: list = []
    with open(fam_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts: list = line.split()
            fid, iid, father, mother, sex, *_ = parts
            fam_list_id.append((fid, iid))
            if return_feature:
                fam_list_feature.append((father, mother, sex))
    if return_feature:
        return fam_list_id, fam_list_feature
    return fam_list_id

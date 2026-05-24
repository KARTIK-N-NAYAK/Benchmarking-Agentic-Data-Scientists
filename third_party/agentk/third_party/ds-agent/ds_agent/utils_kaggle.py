from __future__ import annotations

from functools import total_ordering
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from ds_agent.utils import ListableEnum


class MedalName(ListableEnum):
    Gold = "Gold"
    Silver = "Silver"
    Bronze = "Bronze"
    NoMedal = "No Medal"


@total_ordering
class Medal(BaseModel):
    name: ClassVar[MedalName]
    color: ClassVar[str]
    value: ClassVar[int]

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @staticmethod
    def get_medal_from_str(medal_str: str) -> type[Medal]:
        return KAG_MEDALS_NAME_TO_CLASS[MedalName.get_enum_element(medal_str)]


class GoldMedal(Medal):
    name: ClassVar[MedalName] = MedalName.Gold
    color: ClassVar[str] = "#FFD700"
    value: ClassVar[int] = 3

    def __str__(self) -> str:
        return "🥇"


class SilverMedal(Medal):
    name: ClassVar[MedalName] = MedalName.Silver
    color: ClassVar[str] = "#c0c0c0"
    value: ClassVar[int] = 2

    def __str__(self) -> str:
        return "🥈"


class BronzeMedal(Medal):
    name: ClassVar[MedalName] = MedalName.Bronze
    color: ClassVar[str] = "#bf7c0f"
    value: ClassVar[int] = 1

    def __str__(self) -> str:
        return "🥉"


class NoMedal(Medal):
    name: ClassVar[MedalName] = MedalName.NoMedal
    color: ClassVar[str] = "#59c1eb"
    value: ClassVar[int] = 0


KAGGLE_MEDALS: tuple[type[Medal], ...] = (GoldMedal, SilverMedal, BronzeMedal, NoMedal)
KAG_MEDALS_NAME_TO_CLASS: dict[MedalName, type[Medal]] = {medal.name: medal for medal in KAGGLE_MEDALS}


def get_medal(rank: int, n_entries: int) -> type[Medal]:
    if n_entries < 250:
        if n_entries < 100 and rank <= n_entries * 0.1:
            return GoldMedal
        elif n_entries > 100 and rank <= 10:
            return GoldMedal
        if rank <= n_entries * 0.2:
            return SilverMedal
        if rank <= n_entries * 0.4:
            return BronzeMedal
        else:
            return NoMedal

    if rank <= (10 + 0.2 * n_entries / 100):
        return GoldMedal

    if n_entries < 1000:
        if rank <= 50:
            return SilverMedal
        if rank <= 100:
            return BronzeMedal
        else:
            return NoMedal

    if rank <= n_entries * 0.05:
        return SilverMedal
    if rank <= n_entries * 0.1:
        return BronzeMedal
    return NoMedal


def get_rank_to_get_medal(n_entries: int, medal: type[Medal]) -> int:
    """ Return the last rank awarding a medal if there are `n_entries` participants """
    if medal == GoldMedal:
        if n_entries < 100:
            return int(n_entries / 10)  # top 10%
        if n_entries < 250:
            return 10
        return 10 + int(n_entries * 0.2 / 100)
    elif medal == SilverMedal:
        if n_entries < 250:
            return int(n_entries * 20 / 100)  # top 20%
        if n_entries < 1000:
            return 50
        return int(n_entries * 5 / 100)  # top 5%
    elif medal == BronzeMedal:
        if n_entries < 250:
            return int(n_entries * 40 / 100)  # top 40 %
        if n_entries < 1000:
            return 100
        return int(n_entries * 10 / 100)  # top 10%
    else:
        raise ValueError(medal)


class KaggleLevelNames(ListableEnum):
    Unkown = "Unknown"
    Novice = "Novice"
    Contributor = "Contributor"
    BaseKaggler = "Base"
    Expert = "Expert"
    Master = "Master"
    Grandmaster = "Grandmaster"


class KaggleLevel:
    name: KaggleLevelNames
    color: str
    value: int

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @staticmethod
    def get_kaggle_level_from_str(kaggle_level: str) -> type[KaggleLevel]:
        return KAG_LVL_NAME_TO_KAG_LVL_TYPE[KaggleLevelNames.get_enum_element(value=kaggle_level)]


class UnknownLevel(KaggleLevel):
    name = KaggleLevelNames.Unkown
    value = -1
    color = "grey"


class Novice(KaggleLevel): # TODO: Deprecate
    name = KaggleLevelNames.Novice
    color = "#b3cde0"  # Light Blue
    value = 0


class BaseKaggler(KaggleLevel):
    name = KaggleLevelNames.BaseKaggler
    color = "#6497b1"  # Medium Blue
    value = 1


class Contributor(KaggleLevel):  # TODO: Deprecate
    name = KaggleLevelNames.Contributor
    color = "#6497b1"  # Medium Blue
    value = 1


class Expert(KaggleLevel):
    name = KaggleLevelNames.Expert
    color = "#005b96"  # Deep Blue
    value = 2


class Master(KaggleLevel):
    name = KaggleLevelNames.Master
    color = "#03396c"  # Dark Blue
    value = 3


class Grandmaster(KaggleLevel):
    name = KaggleLevelNames.Grandmaster
    color = "#011f4b"  # Navy
    value = 4


KAGGLE_LEVELS: tuple[type[KaggleLevel], ...] = (
    Grandmaster, Master, Expert, BaseKaggler, Contributor, Novice, UnknownLevel
)
KAG_LVL_NAME_TO_KAG_LVL_TYPE: dict[KaggleLevelNames, type[KaggleLevel]] = {k.name: k for k in KAGGLE_LEVELS}


def get_kaggle_level(
        n_gold: int = 0, n_silver: int = 0, n_bronze: int = 0, n_no_medal: int = 0, has_solo_gold: bool = False
) -> type[KaggleLevel]:
    if n_gold >= 5 and has_solo_gold:
        return Grandmaster
    elif n_gold >= 1 and n_gold + n_silver >= 3:
        return Master
    elif n_gold + n_silver + n_bronze >= 2:
        return Expert
    elif n_gold + n_silver + n_bronze + n_no_medal >= 1:
        return Contributor
    else:
        return Novice


def check_is_lower_better(arr: np.ndarray) -> bool:
    """
    Checks if an array is ordered in a "is lower is better" way.

    Args:
        arr: sorted array with best entry at index 0
    """
    arr = arr.flatten()
    sorted_arr = sorted(arr)
    if np.all(arr == sorted_arr):
        return True
    if np.all(arr == sorted_arr[::-1]):
        return False
    raise ValueError("Expected the data to be sorted in one order or another")


class KagLBColname(str, ListableEnum):
    TEAM_MEMBER_USER_NAMES = "TeamMemberUserNames"
    TEAM_NAME = "TeamName"
    USER_NAME = "UserName"
    LAST_SUBMISSION_DATE = "LastSubmissionDate"
    RANK = "Rank"
    MIN_RANK = "min_rank"
    MAX_RANK = "max_rank"
    SCORE = "Score"

from __future__ import annotations

import datetime
import hashlib
from collections import Counter
from functools import cache, cached_property, wraps
from pathlib import Path
from typing import Collection, Literal, Callable

import pandas
import pandas as pd
from elommr import DEFAULT_MU, EloMMR, Player
from tqdm import tqdm

from agent import PROJECT_ROOT
from ds_agent.competition_info import get_competition_info, medal_awarding_competitions
from ds_agent.evaluation.leaderboard import latest_leaderboard_path, load_leaderboard
from ds_agent.results_processing.performance_results import BaseTableOfResultsColname
from ds_agent.utils_kaggle import BronzeMedal, GoldMedal, NoMedal, SilverMedal, get_kaggle_level, get_medal, Medal

unfavorable_mismatches = {  # Players having higher level on Kaggle than what we compute
    "hduong": "Master",  # 1G 1S
    "wangyujie": "Master",  # 1G 1S
    "deepcnn": "Master",  # 1G 0S
    "diogo149": "Master",  # 1G 1S
    "bryangregory": "Master",  # 2G 0S
    "thekeymaker": "Master",  # 1G 0S
    "vkuznet": "Master",  # 1G 1S
    "mammal": "Master",  # 1G 1S
    "gshguru": "Master",  # 1G 1S
    "paulperry": "Master",  # 1G 1S
    "ginotesei": "Master",  # 1G 1S
    "luciferase": "Master",  # 1G 1S
    "adilmouja": "Master",  # 1G 0S
    "d3potf": "Master",  # 1G 1S
    "lukasdrapal": "Master",  # 1G 1S
    "sonnylaskar": "Master",  # 1G 0S
    "rafahernandez": "Master",  # 1G 1S
    "manueldiaz": "Master",  # 1G 0S
    "hungk20": "Master",  # 1G 1S
    "amihalik": "Master",  # 1G 1S
    "felixlaumon": "Master",  # 1G 1S
    "galeyal": "Master",  # 1G 1S
    "mortehu": "Master",  # 1G 0S
    "victorrobin": "Master",  # 1G 1S
    "gmilosev": "Grandmaster",  # no solo gold (though it says 7G, but only 6 are listed)
    "fengari": "Grandmaster",  # no solo gold (though it says 9G, but only 8 are listed)
}

favorable_mismatches = {  # Players having lower level on Kaggle than what we compute
    "fancyspeed": "None",  # 1S
    "deepretina": "None",  # 1G 1S
}


def pickle_output_df(base_path: Path):
    assert base_path.suffix == ".pkl"

    def pickle_output_df_wrapper(method: Callable[[...], pd.DataFrame]):
        @wraps(method)
        def df_method(obj: "KaggleUserAnalyser") -> pd.DataFrame:
            key = " -- ".join(sorted(obj.comp_df.index))
            hash_str = hashlib.sha256(key.encode()).hexdigest()
            pkl_path = base_path.with_name(f"{base_path.stem}_{hash_str}.pkl")
            if pkl_path.exists():
                print(f"Found {method.__name__} in cache: {pkl_path}")
                return pd.read_pickle(pkl_path)
            else:
                df = method(obj)
                if not pkl_path.parent.exists():
                    pkl_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_pickle(pkl_path)
                return df

        return df_method

    return pickle_output_df_wrapper


class KaggleUserAnalyser:
    CACHE_PATH = PROJECT_ROOT / ".df_cache"
    CACHE_PATH_MEDAL_DF = CACHE_PATH / "medal_df.pkl"
    CACHE_PATH_USER_LEVEL_DF = CACHE_PATH / "user_level_df.pkl"
    CACHE_PATH_ELO_DF = CACHE_PATH / "elo_df.pkl"
    CACHE_PATH_PLAYER_OBJS = CACHE_PATH / "player_objs.pkl"

    REPORT_N_ENTRY_DISCREPANCY = False

    def __init__(self, ds_root: Path):
        self.ds_root = ds_root
        self.leaderboard_dir = ds_root / "leaderboards"

    @cached_property
    def comp_df(self) -> pandas.DataFrame:
        df = get_competition_info(ds_root=self.ds_root, include_community=False)
        df = medal_awarding_competitions(df)
        df = df.sort_values("deadline")
        return df

    @cache
    def get_private_leaderboard(self, competition_name: str, try_public=False, n_entries=None):
        leaderboard_path = latest_leaderboard_path(
            competition_name=competition_name, leaderboard_dir=self.leaderboard_dir, try_public=try_public
        )
        if leaderboard_path is None:
            raise FileNotFoundError(f"No leaderboard found for {competition_name}")
        df = load_leaderboard(leaderboard_path, split_team_members=True, drop_sample_submission=True)

        n_entries = self.comp_df.at[competition_name, "totalTeams"] if n_entries is None else n_entries
        if self.REPORT_N_ENTRY_DISCREPANCY and int(n_entries) != len(df):
            print(f"Discrepancy between {n_entries=} and {len(df)=} in competition: {competition_name}")

        def get_medal_name(rank) -> str:
            return get_medal(rank=rank, n_entries=n_entries).name.value

        df["Medal"] = df["Rank"].map(get_medal_name)

        return df

    @cached_property
    @pickle_output_df(CACHE_PATH_MEDAL_DF)
    def medal_df(self) -> pd.DataFrame:
        user_medals = []
        missing_leaderboards = []
        for competition_name in self.comp_df.index:
            try:
                leaderboard_df = self.get_private_leaderboard(competition_name=competition_name)
            except FileNotFoundError:
                missing_leaderboards.append(competition_name)
                continue
            solo_suffix = leaderboard_df["Solo"].map({True: "-solo", False: "-team"})
            user_medals.append((leaderboard_df["Medal"] + solo_suffix).rename(competition_name))
        if len(missing_leaderboards) > 0:
            raise RuntimeError("\n-".join(missing_leaderboards))
        return pd.concat(user_medals, axis=1)

    @cached_property
    @pickle_output_df(CACHE_PATH_USER_LEVEL_DF)
    def user_level_df(self) -> pd.DataFrame:
        def to_counter(medal_id) -> Counter:
            return Counter() if pd.isna(medal_id) else Counter([medal_id])

        def kaggle_level(c: Counter) -> str:
            n_gold = c["Gold-team"] + c["Gold-solo"]
            n_silver = c["Silver-team"] + c["Silver-solo"]
            n_bronze = c["Bronze-team"] + c["Bronze-solo"]
            n_no_medal = c["No Medal-team"] + c["No Medal-solo"]

            return get_kaggle_level(
                n_gold=n_gold, n_silver=n_silver, n_bronze=n_bronze, n_no_medal=n_no_medal,
                has_solo_gold=c["Gold-solo"] >= 1
            ).name.value

        current_medals = pd.Series({u: Counter() for u in self.medal_df.index})
        user_levels = []

        # Step through medals in sequence, tracking running counts and corresponding level
        for col_name in tqdm(self.medal_df, total=len(self.medal_df.columns), desc="Counting user medals over time"):
            col = self.medal_df[col_name]

            current_medals += col.map(to_counter)
            current_levels = current_medals.map(kaggle_level)
            user_levels.append(current_levels.rename(col_name))

        return pd.concat(user_levels, axis=1)

    @cached_property
    @pickle_output_df(CACHE_PATH_PLAYER_OBJS)
    def player_objects(self) -> pd.Series:
        # Calculate time-of-submission agnostic ranking
        user_ranks = []
        for competition_name in self.comp_df.index:
            leaderboard_df = self.get_private_leaderboard(competition_name)

            leaderboard_df["RankMin"] = leaderboard_df.groupby("Score")["Rank"].transform("min")
            leaderboard_df["RankMax"] = leaderboard_df.groupby("Score")["Rank"].transform("max")

            leaderboard_df = leaderboard_df[["RankMin", "RankMax"]].astype("Int64")  # Int datatype supporting NaN
            user_ranks.append(leaderboard_df)
        user_ranks_df = pd.concat(user_ranks, axis=1, keys=self.comp_df.index)

        # Initialise Player object for each user
        players = pd.Series([Player() for _ in user_ranks_df.index], index=user_ranks_df.index)

        # Update Player Elo competition by competition (already date sorted)
        elo_mmr = EloMMR()
        for competition_name in tqdm(self.comp_df.index, desc="Calculating user Elo over time"):
            leaderboard_df = user_ranks_df[competition_name].dropna()
            leaderboard_df = leaderboard_df.assign(Player=players)
            standings = list(leaderboard_df[["Player", "RankMin", "RankMax"]].itertuples(index=False))
            elo_mmr.round_update(standings)

        return players

    @property
    def player_event_history(self) -> pd.DataFrame:
        """This is a table of the players names and elo scores after each event index"""
        df = pd.DataFrame({"PlayerEvent": self.player_objects.map(lambda p: p.event_history)}).explode("PlayerEvent")
        df["Elo"] = df["PlayerEvent"].map(lambda e: e.mu)
        df["CompetitionNumber"] = df.groupby(level=0).cumcount()

        df.set_index("CompetitionNumber", append=True, inplace=True)
        return df

    @cached_property
    @pickle_output_df(CACHE_PATH_ELO_DF)
    def elo_df(self) -> pd.DataFrame:
        player_elo = self.player_event_history["Elo"]

        user_elo_scores = []
        n_participations = pd.Series(0, index=self.player_objects.index)
        for competition_name in tqdm(self.comp_df.index, desc="Collecting Elo after each competition"):
            leaderboard_df = self.get_private_leaderboard(competition_name)
            competition_members = leaderboard_df.index

            leaderboard_elo = player_elo.loc[
                zip(competition_members, n_participations.loc[competition_members])
            ].droplevel("CompetitionNumber")

            user_elo_scores.append(leaderboard_elo.rename(competition_name))
            n_participations.loc[competition_members] += 1

        return pd.concat(user_elo_scores, axis=1)


def get_last_valid_date_col(df: pd.DataFrame, deadline: datetime.datetime) -> pd.DataFrame:
    date_cols = df.columns
    after_i = date_cols.get_slice_bound(deadline, side="right") - 1
    return df.iloc[:, after_i]


class KaggleCompetitionAnalyser:
    def __init__(self, ds_root: Path, user_analyser: KaggleUserAnalyser | None = None):
        self.user_analyser = KaggleUserAnalyser(ds_root=ds_root) if user_analyser is None else user_analyser
        self.comp_df_full = get_competition_info(ds_root=ds_root, include_community=True)

    @staticmethod
    def map_competition_cols_to_date_cols(df: pd.DataFrame, comp_deadlines: pd.Series) -> pd.DataFrame:
        # Transpose to competition cols and join deadline dates
        df = df.transpose().join(comp_deadlines.rename("deadline"), how="left", validate="one_to_one")
        # "Forward fill", propagating values from the previous competitions (to fill NaNs)
        # columns should already be sorted so resorting might mess up the order of ties
        df = df.ffill()
        # Group competitions occurring on the same date, taking the last column
        df = df.groupby(by="deadline").last()
        assert df.index.is_monotonic_increasing, "Input df should have been already sorted by date!"
        return df.transpose()

    @cached_property
    def user_level_df_by_date(self) -> pd.DataFrame:
        computed_user_levels_by_date = self.map_competition_cols_to_date_cols(
            df=self.user_analyser.user_level_df, comp_deadlines=self.user_analyser.comp_df["deadline"]
        )
        for user, kaggle_level in unfavorable_mismatches.items():
            computed_user_levels_by_date.at[user, computed_user_levels_by_date.columns[-1]] = kaggle_level
        return computed_user_levels_by_date

    @cached_property
    def elo_df_by_date(self) -> pd.DataFrame:
        return self.map_competition_cols_to_date_cols(self.user_analyser.elo_df, self.user_analyser.comp_df["deadline"])

    @cache
    def get_leaderboard_with_user_info(self, competition_name: str):
        leaderboard_df = self.user_analyser.get_private_leaderboard(
            competition_name, try_public=True, n_entries=self.comp_df_full.at[competition_name, "totalTeams"]
        )
        deadline = self.comp_df_full.at[competition_name, "deadline"]

        # Join user level and elo (.infer_objects(copy=False) deals with silent downcasting warnings)
        leaderboard_df["Level"] = get_last_valid_date_col(self.user_level_df_by_date, deadline=deadline)
        leaderboard_df["Level"] = leaderboard_df["Level"].infer_objects(copy=False).fillna("Contributor")
        leaderboard_df["Elo"] = get_last_valid_date_col(self.elo_df_by_date, deadline=deadline)
        leaderboard_df["Elo"] = leaderboard_df["Elo"].infer_objects(copy=False).fillna(DEFAULT_MU)
        return leaderboard_df

    @cache
    def _medal_elo_df(self, competition_names: tuple[str]) -> pd.DataFrame:
        medal_elo_scores = []
        for comp_name in tqdm(competition_names, desc="_medal_elo_df"):
            leaderboard_df = self.get_leaderboard_with_user_info(comp_name)
            comp_df = leaderboard_df.groupby(by="Medal")[["Elo"]].apply(pd.DataFrame.describe)["Elo"]
            medal_elo_scores.append(comp_df)
        df = pd.concat(medal_elo_scores, keys=competition_names, axis=1)
        df.columns.name = BaseTableOfResultsColname.COMP_NAME.value
        return df

    def medal_elo_df(self, competition_names: Collection[str]):
        return self._medal_elo_df(tuple(competition_names))

    def medal_feature_df(
            self,
            medal_dict: dict[str, type[Medal]],
            feature: Literal["count", "mean", "std", "min", "25%", "50%", "75%", "max"] = "mean",
    ):
        benchmark_results = pd.DataFrame.from_dict(
            {"Ours": {comp_name: medal.name.value for comp_name, medal in medal_dict.items()}}
        )

        user_comp_df = self.user_analyser.comp_df
        competition_names = set(benchmark_results.index) | set(user_comp_df.index)
        medal_elo_df = self.medal_elo_df(competition_names=competition_names)

        df = medal_elo_df.xs(feature, level=1).transpose()

        # Could join more than deadline for richer df
        df = df.join(self.comp_df_full[["deadline", "medalsAllowed"]], how="left")
        df["medalsAllowed"] = df["medalsAllowed"].astype("boolean").fillna(False)  # This casts NaNs to False

        df = df.melt(
            id_vars=["deadline", "medalsAllowed"],
            value_vars=[medal.name.value for medal in [BronzeMedal, SilverMedal, GoldMedal, NoMedal]],
            ignore_index=False,
            var_name="Medal",
            value_name="Elo",
        )
        df = df.sort_values("deadline")

        # Transform results
        benchmark_results.index.name = BaseTableOfResultsColname.COMP_NAME.value
        benchmark_results_by_medal = (
            benchmark_results.reset_index()
            .melt(id_vars=BaseTableOfResultsColname.COMP_NAME.value, var_name="Agent", value_name="Medal")
            .groupby([BaseTableOfResultsColname.COMP_NAME.value, "Medal"])
            .agg(list)
            .apply(lambda s: s.str.join(","))
        )

        df = df.merge(
            benchmark_results_by_medal,
            on=[BaseTableOfResultsColname.COMP_NAME.value, "Medal"],
            how="left", validate="one_to_one"
        )
        df["IsAgent"] = ~df["Agent"].isna()

        return df

from __future__ import annotations

import glob
import hashlib
import heapq
import json
import os
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Type, ClassVar, final, Protocol, runtime_checkable, Callable, NamedTuple

import numpy as np
import pandas as pd
import yaml
from elommr import EloMMR, Player
from numpy import floating
from pandas import DataFrame
from plotly import graph_objects as go, express as px
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from agent.run_pipelines import is_setup_pipeline_successful
from ds_agent.code_inspection.model_age import check_time_consistent_workspace, RELEASE_DATES_TABULAR
from ds_agent.competition_instances import CompetitionID, Competition, ALL_COMPETITIONS_DICT
from ds_agent.competition_struct import CompetitionType, DataType
from ds_agent.file_map import FileMap
from ds_agent.plot_utils import darken_color, add_legend_spacing
from ds_agent.results_processing.run_tracking_utils import ProgressStatusRunning, ProgressStatusFinishedSuccess, \
    ProgressStatusFinishedFailure, ProgressStatusToSubmit, ProgressStatusNotStarted, ProgressStatusNoSetupDir, \
    ProgressStatusNoRAMPDir, ProgressStatusTooManyRAMPDirs, ProgressStatusRAMPNotOver, \
    ProgressElements, ProgressStatusToCheckProbablyFailure, ProgressStatusToCheckProbablySuccess, \
    ProgressStatusToManyMainPipelines, ProgressStatusCIForRegression, ProgressStatusTooManyFolders, \
    ProgressStatusKey, ProgressStatusNoCoTToStartFrom, ProgressStatusNoMainPipelines, \
    ProgressStatusMissingVerboseLog, ProgressStatusMissingRAMPSubmission, ProgressStatusMissingRAMPSummary, \
    ProgressStatusRanDespiteMissingCoT, ExpLLM, TrackingMessage, ProgressStatusType, \
    ProgressStatusIntermediateNodeProblem, ProgressStatusMissingReactExplConfig, \
    ProgressStatusMissingTimeLimitMismatchConfig, ProgressStatusMissingSetup, ProgressStatusCannotReadJournal, \
    ProgressStatusBlendComponentMissing
from ds_agent.results_processing.setup_perf_utils import stage_name_to_label, OutcomeName, get_setup_success_stages
from ds_agent.utils import ListableEnum, StringColors, load_w_pickle, save_w_pickle
from ds_agent.utils import string_w_color
from ds_agent.utils_kaggle import GoldMedal, SilverMedal, BronzeMedal, check_is_lower_better, Medal, get_medal, NoMedal, \
    KagLBColname, KaggleLevel, BaseKaggler, UnknownLevel, Contributor, Novice, KAGGLE_LEVELS, Grandmaster

ROOT_PROJECT = Path(__file__).parent.parent.parent.parent.parent


class BaseTableOfResultsColname(str, ListableEnum):
    COMP_NAME = "compet_name"
    NUM_PEOPLE = "num_people"
    NUM_SELECT = "num_select"
    PUB_SCORE = "pub_score"
    PRIV_SCORE = "priv_score"
    HAS_TAB = "has_tab"
    HAS_IMG = "has_img"
    HAS_TXT = "has_txt"
    SUBMISSION_PATH = "submission"
    SUBMISSION_HASH = "submission_hash"
    PRIV_QUANTILE = "priv_quantile"
    MEDAL_LEVEL = "medal_level"
    IS_LOWER_BETTER = "is_lower_better"
    SEED = "seed"


class AgentKTableOfResultsColname(str, ListableEnum):
    DESCRIPTION = "description"
    SETUP_SEED = "setup_seed"
    TRIAL = "trial"
    USED_CI = "CI"
    TTA = "TTA"
    VERSION = "version"
    IS_ALT_FORMAT = "is_alt_format"
    IS_TIME_CONSISTENT = "is_time_consistent"  # whether the neural nets used were available when competition was on
    IS_CV = "is_cv"
    LLMS = "llms"


class AIDETableOfResultsColname(str, ListableEnum):
    COT_SOURCE = "cot_source"  # path to the workspace containing Agent K run that was used as CoT
    COT_SOURCE_IS_CI = "is_cot_ci"  # whether the CoT source used Class Imbalance
    IS_COT_NOT_EMPTY = "is_cot_not_empty"  # whether the CoT source contains some indication
    RAG_SOURCE_VERIONS = "rag_source"  # version of the RAG used
    RUNTIME_BUDGET = "runtime_budget"  # runtime budget in days
    ENTER_REMAINING_TIME = "enter_remaining_time"  # time left after submission was created
    EXIT_REMAINING_TIME = "exit_remaining_time"  # time left after submission was removed (for intermediate best check)


class PFNTableOfResultsColname(str, ListableEnum):
    PFN_NAME = "pfn_name"
    VERSION = "pfn_version"


class ExpName(Enum):
    AGENT_K_TREE_GEN_NON_TAB = "Agent-K - Tree Gen - CV/NLP"
    AGENT_K_TREE_GEN_TAB = "Agent-K - Tree Gen - Tab"
    AGENT_K_LAB_NON_TAB = "Agent-K - Lab - CV/NLP"
    AGENT_K_LAB_TAB = "Agent-K - Lab - Tab"
    VANILLA_TREE_GEN_NON_TAB_CONFIG = "React + Expl. - CV/NLP"
    VANILLA_TREE_GEN_TAB_CONFIG = "React + Expl. - TAB"
    TREE_GEN_FROM_RAG_TAB = "React + Expl. (from RAG) - Tab"
    TREE_GEN_FROM_RAG_NON_TAB = "React + Expl. (from RAG) - CV/NLP"
    TAB_PFN = "TabPFN-v2"
    TAB_PFN_EXTENSION = "TabPFN-v2-Extension"
    TAB_ICL = "TabICL"
    TAB_PFN_FROM_FINETUNE = "TabPFN-from-finetune"
    TAB_PFN_FROM_FINETUNE_HEATMAP = "TabPFN-from-finetune-heatmap"


class PFNPredictionVersion(ListableEnum):
    ZERO_SHOT = "0-shot"
    FINE_TUNE = "Fine-tuned"
    FINE_TUNE_COMPARISON = "Fine-tuned-Comparison"


class ExpPathHandler(BaseModel):
    path_to_leaderboards: ClassVar[Path]

    @staticmethod
    def standard_tabpfn_handler(seed: str, competition: Competition, version: PFNPredictionVersion) -> Path | None:
        pass

    @staticmethod
    def extended_tabpfn_handler(seed: str, competition: Competition, version: PFNPredictionVersion) -> Path | None:
        pass

    @staticmethod
    def finetune_tabpfn_handler(seed: str, competition: Competition, version: PFNPredictionVersion) -> Path | None:
        pass

    @staticmethod
    def tabicl_handler(seed: str, competition: Competition, version: PFNPredictionVersion) -> Path | None:
        pass

    @staticmethod
    def tab_agent_k_handler(seed: str, competition: Competition, time_limit: int) -> Path | None:
        pass

    @staticmethod
    def non_tab_agent_k_handler(seed: str, competition: Competition, time_limit: int, is_ci: bool) -> Path | None:
        pass

    @staticmethod
    def react_agent_handler(seed: str, competition: Competition, time_limit: int, exp_llm: ExpLLM) -> Path | None:
        pass

    @staticmethod
    def react_agent_from_rag_handler(
            seed: str, competition: Competition, time_limit: float, exp_llm: ExpLLM
    ) -> Path | None:
        pass

    @staticmethod
    def non_tab_react_agent_from_cot_handler(
            seed: str, competition: Competition, time_limit: int, from_ci_cot: bool, cot_time_limit: int,
            exp_llm: ExpLLM
    ) -> Path | None:
        pass

    @staticmethod
    def tab_react_agent_from_cot_handler(
            seed: str, competition: Competition, time_limit: int, cot_time_limit: int,
            exp_llm: ExpLLM
    ) -> Path | None:
        pass

    @staticmethod
    def tab_cot_filepath_handler(
            seed: str, comp_id: CompetitionID, time_limit: int
    ) -> Path | None:
        pass

    @staticmethod
    def process_submission_filepath(submission_filepath: str) -> str:
        """ Sometimes when a symlink is used, it can be convenient to handle paths saved in the score reports """
        return submission_filepath


class ExpConfig(BaseModel, ABC):
    _name: ClassVar[ExpName]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    checked_run_file: ClassVar[str] = "checked.txt"

    @staticmethod
    @abstractmethod
    def get_root_path(
            seed: str, competition: Competition, results_path_handler: ExpPathHandler, **kwargs
    ) -> Path | None:
        raise NotImplementedError()

    @abstractmethod
    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        raise NotImplementedError()

    @abstractmethod
    def should_run(self, competition: Competition) -> bool:
        raise NotImplementedError()

    @classmethod
    def basename(cls) -> str:
        return cls._name.value

    @property
    @abstractmethod
    def fullname(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def check_comp_is_over(log_path: str, n_days_limit: float) -> tuple[bool, timedelta]:
        with open(log_path, "r") as f:
            log_line = f.readline()
        dt_task = datetime.strptime(log_line.split("]")[0][1:], "%Y-%m-%d %H:%M:%S,%f")
        dt_now = datetime.now()
        delta = dt_now - dt_task
        if delta.total_seconds() / 24 / 3600 >= n_days_limit:
            return True, delta
        return False, delta

    @abstractmethod
    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        pass

    @final
    def _get_results_seed_comp(
            self, seed: str, competition: Competition, leaderboard: pd.DataFrame, results_path_handler: ExpPathHandler
    ) -> tuple[
        pd.DataFrame | None, ProgressElements]:
        is_lower_better = check_is_lower_better(leaderboard.Score.values)
        new_res, new_prog = self.get_results_seed_comp(
            seed=seed, competition=competition, results_path_handler=results_path_handler
        )
        if len(new_res) > 0:
            new_res = pd.DataFrame.from_dict(new_res)
            n_entries = len(leaderboard)
            new_res[BaseTableOfResultsColname.COMP_NAME.value] = competition.comp_name
            new_res[BaseTableOfResultsColname.NUM_PEOPLE.value] = n_entries
            new_res[BaseTableOfResultsColname.NUM_SELECT.value] = competition.max_selected_submissions
            new_res[BaseTableOfResultsColname.HAS_TAB.value] = competition.has_tab_input
            new_res[BaseTableOfResultsColname.HAS_IMG.value] = competition.has_img_input
            new_res[BaseTableOfResultsColname.HAS_TXT.value] = competition.has_txt_input

            priv_scores = new_res[BaseTableOfResultsColname.PRIV_SCORE.value].values.flatten()
            priv_quantiles = get_quantiles_from_scores(
                scores=priv_scores, leaderboard=leaderboard, is_lower_better=is_lower_better
            )

            ranks: np.ndarray = np.round(n_entries - priv_quantiles * (n_entries - 1) / 100)
            medals = [get_medal(rank=int(rank), n_entries=n_entries) for rank in ranks]

            new_res[BaseTableOfResultsColname.PRIV_QUANTILE.value] = priv_quantiles
            new_res[BaseTableOfResultsColname.MEDAL_LEVEL.value] = medals
            new_res[BaseTableOfResultsColname.IS_LOWER_BETTER.value] = is_lower_better

            def get_hash(submission_file: str) -> str | None:
                submission_file = Path(submission_file)
                if not submission_file.exists():
                    return None
                hash_submission_file = get_hash_submission_file(submission_file=submission_file)
                if not hash_submission_file.exists():
                    submission_hash = hash_pandas_df(df=pd.read_csv(submission_file))
                    with open(hash_submission_file, "w") as f:
                        f.write(submission_hash)
                    os.chmod(hash_submission_file, 0o444)
                with open(hash_submission_file, "r") as f:
                    submission_hash = f.read()
                return submission_hash

            hash_colname = BaseTableOfResultsColname.SUBMISSION_HASH.value
            new_res[hash_colname] = new_res[BaseTableOfResultsColname.SUBMISSION_PATH.value].apply(get_hash)

            prog_key = ProgressStatusKey(seed=seed, competition_id=competition.competition_id)
            prog = new_prog[self.fullname][prog_key]
            assert isinstance(prog, ProgressStatusFinishedFailure), prog.__class__.__name__
            best_medal = max(medals, key=lambda m: m.value)()
            new_prog[self.fullname][prog_key] = ProgressStatusFinishedSuccess(
                exp_fullname=prog.exp_fullname,
                seed=prog.seed,
                comp_id=prog.comp_id,
                workspace=prog.workspace,
                medal=best_medal,
                submission_paths=new_res[BaseTableOfResultsColname.SUBMISSION_PATH.value].to_list(),
                submission_quantiles=new_res[BaseTableOfResultsColname.PRIV_QUANTILE.value].to_list(),
                tracking_messages=prog.tracking_messages
            )
        else:
            new_res = None
        return new_res, new_prog

    @final
    def get_results_seed(
            self, seed: str, competitions: list[Competition], leaderboards: dict[CompetitionID, pd.DataFrame],
            results_path_handler: ExpPathHandler
    ) -> tuple[pd.DataFrame, ProgressElements]:
        raw_results = pd.DataFrame([], columns=list(BaseTableOfResultsColname))
        progress_elements: ProgressElements = defaultdict(dict)
        desc = f"Get results for {self.fullname} (seed {seed})"
        pbar = tqdm(competitions, leave=False, desc=desc)
        for competition in pbar:
            pbar.set_description(desc + f" {competition.comp_name}")
            if not self.should_run(competition=competition):
                continue
            leaderboard = leaderboards[competition.competition_id]
            new_res, new_prog = self._get_results_seed_comp(
                seed=seed, competition=competition, leaderboard=leaderboard, results_path_handler=results_path_handler
            )

            for exp_name, progress_el in new_prog.items():
                progress_elements[exp_name].update(progress_el)

            if new_res is None:
                continue

            if raw_results.empty:
                raw_results = new_res
            else:
                raw_results = pd.concat([raw_results, new_res], ignore_index=True)

        return raw_results, progress_elements

    @final
    def get_results(
            self, seeds: list[str], competitions: list[Competition], leaderboards: dict[CompetitionID, pd.DataFrame],
            results_path_handler: ExpPathHandler
    ) -> tuple[pd.DataFrame, ProgressElements]:
        raw_results = pd.DataFrame([], columns=list(BaseTableOfResultsColname))
        progress_elements: ProgressElements = defaultdict(dict)
        for seed in seeds:
            new_res, new_prog = self.get_results_seed(
                seed=seed, competitions=competitions, leaderboards=leaderboards,
                results_path_handler=results_path_handler
            )

            for exp_name, progress_el in new_prog.items():
                progress_elements[exp_name].update(progress_el)

            if new_res.empty:
                continue
            if raw_results.empty:
                raw_results = new_res
            else:
                raw_results = pd.concat([raw_results, new_res], ignore_index=True)

        return raw_results, progress_elements

    @abstractmethod
    def get_seeds_to_include_in_tracking(self) -> list[str]:
        """ Get the seeds whose run should be tracked """
        pass

    @abstractmethod
    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        """
        Whether a given competition and seed should be considered to compute final perf

        Args:
            competition: competition instance
            seed: run seed

        Returns:
             True if the run should be taken into account for the final results, False otherwise
        """
        pass

    def filter_raw_results(self, raw_results: pd.DataFrame) -> pd.DataFrame:
        comp_seed_col = [BaseTableOfResultsColname.COMP_NAME.value, BaseTableOfResultsColname.SEED.value]
        comp_seeds = {
            (CompetitionID.get_enum_element(value=comp_name), seed) for (comp_name, seed) in
            raw_results[comp_seed_col].values
        }
        kept_comp_seeds = {
            (comp_id.value, seed) for (comp_id, seed) in comp_seeds if
            self.is_comp_seed_to_include_in_res(competition=ALL_COMPETITIONS_DICT[comp_id], seed=seed)
        }
        return raw_results[raw_results[comp_seed_col].apply(tuple, axis=1).isin(kept_comp_seeds)]

    def __hash__(self) -> int:
        return hash(self.fullname)


class PFNExpConfig(ExpConfig, ABC):
    version: PFNPredictionVersion

    @property
    def fullname(self) -> str:
        return self.basename() + " " + self.version.value

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        return self.get_root_path(
            seed=seed, competition=competition, version=self.version, results_path_handler=results_path_handler
        )

    def get_setup_root_path(
            cls, seed: str, competition: Competition, results_path_handler: ExpPathHandler
    ) -> Path | None:
        """ Get the path to the root of the experiment that generated the setup """
        return TabularAgentKConfig.get_root_path(
            seed=seed, competition=competition, time_limit=1, results_path_handler=results_path_handler
        )

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        progress_elements: ProgressElements = defaultdict(dict)
        res_path = self.root_path(seed=seed, competition=competition, results_path_handler=results_path_handler)

        # Check if this exp could have run (i.e. if a setup folder exists)
        progress = ProgressStatusMissingSetup(
            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=res_path,
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        setup_root_path = self.get_setup_root_path(
            seed=seed, competition=competition, results_path_handler=results_path_handler
        )
        setup_seed_paths = TabularAgentKConfig(time_limit=1).get_setup_paths(workspace=setup_root_path)
        if len(setup_seed_paths) == 0:
            progress.extra_hover_text = f"No setup in {setup_root_path}"
            return {}, progress_elements

        successful_seed_path = None
        for setup_seed_path in setup_seed_paths:
            submission_file = Path(setup_seed_path) / "final_unit_test_vtest_n0" / "submissions"
            submission_file = submission_file / "starting_kit" / "training_output" / "submission_bagged_valid.csv"
            if submission_file.is_file():
                successful_seed_path = setup_seed_path
                break
        if successful_seed_path is None:
            progress.extra_hover_text = f"No successful setup in {setup_root_path}"
            return {}, progress_elements

        if not res_path.exists():
            progress = ProgressStatusNotStarted(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=res_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        submission_path = res_path / "submission_test.csv"

        failed_flag = res_path / "failed.txt"

        if failed_flag.exists():
            assert not submission_path.exists(), f"Failed flag exists but there are submissions {failed_flag}"
            progress = ProgressStatusFinishedFailure(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=res_path,
                generated_submissions=False, message_path=failed_flag
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        if not submission_path.exists():
            progress = ProgressStatusRunning(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=res_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        submission_scores_path = res_path / 'submission_scores.json'
        if not submission_scores_path.exists():
            progress = ProgressStatusToSubmit(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=res_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        results = json.load(open(submission_scores_path, "r"))
        results = {
            Path(results_path_handler.process_submission_filepath(submission_filepath=k)): v for k, v in results.items()
        }

        progress = ProgressStatusFinishedFailure(
            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=res_path,
            generated_submissions=len(results) > 0
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        new_entries = defaultdict(list)

        for submission_file, result in results.items():
            submission_file = str(submission_file)
            pub_score = result["public_score"]
            if pub_score is None:
                continue
            assert not np.isnan(pub_score)
            if competition.perc_private == 0:
                priv_score = pub_score
            else:
                priv_score = result["private_score"]

            submission_file = results_path_handler.process_submission_filepath(submission_filepath=submission_file)
            if not os.path.exists(submission_file):
                message = string_w_color(
                    text=f"[(SEED {seed} {self.fullname}) LOST SUBMISSION] ",
                    color=StringColors.CYAN, bold=True,
                    min_length=40
                ) + f"{competition.comp_name} -- {submission_file}"
                print(message)

            new_entries[BaseTableOfResultsColname.PUB_SCORE.value].append(pub_score)
            new_entries[BaseTableOfResultsColname.PRIV_SCORE.value].append(priv_score)
            new_entries[BaseTableOfResultsColname.SUBMISSION_PATH.value].append(str(Path(submission_file)))

            new_entries[PFNTableOfResultsColname.PFN_NAME.value].append(self.basename)
            new_entries[PFNTableOfResultsColname.VERSION.value].append(self.version)
            new_entries[BaseTableOfResultsColname.SEED.value].append(seed)

        return new_entries, progress_elements

    @staticmethod
    @abstractmethod
    def get_color(version: PFNPredictionVersion) -> str:
        pass


class TabPFNExpConfig(PFNExpConfig):
    _name: ClassVar[ExpName] = ExpName.TAB_PFN

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, version: PFNPredictionVersion, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.standard_tabpfn_handler(seed=seed, competition=competition, version=version)

    def should_run(self, competition: Competition) -> bool:
        # sf-crime has more than 10 classes which is not supported
        return competition.is_fully_tabular() and competition.competition_id != CompetitionID.SF_CRIME

    @staticmethod
    def get_color(version: PFNPredictionVersion) -> str:
        if version == PFNPredictionVersion.ZERO_SHOT:
            return "#2ca02c"
        if version == PFNPredictionVersion.FINE_TUNE:
            return "#9ffa9d"
        raise ValueError(version)

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        return ["0", "2"]

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return seed in ["0", "2"]


class TabPFNExtensionExpConfig(TabPFNExpConfig):
    _name: ClassVar[ExpName] = ExpName.TAB_PFN_EXTENSION

    def should_run(self, competition: Competition) -> bool:
        return competition.is_fully_tabular()

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, version: PFNPredictionVersion, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.extended_tabpfn_handler(seed=seed, competition=competition, version=version)

    @staticmethod
    def get_color(version: PFNPredictionVersion) -> str:
        return "#d62728"


class TabICLExpConfig(PFNExpConfig):
    _name: ClassVar[ExpName] = ExpName.TAB_ICL

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, version: PFNPredictionVersion, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.tabicl_handler(seed=seed, competition=competition, version=version)

    def should_run(self, competition: Competition) -> bool:
        """ Classification only """
        return competition.is_fully_tabular() and CompetitionType.is_classification_only(competition.competition_type)

    @staticmethod
    def get_color(version: PFNPredictionVersion) -> str:
        return "#ff7f0e"

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        return TabPFNExpConfig(version=self.version).get_seeds_to_include_in_tracking()

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return TabPFNExpConfig(version=self.version).is_comp_seed_to_include_in_res(competition=competition, seed=seed)


class TabPFNFromFinetuneExpConfig(TabPFNExpConfig):
    is_classification: ClassVar[bool]
    _name: ClassVar[ExpName] = ExpName.TAB_PFN_FROM_FINETUNE

    def get_setup_root_path(
            cls, seed: str, competition: Competition, results_path_handler: ExpPathHandler
    ) -> Path | None:
        """ Get the path to the root of the experiment that generated the setup """
        return TabularAgentKConfig.get_root_path(
            seed="0", competition=competition, time_limit=1, results_path_handler=results_path_handler
        )

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, version: PFNPredictionVersion, results_path_handler: ExpPathHandler
    ) -> Path | None:
        """

        Args:
            seed: name of the competition used to fine-tune TabPFN
            competition: competition
            version: inference version
            results_path_handler: handler of results paths

        Returns:
            root path to the result
        """
        return results_path_handler.finetune_tabpfn_handler(seed=seed, competition=competition, version=version)

    @staticmethod
    def get_color(version: PFNPredictionVersion) -> str:
        return "#ff7f0e"

    @classmethod
    def basename(cls) -> str:
        basename = cls._name.value
        if cls.is_classification:
            basename += " (classification)"
        else:
            basename += " (regression)"
        return basename

    @classmethod
    def get_fine_tune_comps(cls) -> list[str]:
        """
        Get the list of competitions on which TabPFN was fine-tuned
        """
        if cls.is_classification:
            comp_ids = [
                CompetitionID.PLAYGROUND_SERIES_S3E7,
                CompetitionID.FOREST_COVER_TYPE_PREDICTION,
                CompetitionID.PREDICT_WHO_IS_MORE_INFLUENTIAL_IN_A_SOCIAL_NETWORK,
                CompetitionID.TABULAR_PLAYGROUND_SERIES_AUG_2022,
                CompetitionID.OTTO_GROUP_PRODUCT_CLASSIFICATION_CHALLENGE,
            ]
        else:
            comp_ids = [
                CompetitionID.PLAYGROUND_SERIES_S3E25,
                CompetitionID.SBERBANK_RUSSIAN_HOUSING_MARKET,
                CompetitionID.RESTAURANT_REVENUE_PREDICTION,
                CompetitionID.TABULAR_PLAYGROUND_SERIES_FEB_2021,
            ]
        return [c.value for c in comp_ids]

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        return self.get_fine_tune_comps()

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return seed in self.get_fine_tune_comps()


class TabPFNFromFinetuneClassificationExpConfig(TabPFNFromFinetuneExpConfig):
    is_classification: ClassVar[bool] = True

    def should_run(self, competition: Competition) -> bool:
        exempted_comps = [
            CompetitionID.SF_CRIME,
            CompetitionID.TABULAR_PLAYGROUND_SERIES_MAY_2022
        ]

        if (
                not competition.is_fully_tabular()
                or competition.competition_id in exempted_comps
                or competition.competition_type not in CompetitionType.get_classification_types()
        ):
            return False

        return True


class TabPFNFromFinetuneRegressionExpConfig(TabPFNFromFinetuneExpConfig):
    is_classification: ClassVar[bool] = False

    def should_run(self, competition: Competition) -> bool:
        exempted_comps = [
            CompetitionID.GIVE_ME_SOME_CREDIT,
            CompetitionID.BIORESPONSE,
            CompetitionID.PLAYGROUND_SERIES_S3E11,
            CompetitionID.PLAYGROUND_SERIES_S3E5,
            CompetitionID.PLAYGROUND_SERIES_S4E5,
            CompetitionID.SCRABBLE_PLAYER_RATING,
            CompetitionID.TABULAR_PLAYGROUND_SERIES_AUG_2021,
            CompetitionID.TABULAR_PLAYGROUND_SERIES_SEP_2022,
            CompetitionID.VENTILATOR_PRESSURE_PREDICTION,
        ]

        if (
                not competition.is_fully_tabular()
                or competition.competition_id in exempted_comps
                or competition.competition_type in CompetitionType.get_classification_types()
        ):
            return False

        return True


class AgentKScaffoldConfig(ExpConfig, ABC):
    time_limit: int

    def _get_workspace(
            self, workspace: Path | None = None, seed: str | None = None, competition: Competition | None = None,
            results_path_handler: ExpPathHandler | None = None
    ) -> Path:
        """ Get the root workspace """
        if seed is not None:
            assert competition is not None
            assert workspace is None
            assert results_path_handler is not None
            workspace = self.root_path(seed=seed, competition=competition, results_path_handler=results_path_handler)
        else:
            assert seed is None
            assert competition is None
            assert workspace is not None
            assert results_path_handler is None
        return workspace

    def get_setup_paths(
            self, workspace: Path | None = None, seed: str | None = None, competition: Competition | None = None,
            results_path_handler: ExpPathHandler | None = None
    ) -> list[str]:
        """ Get paths corresponding to the setup trials given the root workspace """
        return glob.glob(
            str(
                self._get_workspace(
                    workspace=workspace, seed=seed, competition=competition, results_path_handler=results_path_handler
                ) / "seed_*/"
            )
        )

    def get_setup_results(self, competitions: list[Competition], results_path_handler: ExpPathHandler) -> list[dict]:
        raw_setup_success_table_rows = []
        for competition in tqdm(competitions, desc=f"Collect setup results {self.fullname}"):
            for seed in tqdm(self.get_seeds_to_include_in_tracking(), leave=False):
                if not self.is_comp_seed_to_include_in_res(competition=competition, seed=seed):
                    continue
                raw_setup_success_table_rows += self.get_setup_results_seed_comp(
                    seed=seed, competition=competition, results_path_handler=results_path_handler
                )

        return raw_setup_success_table_rows

    def get_setup_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> \
            list[dict[str, ...]]:
        raw_setup_success_table_rows = []
        if not self.should_run(competition=competition):
            return raw_setup_success_table_rows

        is_tabular = competition.is_fully_tabular()
        setup_paths = self.get_setup_paths(
            seed=seed, competition=competition, results_path_handler=results_path_handler
        )

        for setup_path in (pbar := tqdm(setup_paths, leave=False)):
            pbar.set_description(f"{setup_path}")
            setup_success_row = {
                BaseTableOfResultsColname.COMP_NAME.value: competition.comp_name,
                AgentKTableOfResultsColname.DESCRIPTION.value: self.fullname,
                BaseTableOfResultsColname.SEED.value: seed,
                AgentKTableOfResultsColname.SETUP_SEED.value: os.path.basename(setup_path).split("_")[-1],
            }

            plan_path = Path(setup_path) / FileMap.SETUP_PLAN_JSON.value
            if not plan_path.exists():
                print(f"no plan {setup_path}")
                continue

            with open(plan_path, "r") as f:
                plan = json.load(f)

            should_check_final_unit_test = True

            for stage in tqdm(plan.keys(), desc="Scanning stages", leave=False):
                stage_label = stage_name_to_label(stage_name=stage)
                if stage_label is None or stage == "final_unit_test":
                    continue

                stage_entry = plan.get(stage)
                if not isinstance(stage_entry, dict):
                    setup_success_row[stage_label] = None
                    continue

                status_dict = stage_entry.get("status")
                if not isinstance(status_dict, dict):
                    setup_success_row[stage_label] = None
                    continue

                status_str = status_dict.get("status_str")
                setup_success_row[stage_label] = status_str
                if status_str not in get_setup_success_stages():
                    should_check_final_unit_test = False

            stage_label = stage_name_to_label(stage_name="final_unit_test")
            if should_check_final_unit_test:
                success = is_setup_pipeline_successful(Path(setup_path), is_tabular=is_tabular)
                setup_success_row[stage_label] = OutcomeName.SUCCESS.value if success else OutcomeName.FAILURE.value
            else:
                setup_success_row[stage_label] = OutcomeName.NOT_REACHED.value

            raw_setup_success_table_rows.append(setup_success_row)
        return raw_setup_success_table_rows


class TabularAgentKConfig(AgentKScaffoldConfig):
    _name: ClassVar[ExpName] = ExpName.AGENT_K_LAB_TAB

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, time_limit: int, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.tab_agent_k_handler(seed=seed, competition=competition, time_limit=time_limit)

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        return self.get_root_path(
            seed=seed, competition=competition, time_limit=self.time_limit, results_path_handler=results_path_handler
        )

    def should_run(self, competition: Competition) -> bool:
        return competition.is_fully_tabular()

    @property
    def fullname(self) -> str:
        return self._name.value + f" {self.time_limit}D"

    def get_exp_check_file(
            self, workspace: Path | None = None, seed: str | None = None, competition: Competition | None = None,
            results_path_handler: ExpPathHandler | None = None
    ) -> Path:
        """ Get the path to the file assessing that the experiment has been run properly (though it may have failed) """
        return self._get_workspace(
            workspace=workspace, seed=seed, competition=competition, results_path_handler=results_path_handler
        ) / self.checked_run_file

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        new_entries = defaultdict(list)
        progress_elements: ProgressElements = defaultdict(dict)

        comp_id = competition.competition_id

        workspace = self.root_path(seed=seed, competition=competition, results_path_handler=results_path_handler)

        progress = ProgressStatusNotStarted(seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=workspace)
        progress_elements[self.fullname][progress.to_key()] = progress

        if not workspace.exists():
            return {}, progress_elements

        assert workspace.is_dir()

        setup_workspaces = self.get_setup_paths(workspace=workspace)
        if len(setup_workspaces) == 0:
            progress = ProgressStatusNoSetupDir(
                seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=workspace
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        checked_run_abs_path = self.get_exp_check_file(workspace=workspace)

        setup_seed = max(map(lambda p: int(Path(p).name.split("_")[-1]), setup_workspaces))
        setup_workspace = workspace / f"seed_{setup_seed}"
        ramp_workspace_patterns = [setup_workspace / "ramp_kit_v*", setup_workspace / "ramp_kitv*"]
        ramp_workspaces = [
            p for ramp_workspace_pattern in ramp_workspace_patterns for p in glob.glob(str(ramp_workspace_pattern))
        ]
        if len(ramp_workspaces) == 0:
            if checked_run_abs_path.exists():
                progress = ProgressStatusFinishedFailure(
                    seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=workspace,
                    generated_submissions=False, message_path=checked_run_abs_path
                )
            else:
                progress = ProgressStatusNoRAMPDir(
                    seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=setup_workspace
                )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements
        if len(ramp_workspaces) > 1:
            progress = ProgressStatusTooManyRAMPDirs(
                seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=" ".join(ramp_workspaces)
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        ramp_workspace = Path(ramp_workspaces[0])
        model = ramp_workspace.name[len("ramp_kit_v"):]
        assert model in ["qwen2.5-72b_n0", "qwen2.5-72b-lmdeploy_n0"], ramp_workspace.name

        final_preds_dir = ramp_workspace / 'final_test_predictions'
        combinations = ["bagged_then_blended", "last_blend"]
        final_pred_files = [final_preds_dir / f"auto_v{model}_{combination}_030.csv" for combination in combinations]
        should_keep_running = False
        for p in final_pred_files:
            if not p.exists():
                progress = ProgressStatusRAMPNotOver(
                    seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=p
                )
                progress_elements[self.fullname][progress.to_key()] = progress
                should_keep_running = True
        if should_keep_running:
            if checked_run_abs_path.exists():
                progress = ProgressStatusFinishedFailure(
                    seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=workspace,
                    generated_submissions=False, message_path=checked_run_abs_path
                )
                progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        sub_suffixes = ["bagged_then_blended", "last_blend"]
        expected_sub_files = [final_preds_dir / f"auto_v{model}_{suffix}_030.csv" for suffix in sub_suffixes]
        score_report_paths = [Path(str(sub_file).replace(".csv", "_scores.json")) for sub_file in expected_sub_files]

        for expected_sub_file in expected_sub_files:
            if expected_sub_file.exists():
                continue
            progress = ProgressStatusMissingRAMPSubmission(
                seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=expected_sub_file
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        for score_report_path in score_report_paths:
            if score_report_path.exists():
                continue
            progress = ProgressStatusToSubmit(
                seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=score_report_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        results = {
            k: v for score_report_path in score_report_paths for k, v in json.load(open(score_report_path, "r")).items()
        }

        progress = ProgressStatusFinishedFailure(
            seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=final_preds_dir,
            generated_submissions=len(results) > 0,
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        for submission_file, score_report_path in zip(expected_sub_files, score_report_paths):
            results = json.load(open(score_report_path, "r"))
            assert list(results.keys()) == [str(submission_file)], (list(results.keys), submission_file)
            result = list(results.values())[0]

            combination = submission_file.name.replace(f"auto_v{model}_", "").replace("_030_scores.json", "")

            pub_score = result["public_score"]
            if pub_score is None:
                continue
            assert not np.isnan(pub_score)
            if competition.perc_private == 0:
                priv_score = pub_score
            else:
                priv_score = result["private_score"]

            new_entries[BaseTableOfResultsColname.PUB_SCORE.value].append(pub_score)
            new_entries[BaseTableOfResultsColname.PRIV_SCORE.value].append(priv_score)
            new_entries[BaseTableOfResultsColname.SUBMISSION_PATH.value].append(str(Path(submission_file)))

            new_entries[AgentKTableOfResultsColname.LLMS.value].append(model)
            new_entries[BaseTableOfResultsColname.SEED.value].append(seed)
            new_entries[AgentKTableOfResultsColname.SETUP_SEED.value].append(str(setup_seed))
            new_entries[AgentKTableOfResultsColname.TRIAL.value].append(combination)
            new_entries[AgentKTableOfResultsColname.DESCRIPTION.value].append(f"-")
            new_entries[AgentKTableOfResultsColname.USED_CI.value].append(np.nan)
            new_entries[AgentKTableOfResultsColname.TTA.value].append(np.nan)
            new_entries[AgentKTableOfResultsColname.IS_CV.value].append(np.nan)
            new_entries[AgentKTableOfResultsColname.IS_ALT_FORMAT.value].append(np.nan)

            new_entries[AgentKTableOfResultsColname.VERSION.value].append(self.fullname)
            new_entries[AgentKTableOfResultsColname.IS_TIME_CONSISTENT.value].append(
                self.is_time_consistent(competition=competition)
            )

        dest_cot_path = results_path_handler.tab_cot_filepath_handler(
            seed=seed, comp_id=comp_id, time_limit=self.time_limit
        )

        if dest_cot_path is None or not dest_cot_path.exists():
            progress = ProgressStatusMissingRAMPSummary(
                seed=seed, comp_id=comp_id, exp_fullname=self.fullname, workspace=ramp_workspace
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        return new_entries, progress_elements

    @staticmethod
    def is_time_consistent(competition: Competition) -> bool:
        if competition.leaderboard_end_date.is_rolling:
            return True
        return competition.leaderboard_end_date.end_date.date() >= max(RELEASE_DATES_TABULAR.values())

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        return ["0", "1", "2"]

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return seed in ["0", "2"]


class NonTabularAgentKConfig(AgentKScaffoldConfig):
    _name: ClassVar[ExpName] = ExpName.AGENT_K_LAB_NON_TAB
    is_ci: bool

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, time_limit: int, is_ci: bool, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.non_tab_agent_k_handler(
            seed=seed, competition=competition, time_limit=time_limit, is_ci=is_ci
        )

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        return self.get_root_path(
            seed=seed, competition=competition, time_limit=self.time_limit, is_ci=self.is_ci,
            results_path_handler=results_path_handler
        )

    def should_run(self, competition: Competition) -> bool:
        return self.aux_should_run(competition=competition, is_ci=self.is_ci)

    @staticmethod
    def aux_should_run(competition: Competition, is_ci: bool) -> bool:
        if competition.is_fully_tabular():
            return False
        if not is_ci:
            return True
        return competition.competition_type in CompetitionType.get_classification_types()

    @property
    def fullname(self) -> str:
        fullname = str(self._name.value)
        if self.is_ci:
            fullname += " (CI)"
        fullname += f" ({self.time_limit}D)"
        return fullname

    @staticmethod
    def get_selected_blend_submissions(blend_path: str) -> list:
        """
        Return the names of the submissions that were blended

        Args:
            blend_path: path to the blend workspace

        Returns:
            Names of the blended submissions (should be > 1)
        """
        blend_command_path = Path(blend_path) / FileMap.BLEND_COMMAND_TXT.value
        with open(blend_command_path, "r") as f:
            blend_command = f.read()

        submission_paths = blend_command.split("--submissions ")[-1].split("2>")[0].split()
        submission_names = set([os.path.basename(submission_path) for submission_path in submission_paths])

        return list(submission_names)

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        blend_trial_tag = "B"

        stuffs_to_write = []
        missing_models_release_dates = set()

        n_days_limit = self.time_limit + .2

        non_tta_submission_scores_json = "submission_*scores.json"
        tta_submission_scores_json = "tta-submission_*scores.json"
        cv_submission_scores_json = "cv-submission_*scores.json"

        submission_patterns = ["cv-submission*.csv", "tta*submission*.csv", "submission*.csv"]

        class SubmissionPattern(NamedTuple):
            is_tta: bool
            is_cv: bool
            pattern: str

        agent_k_submission_score_json_patterns: list[SubmissionPattern] = [
            SubmissionPattern(is_tta=False, is_cv=False, pattern=non_tta_submission_scores_json),
            SubmissionPattern(is_tta=True, is_cv=False, pattern=tta_submission_scores_json),
            SubmissionPattern(is_tta=False, is_cv=True, pattern=cv_submission_scores_json)
        ]

        progress_elements: ProgressElements = defaultdict(dict)
        tracking_messages = []

        new_entries = defaultdict(list)

        comp_name = competition.comp_name

        results_path = self.root_path(competition=competition, seed=seed, results_path_handler=results_path_handler)

        progress = ProgressStatusNotStarted(
            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=results_path
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        agent_k_setup_log = os.path.join(results_path, "logs/setup/seed_0/start.log")
        if not os.path.exists(agent_k_setup_log):
            return {}, progress_elements

        checked_file = os.path.join(results_path, self.checked_run_file)
        main_pipeline_pattern = os.path.join(results_path, "seed_*", "main_pipeline")
        main_pipeline_paths = glob.glob(main_pipeline_pattern)
        if len(main_pipeline_paths) > 1:
            progress = ProgressStatusToManyMainPipelines(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=main_pipeline_pattern
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements
        if len(main_pipeline_paths) == 0:
            if os.path.exists(checked_file):
                progress = ProgressStatusFinishedFailure(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=results_path,
                    generated_submissions=False
                )
                progress_elements[self.fullname][progress.to_key()] = progress
            else:
                progress = ProgressStatusNoMainPipelines(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                    workspace=main_pipeline_pattern
                )
                progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        main_pipeline_path = main_pipeline_paths[0]
        setup_seed = os.path.basename(os.path.dirname(main_pipeline_path)).split("_")[-1]
        for submission_path in glob.glob(os.path.join(main_pipeline_path, "submissions", "*")):
            if len(glob.glob(os.path.join(submission_path, non_tta_submission_scores_json))) > 0:
                continue
            for submission_pattern in submission_patterns:
                if len(glob.glob(os.path.join(submission_path, submission_pattern))) == 0:
                    continue
                progress = ProgressStatusToSubmit(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                    workspace=submission_path
                )
                progress_elements[self.fullname][progress.to_key()] = progress
                return {}, progress_elements

        submission_scores_pattern = os.path.join(main_pipeline_path, "submissions", "*", non_tta_submission_scores_json)
        submission_scores_paths = glob.glob(submission_scores_pattern)
        submission_trials_workspaces = sorted(
            set(
                [os.path.dirname(scores_path) for scores_path in submission_scores_paths if
                 "blend" not in os.path.basename(os.path.dirname(scores_path))
                 ]
            )
        )
        trials_numbers = list(map(str, range(len(submission_trials_workspaces))))

        submission_blend_workspaces = glob.glob(os.path.join(main_pipeline_path, "submissions", "blend_*"))
        if len(submission_blend_workspaces) > 1:
            raise RuntimeError("More than one blend")

        if len(submission_blend_workspaces) == 1:
            submission_blend_workspace = submission_blend_workspaces[0]
            submission_blend_score_pattern = os.path.join(submission_blend_workspace, non_tta_submission_scores_json)
            submission_blend_score_paths = glob.glob(submission_blend_score_pattern)

            if len(submission_blend_score_paths) > 0:
                trials_numbers.append(blend_trial_tag)
                submission_trials_workspaces.append(submission_blend_workspace)

        if not os.path.exists(checked_file):
            comp_should_be_over, delta = self.check_comp_is_over(
                log_path=agent_k_setup_log, n_days_limit=n_days_limit
            )
            if not comp_should_be_over:
                progress = ProgressStatusRunning(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                    workspace=results_path
                )
                progress_elements[self.fullname][progress.to_key()] = progress
                return {}, progress_elements

            if len(submission_scores_paths) == 0:
                progress_status_class = ProgressStatusToCheckProbablyFailure
            else:
                progress_status_class = ProgressStatusToCheckProbablySuccess

            progress = progress_status_class(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=results_path,
                delta=delta
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            stuffs_to_write.append(f"sudo touch {os.path.join(results_path, self.checked_run_file)}")
            return {}, progress_elements

        info_path = os.path.join(main_pipeline_path, "info.pkl")
        info = load_w_pickle(info_path)
        version = info.get("ds_task_version").replace("v", "")
        assert isinstance(version, str), (version, main_pipeline_path)
        if self.is_ci:
            version += "-CI"

        progress = ProgressStatusFinishedFailure(
            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=results_path,
            generated_submissions=False
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        time_consistent_submission = {}

        for trial, submission_workspace in zip(trials_numbers, submission_trials_workspaces):
            if trial == blend_trial_tag:
                is_time_consistent = True
                selected_submissions = NonTabularAgentKConfig.get_selected_blend_submissions(
                    blend_path=submission_workspace
                )
                if len(selected_submissions) == 0:
                    progress = ProgressStatusBlendComponentMissing(
                        seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                        workspace=submission_workspace
                    )
                    progress_elements[self.fullname][progress.to_key()] = progress
                    return {}, progress_elements
                if len(selected_submissions) == 1:
                    tracking_message = TrackingMessage(
                        tracking_type=ProgressStatusType.SINGLE_BLEND_COMPONENT,
                        header=f"(SEED {seed} {'CI' if self.is_ci else ''})",
                        color=StringColors.MAGENTA,
                        body=f"{comp_name} -- {submission_workspace} -- Only {selected_submissions[0]} in the blend"
                    )
                    tracking_messages.append(tracking_message)

                for selected_submission in selected_submissions:
                    if selected_submission not in time_consistent_submission:
                        progress = ProgressStatusBlendComponentMissing(
                            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                            workspace=submission_workspace,
                            extra_hover_text=f"Failed to check submission: {selected_submission}"
                        )
                        progress_elements[self.fullname][progress.to_key()] = progress
                        return {}, progress_elements
                    if not time_consistent_submission[selected_submission]:
                        is_time_consistent = False  # selected sub is time inconsistent -> blend is time inconsistent
                        break
            else:
                errors = check_time_consistent_workspace(
                    competition_id=competition.competition_id,
                    workspace_path=Path(submission_workspace),
                    skip_checked=True,
                    write_json=FileMap.TOO_RECENT_CHECK.value
                )
                if len(errors) > 0:
                    for error in errors:
                        if "TooRecentException" in errors:
                            continue
                        if 'MissingEntry' in errors:
                            missing_models_release_dates.update(errors['MissingEntry'])
                        progress.tracking_messages.append(
                            TrackingMessage(
                                tracking_type=ProgressStatusType.RELEASE_DATE_ERROR,
                                header=f"(SEED {seed} {'CI' if self.is_ci else ''})",
                                color=StringColors.BLUE,
                                body=f"{comp_name} -- {submission_workspace} -- {error}: {errors[error]}"
                            )
                        )
                    is_time_consistent = False
                else:
                    is_time_consistent = True
                    for input_data_type in [DataType.IMG, DataType.TXT]:
                        too_recent_check_path = os.path.join(
                            submission_workspace, input_data_type.value + "_" + FileMap.TOO_RECENT_CHECK.value
                        )
                        if input_data_type in competition.input_types:
                            check_release_dates = json.load(open(too_recent_check_path, "r"))
                            is_time_consistent &= check_release_dates["is_success"]

                time_consistent_submission[os.path.basename(submission_workspace)] = is_time_consistent

            for sub_scores_json_pattern in agent_k_submission_score_json_patterns:
                # TODO: raise message if tta / cv is missing
                submission_scores_paths = glob.glob(os.path.join(submission_workspace, sub_scores_json_pattern.pattern))

                for submission_scores_path in submission_scores_paths:
                    try:
                        results = json.load(open(submission_scores_path, "r"))
                    except JSONDecodeError:
                        print(submission_scores_paths, submission_scores_path)
                        continue

                    for submission_file, result in results.items():
                        pub_score = result["public_score"]
                        if pub_score is None:
                            continue
                        assert not np.isnan(pub_score)
                        if competition.perc_private == 0:
                            priv_score = pub_score
                        else:
                            priv_score = result["private_score"]

                        submission_file = results_path_handler.process_submission_filepath(
                            submission_filepath=submission_file
                        )
                        if not os.path.exists(submission_file):
                            progress.tracking_messages.append(
                                TrackingMessage(
                                    tracking_type=ProgressStatusType.SUBMISSION_NOT_FOUND,
                                    header=f"(SEED {seed} {'CI' if self.is_ci else ''})",
                                    color=StringColors.BLUE,
                                    body=f"{comp_name} -- {submission_file}"
                                )
                            )

                        new_entries[BaseTableOfResultsColname.PUB_SCORE.value].append(pub_score)
                        new_entries[BaseTableOfResultsColname.PRIV_SCORE.value].append(priv_score)
                        new_entries[BaseTableOfResultsColname.SUBMISSION_PATH.value].append(str(Path(submission_file)))

                        new_entries[AgentKTableOfResultsColname.LLMS.value].append("(TO_CHECK)")
                        new_entries[BaseTableOfResultsColname.SEED.value].append(seed)
                        new_entries[AgentKTableOfResultsColname.SETUP_SEED.value].append(setup_seed)
                        new_entries[AgentKTableOfResultsColname.TRIAL.value].append(str(trial))
                        new_entries[AgentKTableOfResultsColname.DESCRIPTION.value].append(f"-")

                        new_entries[AgentKTableOfResultsColname.USED_CI.value].append(self.is_ci)
                        new_entries[AgentKTableOfResultsColname.TTA.value].append(sub_scores_json_pattern.is_tta)
                        new_entries[AgentKTableOfResultsColname.IS_CV.value].append(sub_scores_json_pattern.is_cv)
                        is_alt_format = "_alt.csv" in os.path.basename(submission_file)
                        new_entries[AgentKTableOfResultsColname.IS_ALT_FORMAT.value].append(is_alt_format)

                        new_entries[AgentKTableOfResultsColname.VERSION.value].append(version)
                        new_entries[AgentKTableOfResultsColname.IS_TIME_CONSISTENT.value].append(is_time_consistent)

        progress.generated_submissions = len(new_entries) > 0
        progress.tracking_messages.extend(tracking_messages)

        if len(stuffs_to_write) > 0:
            print("\n\n--- To execute potentially | BE AWARE OF WHAT YOU'RE DOING ---")
            for s in sorted(stuffs_to_write):
                print(s)

        if len(missing_models_release_dates) > 0:
            print("\n\n--- To execute potentially | BE AWARE OF WHAT YOU'RE DOING ---")
            for m in missing_models_release_dates:
                print(f"\t- {m}")

        return new_entries, progress_elements

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        if self.is_ci:
            return ["0", "1"]
        return ["0", "1", "2"]

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        """
        Whether a given competition and seed should be considered to compute final perf

        Args:
            competition: competition instance
            seed: run seed

        Returns:
             True if the run should be taken into account for the final results, False otherwise
        """
        if self.aux_should_run(competition=competition, is_ci=True):
            seeds = ["1"]
        else:
            seeds = ["0", "1"]
        return seed in seeds


@runtime_checkable
class GetCoTPathCallable(Protocol):  # define for the sake of typing
    def __call__(self, competition: Competition, seed: str, results_path_handler: ExpPathHandler) -> tuple[
        str | None, bool]: ...


class AgentKDSCoT(BaseModel):
    from_ci: bool
    time_limit: int
    get_cot_path: GetCoTPathCallable

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TreeGenConfig(ExpConfig, ABC):
    time_limit: int
    draft_only: bool = False  # whether to only extract results of draft nodes.
    intermediate_best_only: bool = False
    exp_llm: ExpLLM  # LLM used

    @model_validator(mode='after')
    def check_flags(self) -> TreeGenConfig:
        if self.draft_only and self.intermediate_best_only:
            raise ValueError("draft_only and intermediate_best_only cannot both be True.")
        return self

    @staticmethod
    def calculate_hf_retry_time(hf_time_file: Path) -> float:
        """
        Calculate the time wastage due to LLM retries
        Args:
            hf_time_file: Log file name contains LLM retries information

        Returns: Total time wasted for retries

        """
        hf_retry_time = 0
        if os.path.exists(hf_time_file):
            with open(hf_time_file, 'r') as f:
                time_wasted = json.load(f)
                for key in time_wasted:
                    hf_retry_time += float(time_wasted[key])

        return hf_retry_time

    @staticmethod
    def get_job_budget(log_path: Path, config_run_time: float) -> float:
        """
        Calculate the budge of the job run
        Args:
            log_path: Path to the run log
            config_run_time: Time recorded in the config after run

        Returns: Total time budget of the run

        """
        # TODO~ save this computation in some file so there is no need to recompute each time
        with open(log_path, 'r') as f:
            log_data = f.readlines()
        previous_run_time = 0
        monitored_run_time = 0
        is_resume_successful = False
        for line in log_data[1:]:
            if "Previous run time" in line:
                previous_run_time = int(line.split(' ')[-1])
                is_resume_successful = False
            elif "Agent is parsing execution results" in line and not is_resume_successful:
                is_resume_successful = True
                monitored_run_time = previous_run_time

        total_time_span = config_run_time + monitored_run_time
        model_retry_time_file = Path(log_path).parent.parent.parent / "hf_retries.json"

        llm_time_waste = TreeGenConfig.calculate_hf_retry_time(hf_time_file=model_retry_time_file)
        total_time_span -= llm_time_waste

        return total_time_span / 3600 / 24

    def aux_get_result(
            self, seed: str, competition: Competition, cot_start: AgentKDSCoT | None,
            results_path_handler: ExpPathHandler
    ) -> tuple[dict[str, list], ProgressElements]:
        stuffs_to_write = []
        n_days_limit = self.time_limit + .2
        new_entries = defaultdict(list)

        if self.draft_only:
            head = "draft"
        elif self.intermediate_best_only:
            head = "intermediate_best"
        else:
            head = "best"

        progress_elements: ProgressElements = defaultdict(dict)

        results_path = self.root_path(seed=seed, competition=competition, results_path_handler=results_path_handler)
        results_dirs = glob.glob(str(results_path) + "/*/")

        progress = ProgressStatusNotStarted(
            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=results_path
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        if cot_start is not None and cot_start.from_ci and competition.competition_type not in CompetitionType.get_classification_types():
            if len(results_dirs) > 0:
                progress = ProgressStatusCIForRegression(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=results_path
                )
                progress_elements[self.fullname][progress.to_key()] = progress
                stuffs_to_write.append(f"rm -r {results_path}")
            return {}, progress_elements

        if cot_start is not None:
            cot_path, is_cot_not_empty = cot_start.get_cot_path(
                seed=seed, competition=competition, results_path_handler=results_path_handler
            )
        else:
            cot_path, is_cot_not_empty = None, False
        cot_source_is_ci = np.nan if cot_start is None else cot_start.from_ci

        if len(results_dirs) == 0:  # no workspace for the current exp
            if cot_start is not None and cot_path is None:  # is it blocked because there is no CoT?
                progress = ProgressStatusNoCoTToStartFrom(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                    workspace=results_path
                )
                progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        if len(results_dirs) > 1:
            progress = ProgressStatusTooManyFolders(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=results_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress

            for res_dir in sorted(results_dirs)[:-1]:
                if os.getenv("DISABLE_ASSERT_CHECKS") not in ["True", "true", "1"]:
                    assert len(glob.glob(res_dir + f"/{head}_submissions/*")) == 0, (
                        results_dirs, glob.glob(res_dir + f"/{head}_submissions/*")
                    )
                stuffs_to_write.append(f"sudo rm -r {res_dir}")
            return {}, progress_elements

        if cot_start is not None and cot_path is None:  # is it blocked because there is no CoT?
            progress = ProgressStatusRanDespiteMissingCoT(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=results_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        aide_workspace = results_dirs[0]

        aide_run_log_patterns = [
            f"{aide_workspace}/{log_folder}/aide.verbose.log" for log_folder in ["/run_logs/*/", "/logs/"]
        ]
        aide_run_logs = [p for log_pattern in aide_run_log_patterns for p in glob.glob(log_pattern)]
        if os.getenv("DISABLE_ASSERT_CHECKS") not in ["True", "true", "1"]:
            assert len(aide_run_logs) == 1, (aide_run_log_patterns, aide_run_logs)
        elif len(aide_run_logs) != 1:
            assert len(aide_run_logs) == 0, aide_run_logs
            progress = ProgressStatusMissingVerboseLog(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=results_path
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements
        aide_run_log = aide_run_logs[0]
        config_file = os.path.join(os.path.dirname(aide_run_log), "config.yaml")
        if not os.path.exists(config_file):
            progress = ProgressStatusMissingReactExplConfig(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=results_path,
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        workspace_name = Path(aide_workspace).stem
        workspace_year = workspace_name.split("-")[0]
        # Skip budget mismatch check for old workspaces
        if int(workspace_year) > 2024:
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.Loader)
            config_time_limit = config["agent"]["time_limit"]
            log_file = Path(aide_run_log).parent / "aide.log"

            job_run_time = TreeGenConfig.get_job_budget(log_path=log_file, config_run_time=config_time_limit)
            if abs(self.time_limit - job_run_time) > (1 / 48):
                progress = ProgressStatusMissingTimeLimitMismatchConfig(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                    workspace=config_file, extra_hover_text=f"Expected: {self.time_limit}, ran with {job_run_time:.1f}"
                )
                progress_elements[self.fullname][progress.to_key()] = progress
                return {}, progress_elements

        comp_should_be_over, delta = self.check_comp_is_over(log_path=aide_run_log, n_days_limit=n_days_limit)
        if not comp_should_be_over:
            progress = ProgressStatusRunning(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=aide_workspace
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        best_submission_dir = aide_workspace + f"/best_submissions"
        best_submissions_paths = glob.glob(f"{best_submission_dir}/submission_*.csv")
        submission_dir = aide_workspace + f"/{head}_submissions"
        submissions_paths = glob.glob(f"{submission_dir}/submission_*.csv")
        best_submissions_scores_path = best_submission_dir + "/submission_scores.json"
        submission_scores_path = f"{submission_dir}/submission_scores.json"

        if not os.path.exists(os.path.join(aide_workspace, self.checked_run_file)):
            if len(submissions_paths) == 0:
                progress_status_class = ProgressStatusToCheckProbablyFailure
            else:
                progress_status_class = ProgressStatusToCheckProbablySuccess
            progress = progress_status_class(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                workspace=aide_workspace, delta=delta
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            stuffs_to_write.append(f"sudo touch {os.path.join(aide_workspace, self.checked_run_file)}")
            return {}, progress_elements

        progress = ProgressStatusFinishedFailure(
            seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=aide_workspace,
            generated_submissions=False
        )
        progress_elements[self.fullname][progress.to_key()] = progress

        if len(submissions_paths) == 0 and len(best_submissions_paths) == 0:
            return {}, progress_elements

        if not os.path.exists(submission_scores_path) and len(submissions_paths) > 0:
            progress = ProgressStatusToSubmit(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=aide_workspace,
                extra_hover_text=f"No file: {submission_scores_path}"
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements
        elif not os.path.exists(best_submissions_scores_path):
            progress = ProgressStatusToSubmit(
                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname, workspace=aide_workspace,
                extra_hover_text=f"No file: {best_submissions_scores_path}"
            )
            progress_elements[self.fullname][progress.to_key()] = progress
            return {}, progress_elements

        def get_results(sub_score_path_json: str) -> dict[str, ...]:
            if not os.path.exists(sub_score_path_json):
                return {}
            results_ = json.load(open(sub_score_path_json, "r"))
            clean_results = {}
            for submission_file_, result_ in results_.items():
                submission_file_ = os.path.dirname(sub_score_path_json) + "/" + os.path.basename(submission_file_)
                if not os.path.exists(submission_file_):
                    continue
                clean_results[submission_file_] = result_
            return clean_results

        results = get_results(sub_score_path_json=submission_scores_path)
        results_submission_basenames = {os.path.basename(submission_file) for submission_file in results}
        if self.intermediate_best_only:
            best_results = get_results(sub_score_path_json=best_submissions_scores_path)
            for submission_file in best_results:
                if os.path.basename(submission_file) not in results_submission_basenames:
                    results[submission_file] = best_results[submission_file]

        assert len(results) > 0, results

        for submission_path in submissions_paths:
            if submission_path not in results:
                progress = ProgressStatusToSubmit(
                    seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                    workspace=submission_path
                )
                progress_elements[self.fullname][progress.to_key()] = progress
                return {}, progress_elements

        if self.intermediate_best_only:
            enter_exit_remaining_time_json = os.path.join(submission_dir, "enter_exit_remaining_time.json")
            if not os.path.exists(enter_exit_remaining_time_json):
                # map the best nodes to the node replacing it as best intermediate
                N_BEST_NODES = 4
                metric_order = None

                old_best_to_new_best = {}
                current_best_nodes = []
                try:
                    journal = json.load(open(Path(aide_run_log).parent / "journal.json"))
                except json.decoder.JSONDecodeError:
                    progress = ProgressStatusCannotReadJournal(
                        seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                        workspace=results_path
                    )
                    progress_elements[self.fullname][progress.to_key()] = progress
                    return {}, progress_elements
                for i, node in enumerate(journal["nodes"]):
                    if node["is_buggy"]:
                        continue

                    val = node["metric"]["value"]
                    is_max = node["metric"]["maximize"]
                    if not is_max:
                        val = - val

                    if metric_order is None:
                        metric_order = is_max
                    else:
                        if metric_order != is_max:
                            break

                    node_id = "submission_" + node['id'] + ".csv"

                    if len(current_best_nodes) < N_BEST_NODES:
                        heapq.heappush(current_best_nodes, (val, node_id, i))
                        old_best_to_new_best[node_id] = None
                        continue

                    new_best = val > current_best_nodes[0][0]

                    if not new_best:
                        continue

                    v, old_best, _ = heapq.heappop(current_best_nodes)
                    old_best_to_new_best[old_best] = node_id
                    old_best_to_new_best[node_id] = None
                    heapq.heappush(current_best_nodes, (val, node_id, i))

                remaining_time_json = os.path.join(submission_dir, "remaining_time.json")
                ignore_remaining_time_json = os.getenv("IGNORE_REMAINING_TIME_JSON", False)
                if not ignore_remaining_time_json and os.path.exists(remaining_time_json):
                    best_nodes_to_enter_remain_time = json.load(open(remaining_time_json, "r"))
                else:
                    best_nodes_to_enter_remain_time = {}

                    with open(Path(aide_run_log).parent / "aide.log", "r") as f:
                        log_lines = f.readlines()

                    first_timestamp = None

                    for line in log_lines:
                        if not line.strip():
                            continue
                        try:
                            timestamp_part = line.split(']')[0].strip('[')
                            timestamp = timestamp_part.strip()
                            timestamp = datetime.strptime(timestamp.replace(',', '.'), '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            continue
                        if first_timestamp is None:
                            first_timestamp = timestamp

                        if "added to best nodes" in line:
                            if "is added" in line:
                                tag = " is added"
                            elif "is not added" in line:
                                tag = " is not added"
                            else:
                                raise RuntimeError(line)
                            node_info_part = line.split('Node ')[1].split(tag)[0]
                            node_id = node_info_part.strip()
                            node_id = "submission_" + node_id + ".csv"
                            if node_id not in old_best_to_new_best:
                                continue
                            time_since_start = (timestamp - first_timestamp).total_seconds()
                            best_nodes_to_enter_remain_time[node_id] = self.time_limit * 3600 * 24 - time_since_start
                    with open(remaining_time_json, "w") as f:
                        json.dump(best_nodes_to_enter_remain_time, f)

                best_nodes_to_enter_exit_remain_time: dict[str, tuple[float, float]] = {}
                for node_id, enter_time in best_nodes_to_enter_remain_time.items():
                    new_best_node = old_best_to_new_best.get(node_id, None)
                    if new_best_node is None:
                        exit_time = min(best_nodes_to_enter_remain_time.values()) - 1
                    else:
                        if os.getenv("DISABLE_ASSERT_CHECKS") not in ["True", "true", "1"]:
                            assert new_best_node in best_nodes_to_enter_remain_time, (aide_run_log, new_best_node)
                        elif new_best_node not in best_nodes_to_enter_remain_time:
                            extra_hover_text = f"new_best_node not in best_nodes_to_enter_remain_time -- "
                            extra_hover_text += new_best_node
                            for node, remaining_time in best_nodes_to_enter_remain_time.items():
                                extra_hover_text += f" -- {node}: {remaining_time}"
                            progress = ProgressStatusIntermediateNodeProblem(
                                seed=seed, comp_id=competition.competition_id, exp_fullname=self.fullname,
                                workspace=aide_run_log,
                                extra_hover_text=extra_hover_text
                            )
                            progress_elements[self.fullname][progress.to_key()] = progress
                            return {}, progress_elements
                        exit_time = best_nodes_to_enter_remain_time[new_best_node]
                    best_nodes_to_enter_exit_remain_time[node_id] = (enter_time, exit_time)

                with open(enter_exit_remaining_time_json, "w") as f:
                    json.dump(best_nodes_to_enter_exit_remain_time, f)
            else:
                best_nodes_to_enter_exit_remain_time = json.load(open(enter_exit_remaining_time_json, "r"))

        for submission_file, result in results.items():
            pub_score = result["public_score"]
            if pub_score is None:
                continue
            assert not np.isnan(pub_score)
            if competition.perc_private == 0:
                priv_score = pub_score
            else:
                priv_score = result["private_score"]

            if not os.path.exists(submission_file):
                progress.tracking_messages.append(
                    TrackingMessage(
                        tracking_type=ProgressStatusType.SUBMISSION_NOT_FOUND,
                        header=f"(SEED {seed} {self.fullname})",
                        color=StringColors.CYAN,
                        body=f"{competition.comp_name} -- {submission_file}"
                    )
                )

            new_entries[BaseTableOfResultsColname.PUB_SCORE.value].append(pub_score)
            new_entries[BaseTableOfResultsColname.PRIV_SCORE.value].append(priv_score)
            new_entries[BaseTableOfResultsColname.SUBMISSION_PATH.value].append(str(Path(submission_file)))

            new_entries[AIDETableOfResultsColname.COT_SOURCE.value].append(cot_path)
            new_entries[AIDETableOfResultsColname.COT_SOURCE_IS_CI.value].append(cot_source_is_ci)
            new_entries[AIDETableOfResultsColname.IS_COT_NOT_EMPTY.value].append(is_cot_not_empty)
            new_entries[AIDETableOfResultsColname.RAG_SOURCE_VERIONS.value].append(np.nan)
            new_entries[AIDETableOfResultsColname.RUNTIME_BUDGET.value].append(self.time_limit)
            new_entries[BaseTableOfResultsColname.SEED.value].append(seed)
            if self.intermediate_best_only:
                try:
                    enter_time, exit_time = best_nodes_to_enter_exit_remain_time[os.path.basename(submission_file)]
                except KeyError as e:
                    print("Submission file", submission_file)
                    print("Best nodes to enter exit remain time", best_nodes_to_enter_exit_remain_time)
                    raise e
            else:
                enter_time, exit_time = np.nan, np.nan
            new_entries[AIDETableOfResultsColname.ENTER_REMAINING_TIME.value].append(enter_time)
            new_entries[AIDETableOfResultsColname.EXIT_REMAINING_TIME.value].append(exit_time)

        progress.generated_submissions = len(new_entries) > 0

        if len(stuffs_to_write) > 0:
            print("\n\n--- To execute potentially | BE AWARE OF WHAT YOU'RE DOING ---")
            for s in sorted(stuffs_to_write):
                print(s)

        return new_entries, progress_elements


class TreeGenTabConfig(TreeGenConfig, ABC):
    def should_run(self, competition: Competition) -> bool:
        return competition.is_fully_tabular()


class VanillaTreeGenConfig(TreeGenConfig, ABC):

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, time_limit: int, exp_llm: ExpLLM, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.react_agent_handler(
            seed=seed, competition=competition, time_limit=time_limit, exp_llm=exp_llm
        )

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        return self.get_root_path(
            seed=seed, competition=competition, time_limit=self.time_limit, exp_llm=self.exp_llm,
            results_path_handler=results_path_handler
        )

    @property
    def fullname(self) -> str:
        fullname = f"{self._name.value}  ({self.exp_llm.value}) ({self.time_limit}D)"
        if self.draft_only:
            fullname += " [DRAFT]"
        if self.intermediate_best_only:
            fullname += " [INTERMEDIATE]"
        return fullname

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        return self.aux_get_result(
            seed=seed, competition=competition, cot_start=None, results_path_handler=results_path_handler
        )

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        if self.exp_llm == ExpLLM.LLM_PLAYGROUND_QWEN2_5_72B:
            return ["0", "1"]
        if self.exp_llm == ExpLLM.DEEPSEEK_R1:
            return ["0", "2"]
        raise ValueError(self.exp_llm)

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return seed in self.get_seeds_to_include_in_tracking()


class VanillaTreeGenNonTabConfig(VanillaTreeGenConfig):
    _name: ClassVar[ExpName] = ExpName.VANILLA_TREE_GEN_NON_TAB_CONFIG

    def should_run(self, competition: Competition) -> bool:
        return not competition.is_fully_tabular()


class VanillaTreeGenTabConfig(VanillaTreeGenConfig, TreeGenTabConfig):
    _name: ClassVar[ExpName] = ExpName.VANILLA_TREE_GEN_TAB_CONFIG


class TreeGenFromRAGConfig(TreeGenConfig, ABC):
    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, time_limit: float, exp_llm: ExpLLM,
            results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.react_agent_from_rag_handler(
            seed=seed, competition=competition, time_limit=time_limit, exp_llm=exp_llm
        )

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path:
        return self.get_root_path(
            seed=seed, competition=competition, time_limit=self.time_limit, exp_llm=self.exp_llm,
            results_path_handler=results_path_handler
        )

    @property
    def fullname(self) -> str:
        fullname = f"{self._name.value} ({self.exp_llm.value}) ({self.time_limit}D)"
        if self.draft_only:
            fullname += " [DRAFT]"
        elif self.intermediate_best_only:
            fullname += " [INTERMEDIATE]"
        return fullname

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        return self.aux_get_result(
            seed=seed, competition=competition, cot_start=None, results_path_handler=results_path_handler
        )

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        if self.exp_llm == ExpLLM.LLM_PLAYGROUND_QWEN2_5_72B:
            return ["1", "2"]
        raise ValueError(self.exp_llm)

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return seed in self.get_seeds_to_include_in_tracking()


class TreeGenFromRAGNonTabConfig(TreeGenFromRAGConfig):
    _name: ClassVar[ExpName] = ExpName.TREE_GEN_FROM_RAG_NON_TAB

    def should_run(self, competition: Competition) -> bool:
        return not competition.is_fully_tabular()


class TreeGenFromRAGTabConfig(TreeGenFromRAGConfig, TreeGenTabConfig):
    _name: ClassVar[ExpName] = ExpName.TREE_GEN_FROM_RAG_TAB


class AgentKWarmTreeGenConfig(TreeGenConfig, ABC):
    cot_time_limit: int

    @abstractmethod
    def get_scaffold_exp_config(self) -> AgentKScaffoldConfig:
        pass

    def get_seeds_to_include_in_tracking(self) -> list[str]:
        return self.get_scaffold_exp_config().get_seeds_to_include_in_tracking()

    def is_comp_seed_to_include_in_res(self, competition: Competition, seed: str) -> bool:
        return self.get_scaffold_exp_config().is_comp_seed_to_include_in_res(competition=competition, seed=seed)


class AgentKWarmTreeGenNonTabConfig(AgentKWarmTreeGenConfig):
    _name: ClassVar[ExpName] = ExpName.AGENT_K_TREE_GEN_NON_TAB
    from_ci_cot: bool

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, time_limit: int, from_ci_cot: bool,
            cot_time_limit: int, exp_llm: ExpLLM, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.non_tab_react_agent_from_cot_handler(
            seed=seed, competition=competition, time_limit=time_limit, from_ci_cot=from_ci_cot,
            cot_time_limit=cot_time_limit, exp_llm=exp_llm
        )

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        return self.get_root_path(
            seed=seed, competition=competition, time_limit=self.time_limit, from_ci_cot=self.from_ci_cot,
            cot_time_limit=self.cot_time_limit, exp_llm=self.exp_llm, results_path_handler=results_path_handler
        )

    @property
    def fullname(self) -> str:
        fullname: str = f"Lab {self.cot_time_limit}D"
        if self.from_ci_cot:
            fullname += " (from CI)"
        fullname += f" + {self._name.value} ({self.exp_llm.value}) ({self.time_limit}D)"
        if self.draft_only:
            fullname += " [DRAFT]"
        if self.intermediate_best_only:
            fullname += " [INTERMEDIATE]"
        return fullname

    def get_scaffold_exp_config(self) -> AgentKScaffoldConfig:
        return NonTabularAgentKConfig(time_limit=self.cot_time_limit, is_ci=self.from_ci_cot)

    def get_cot_path(self, competition: Competition, seed: str, results_path_handler: ExpPathHandler) -> tuple[
        str | None, bool]:
        """ Get the path of the agent K workspace that was used to generate the dsCoT

        Args:
            competition: competition
            seed: seed
            results_path_handler: Handler of results paths
        Returns:
            path to CoT run and whether the cot is empty
        """
        lab_config = self.get_scaffold_exp_config()
        _, track = lab_config.get_results_seed_comp(
            seed=seed, competition=competition, results_path_handler=results_path_handler
        )
        track = track[lab_config.fullname][ProgressStatusKey(seed=seed, competition_id=competition.competition_id)]

        cot_path = str(
            lab_config.root_path(competition=competition, seed=seed, results_path_handler=results_path_handler)
        )
        if isinstance(track, ProgressStatusFinishedSuccess):
            return cot_path, True
        if isinstance(track, ProgressStatusFinishedFailure):
            return cot_path, track.generated_submissions

        return None, False

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        pd.DataFrame | None, ProgressElements]:
        cot = AgentKDSCoT(from_ci=self.from_ci_cot, time_limit=self.cot_time_limit, get_cot_path=self.get_cot_path)
        return self.aux_get_result(
            seed=seed, competition=competition, cot_start=cot, results_path_handler=results_path_handler
        )

    def should_run(self, competition: Competition) -> bool:
        return self.get_scaffold_exp_config().should_run(competition=competition)


class AgentKWarmTreeGenTabConfig(AgentKWarmTreeGenConfig, TreeGenTabConfig):
    _name: ClassVar[ExpName] = ExpName.AGENT_K_TREE_GEN_TAB

    @staticmethod
    def get_root_path(
            seed: str, competition: Competition, time_limit: int, cot_time_limit: int,
            exp_llm: ExpLLM, results_path_handler: ExpPathHandler
    ) -> Path | None:
        return results_path_handler.tab_react_agent_from_cot_handler(
            seed=seed, competition=competition, time_limit=time_limit, cot_time_limit=cot_time_limit, exp_llm=exp_llm
        )

    def root_path(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> Path | None:
        return self.get_root_path(
            seed=seed, competition=competition, time_limit=self.time_limit, cot_time_limit=self.cot_time_limit,
            exp_llm=self.exp_llm, results_path_handler=results_path_handler
        )

    @property
    def fullname(self) -> str:
        fullname: str = f"Lab {self.cot_time_limit}D"
        fullname += f" + {self._name.value} ({self.exp_llm.value}) ({self.time_limit}D)"
        if self.draft_only:
            fullname += " [DRAFT]"
        elif self.intermediate_best_only:
            fullname += " [INTERMEDIATE]"
        return fullname

    def get_scaffold_exp_config(self) -> AgentKScaffoldConfig:
        return TabularAgentKConfig(time_limit=self.cot_time_limit)

    def get_cot_path(self, competition: Competition, seed: str, results_path_handler: ExpPathHandler) -> tuple[
        str | None, bool]:
        """ Get the path of the agent K workspace that was used to generate the dsCoT, also return whether the cot is
        empty or not

        Args:
            competition: competition
            seed: seed
            results_path_handler: Handler of results paths
        """
        scaffold_exp_config = self.get_scaffold_exp_config()
        assert isinstance(scaffold_exp_config, TabularAgentKConfig)
        cot_path = results_path_handler.tab_cot_filepath_handler(
            seed=seed, comp_id=competition.competition_id, time_limit=self.cot_time_limit
        )
        if cot_path.exists():
            return str(cot_path), True

        _, scaffold_track = scaffold_exp_config.get_results_seed_comp(
            seed=seed, competition=competition, results_path_handler=results_path_handler
        )
        progress_key = ProgressStatusKey(seed=seed, competition_id=competition.competition_id)
        scaffold_track = scaffold_track[scaffold_exp_config.fullname][progress_key]

        if isinstance(scaffold_track, ProgressStatusFinishedFailure):
            cot_path = str(
                scaffold_exp_config.root_path(
                    seed=seed, competition=competition, results_path_handler=results_path_handler
                )
            )
            return cot_path, scaffold_track.generated_submissions

        return None, False

    def get_results_seed_comp(self, seed: str, competition: Competition, results_path_handler: ExpPathHandler) -> tuple[
        dict[str, list], ProgressElements]:
        cot_start = AgentKDSCoT(from_ci=False, time_limit=self.cot_time_limit, get_cot_path=self.get_cot_path)
        return self.aux_get_result(
            seed=seed, competition=competition, cot_start=cot_start, results_path_handler=results_path_handler
        )


V1_0_MEDAL_DICT = {
    "sbu-ai-finalproject": GoldMedal,
    "sign-language-image-classification": GoldMedal,
    "world-championship-2023-embryo-classification": GoldMedal,
    "sentiment-analysis-on-movie-reviews": GoldMedal,
    "playground-series-s3e9": GoldMedal,
    "playground-series-s3e15": GoldMedal,
    "nlpsci": SilverMedal,
    "home-data-for-ml-course": SilverMedal,
    "invasive-species-monitoring": SilverMedal,
    "dogs-vs-cats-redux-kernels-edition": BronzeMedal,
    "playground-series-s4e4": BronzeMedal,
    "playground-series-s3e22": BronzeMedal,
    "otto-group-product-classification-challenge": BronzeMedal,
    "ml-olympiad-landscape-image-classification": BronzeMedal,
    "noaa-right-whale-recognition": BronzeMedal,
    "nlp1000-ml-challenge": BronzeMedal,
}


def get_candidate_leaderboard_path(competition: Competition, root_path_to_leaderboard: pathlib.Path | str) -> list[str]:
    if isinstance(root_path_to_leaderboard, str):
        root_path_to_leaderboard = Path(root_path_to_leaderboard)
    potential_files = []
    try_public = False
    if competition.leaderboard_end_date.end_date is None:
        print(competition.comp_name, "No end date")
        try_public = True
    elif competition.leaderboard_end_date.end_date > datetime.today():
        print(competition.comp_name, f"Not over: {competition.leaderboard_end_date.end_date}")
        try_public = True
    elif competition.perc_private == 0:
        try_public = True

    if try_public:
        for leaderboard_pattern in ["public", "private"]:
            path_to_leaderboard = root_path_to_leaderboard / (
                    competition.comp_name + "-" + leaderboard_pattern + "leaderboard-*.csv")
            potential_files.extend(glob.glob(str(path_to_leaderboard)))
    else:
        leaderboard_pattern = "private"
        path_to_leaderboard = root_path_to_leaderboard / (
                competition.comp_name + "-" + leaderboard_pattern + "leaderboard-*.csv")
        potential_files.extend(glob.glob(str(path_to_leaderboard)))
    return potential_files


def get_leaderboards_from_competitions(competitions: list[Competition], root_path_to_leaderboard: pathlib.Path | str) \
        -> dict[CompetitionID, DataFrame]:
    leaderboards: dict[CompetitionID, pd.DataFrame] = {}
    for competition in tqdm(sorted(competitions, key=lambda c: c.start_date, reverse=1), desc="Get leaderboards"):
        matching = get_candidate_leaderboard_path(
            competition=competition, root_path_to_leaderboard=root_path_to_leaderboard
        )
        if len(matching) != 1:
            print(competition.comp_name, matching)
            matching = sorted(matching)
        leaderboard = pd.read_csv(matching[0])
        assert sum(leaderboard.Rank > 0) > 0
        leaderboard = leaderboard[leaderboard.Rank > 0]  # drop sample submissions
        leaderboards[competition.competition_id] = leaderboard.drop(["TeamId", "SubmissionCount"], axis=1)
    return leaderboards


def expand_team_members(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """ Create separate rows for each team member in the leaderboard """
    rows = []
    for _, row in leaderboard.iterrows():
        members = row[KagLBColname.TEAM_MEMBER_USER_NAMES.value].split(",")  # Split by comma
        for member in members:
            try:
                date = datetime.strptime(row[KagLBColname.LAST_SUBMISSION_DATE.value], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                date = row[KagLBColname.LAST_SUBMISSION_DATE.value]
                assert np.isnan(date), date
            rows.append(
                {
                    KagLBColname.RANK.value: row[KagLBColname.RANK.value],
                    KagLBColname.SCORE.value: row[KagLBColname.SCORE.value],
                    KagLBColname.LAST_SUBMISSION_DATE.value: date,
                    KagLBColname.USER_NAME.value: member  # Add each member individually
                }
            )
    return pd.DataFrame(rows)


def get_quantiles_from_scores(
        scores: float | np.ndarray,
        leaderboard: pd.DataFrame,
        is_lower_better: bool | None,
        return_rank: bool = False
):
    """
    Args:
        scores: a single score (float) or 1D array of scores
        leaderboard: leaderboard of the competition
        is_lower_better: whether lower score is better
        return_rank: whether to return both rank and quantile

    Returns:
        - If return_rank=False: quantiles
        - If return_rank=True: (ranks, quantiles)
    """
    # Ensure scores is an ndarray
    scalar_input = np.isscalar(scores)
    scores = np.atleast_1d(scores).astype(float)

    if is_lower_better is None:
        is_lower_better = check_is_lower_better(leaderboard.Score.values)

    # Compute ranks
    if is_lower_better:
        ranks = np.sum(scores[None, :] > leaderboard.Score.values.flatten()[:, None], axis=0)
    else:
        ranks = np.sum(scores[None, :] < leaderboard.Score.values.flatten()[:, None], axis=0)

    # Convert to quantiles (higher is better)
    quantiles = 100 - ranks / len(leaderboard) * 100
    quantiles[np.isnan(scores)] = 0

    # If input was scalar, return scalar outputs
    if scalar_input:
        ranks = int(ranks[0])
        quantiles = float(quantiles[0])

    if return_rank:
        return ranks, quantiles
    return quantiles


def get_quantile_from_several_submissions(
        pub_priv_scores: np.ndarray, leaderboard: pd.DataFrame, num_select: int,
        comp_name: str, is_lower_better: bool | None
) -> tuple[float, float, int]:
    """
    Get the quantile from an array of shape (n_submission, 2) with the first column corresponding to public scores
    and the second column corresponding to private scores.

    Args:
        pub_priv_scores: 2d-array (number of submissions, 2), i-th element [pub_score, priv_score] of i-th submission
        leaderboard:  competition leaderboard
        num_select: number of selected predictions to submit
        comp_name: competition name
        is_lower_better: whether lower score is better than higher score

    Returns:
        quantile: quantile achieved
        priv_perf: the best score among selected ones
        selected_ind: the index corresponding to the best submission
    """
    if is_lower_better is None:
        is_lower_better = check_is_lower_better(leaderboard.Score.values)
    sorted_inds = np.argsort([pub_priv[0] for pub_priv in pub_priv_scores])
    if not is_lower_better:
        sorted_inds = sorted_inds[::-1]
    sorted_perfs = [pub_priv_scores[i] for i in sorted_inds]
    last_selected_perf_public = sorted_perfs[:num_select][-1][0]
    selected_perfs: list[tuple[float, float]] = sorted_perfs[:num_select] + [
        pub_priv for pub_priv in sorted_perfs[num_select:] if pub_priv[0] == last_selected_perf_public
    ]
    if len(selected_perfs) > num_select:
        print(f"\t- There is a public tie: {comp_name}")

    priv_perfs: list[float] = [pub_priv[1] for pub_priv in selected_perfs[:num_select]]
    if is_lower_better:
        ind = np.argmin(priv_perfs).item()
        priv_perf = priv_perfs[ind]
        rank = np.sum(priv_perf > leaderboard.Score.values.flatten()) + 1
    else:
        ind = np.argmax(priv_perfs).item()
        priv_perf = priv_perfs[ind]
        rank = np.sum(priv_perf < leaderboard.Score.values.flatten()) + 1

    selected_ind: int = sorted_inds[ind]

    quantile = 100 - (rank - 1) / len(leaderboard) * 100
    return quantile, priv_perf, selected_ind


def get_medals_from_quantiles(
        compname_quantiles: dict[str, float], leaderboards: dict[str, pd.DataFrame],
        real_participant: bool = False
) -> dict[str, Type[Medal]]:
    """ 
    Args:
        compname_quantiles: dictionary mapping competition names to quantiles 
        leaderboards:
        real_participant: 

    Returns:

    """
    n_entries_s = {
        compname: len(leaderboards[CompetitionID.get_enum_element(compname)]) + int(not real_participant)
        for compname in compname_quantiles
    }
    ranks = {
        compname: np.round(n_entries_s[compname] - compname_quantiles[compname] * (n_entries_s[compname] - 1) / 100) for
        compname in compname_quantiles
    }
    medals = {
        compname: get_medal(rank=int(ranks[compname]), n_entries=n_entries_s[compname])
        for compname in compname_quantiles
    }
    return medals


def deduplicate_subs(raw_results: pd.DataFrame) -> pd.DataFrame:
    """ Remove submissions that have exactly the same predictions """
    missing_hash_subset = raw_results[raw_results[BaseTableOfResultsColname.SUBMISSION_HASH.value].isna()]
    raw_results = raw_results.dropna(subset=[BaseTableOfResultsColname.SUBMISSION_HASH.value])  # no missing hash
    unique_columns = [
        BaseTableOfResultsColname.COMP_NAME.value,
        BaseTableOfResultsColname.SUBMISSION_HASH.value,
        BaseTableOfResultsColname.PRIV_SCORE.value,
    ]
    unique_subset = raw_results.drop_duplicates(subset=unique_columns, keep="first")
    return pd.concat([missing_hash_subset, unique_subset], ignore_index=True)


def add_medal_column(quantiles: pd.DataFrame, leaderboards: dict[CompetitionID, pd.DataFrame]) -> None:
    """ Add a column indicating obtained medals to a dataframe having a 'quantile' column. """
    medal_dict = get_medals_from_quantiles(
        compname_quantiles=quantiles[BaseTableOfResultsColname.PRIV_QUANTILE.value].to_dict(),
        leaderboards=leaderboards
    )
    medal_df = pd.DataFrame.from_dict({comp: [medal] for comp, medal in medal_dict.items()}).T
    medal_df.columns = [BaseTableOfResultsColname.MEDAL_LEVEL.value]
    quantiles[BaseTableOfResultsColname.MEDAL_LEVEL.value] = medal_df.loc[quantiles.index]


def get_quantiles_from_raw_results(
        raw_results: pd.DataFrame, competitions: list[Competition], leaderboards: dict[CompetitionID, pd.DataFrame],
        include_missing: bool, pbar_desc: str | None = None
) -> pd.DataFrame:
    """

    Args:
        raw_results: table containing raw results for the different competitions
        competitions: competitions to consider
        leaderboards: map of competition id to leaderboard
        include_missing: whether to add 0-quantiles for the missing entries of raw_results
        pbar_desc: set pbar description

    Returns:
        quantiles_df: a table whose indices are competition names and having a column of private quantiles

    """
    quantiles = []
    if pbar_desc is None:
        pbar_desc = "Get clean results"
    pbar = tqdm(competitions)
    for competition in pbar:
        pbar.set_description(f"{pbar_desc} -- {competition.comp_name}")
        comp_results = raw_results[raw_results[BaseTableOfResultsColname.COMP_NAME.value] == competition.comp_name]
        leaderboard = leaderboards[competition.competition_id]
        if len(comp_results) == 0:
            if not include_missing:
                continue
            is_lower_better = check_is_lower_better(arr=leaderboard.Score.values)
            quantiles.append([competition.comp_name, 0, np.nan, np.nan, is_lower_better])
            continue
        perf_groups = [tuple(scores) for scores in comp_results[
            [BaseTableOfResultsColname.PUB_SCORE.value, BaseTableOfResultsColname.PRIV_SCORE.value]].values]
        submission_files = comp_results[BaseTableOfResultsColname.SUBMISSION_PATH.value].values

        fields_that_should_be_equal = [
            BaseTableOfResultsColname.NUM_PEOPLE.value,
            BaseTableOfResultsColname.NUM_SELECT.value
        ]
        for field in fields_that_should_be_equal:
            vals = comp_results[field].values
            assert np.all(vals[0] == vals), (competition.comp_name, vals, field)
        assert len(leaderboard) == comp_results[BaseTableOfResultsColname.NUM_PEOPLE.value].iloc[0], (
            len(leaderboard), comp_results[BaseTableOfResultsColname.NUM_PEOPLE.value].iloc[0]
        )
        assert comp_results[BaseTableOfResultsColname.NUM_SELECT.value].iloc[0] > 0, \
            comp_results[BaseTableOfResultsColname.NUM_SELECT.value].iloc[0]

        num_select = int(comp_results[BaseTableOfResultsColname.NUM_SELECT.value].iloc[0])
        is_lower_better = bool(comp_results[BaseTableOfResultsColname.IS_LOWER_BETTER.value].iloc[0])

        quantile, priv_score, index = get_quantile_from_several_submissions(
            pub_priv_scores=np.array(perf_groups), leaderboard=leaderboard, num_select=num_select,
            comp_name=competition.comp_name, is_lower_better=is_lower_better
        )
        submission_path = submission_files[index]
        quantiles.append([competition.comp_name, quantile, priv_score, submission_path, is_lower_better])
    columns = [
        BaseTableOfResultsColname.COMP_NAME.value,
        BaseTableOfResultsColname.PRIV_QUANTILE.value,
        BaseTableOfResultsColname.PRIV_SCORE.value,
        BaseTableOfResultsColname.SUBMISSION_PATH.value,
        BaseTableOfResultsColname.IS_LOWER_BETTER.value
    ]
    df = pd.DataFrame(
        quantiles, columns=columns
    )

    df = df.set_index(BaseTableOfResultsColname.COMP_NAME.value)
    add_medal_column(quantiles=df, leaderboards=leaderboards)
    return df


def plot_a_vs_b(
        a_clean_results: pd.DataFrame,
        b_clean_results: pd.DataFrame,
        method_a_label: str,
        method_b_label: str,
        competitions: list[Competition],
) -> tuple[go.Figure, int, int]:
    """
    Compare results of methods A and method B.

    Args:
        a_clean_results: results achieved with method A
        b_clean_results: results achieved with method B
        method_a_label: label of A
        method_b_label: label of B
        competitions: list of competitions to consider

    Returns:
        figure: the figure
        success: number of absolut success of A and B
    """
    tab_comp_names = [competition.comp_name for competition in competitions if competition.is_fully_tabular()]
    non_tab_comp_names = [competition.comp_name for competition in competitions if not competition.is_fully_tabular()]
    comp_awards_medals = {competition.comp_name: competition.award_medals for competition in competitions}

    method_a_non_tab_res = a_clean_results.loc[non_tab_comp_names]
    method_a_tab_res = a_clean_results.loc[tab_comp_names]
    method_b_tab_res = b_clean_results.loc[tab_comp_names]
    method_b_non_tab_res = b_clean_results.loc[non_tab_comp_names]

    award_medals_color = "rgba(128, 0, 128, 0.5)"

    fig = go.Figure()

    # Add diagonal line for comparison (optional, to show equal values line)
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name="Equal Quantiles",
            showlegend=False
        )
    )

    # Define grid for heatmap
    x_range = np.linspace(0, 100, 100)
    y_range = np.linspace(0, 100, 100)
    x, y = np.meshgrid(x_range, y_range)

    # Define gradient effect based on distance from x=y line
    z = (y - x)  # More negative = darker green in the top left

    fig.add_trace(
        go.Heatmap(
            x=x_range, y=y_range, z=z,
            colorscale=[(0, 'red'), (0.5, 'white'), (1, 'green')],
            opacity=0.3,  # Adjust for transparency
            showscale=False,
            showlegend=False,
            hoverinfo='skip',
            zorder=-2
        )
    )

    for method_a_res, method_b_res, symbol, label in zip(
            [method_a_non_tab_res, method_a_tab_res],
            [method_b_non_tab_res, method_b_tab_res], ['circle', 'diamond'], ["CV / NLP", "Tab."]
    ):
        method_a_medals = method_a_res[BaseTableOfResultsColname.MEDAL_LEVEL.value]
        method_b_medals = method_b_res[BaseTableOfResultsColname.MEDAL_LEVEL.value]

        fig.add_trace(
            go.Scatter(
                x=method_a_res[BaseTableOfResultsColname.PRIV_QUANTILE.value].values.flatten(),
                y=method_b_res[BaseTableOfResultsColname.PRIV_QUANTILE.value].values.flatten(),
                mode='markers',
                marker=dict(
                    symbol="circle",
                    size=30,
                    color=[
                        award_medals_color if comp_awards_medals[comp_name] else 'rgba(0, 0, 0, 0)' for comp_name in
                        method_b_res.index
                    ],
                ),
                name="Awards medals",
                showlegend=False,
                zorder=-1,
                hoverinfo='skip'  # Disables hover text
            )
        )

        fig.add_trace(
            go.Scatter(
                x=method_a_res[BaseTableOfResultsColname.PRIV_QUANTILE.value].values.flatten(),
                y=method_b_res[BaseTableOfResultsColname.PRIV_QUANTILE.value].values.flatten(),
                mode='markers',
                marker=dict(
                    symbol=symbol,
                    size=14,
                    color=[
                        darken_color(method_a_medals.loc[comp_name].color, factor=1.25) for comp_name in
                        method_b_res.index
                    ],
                    line=dict(
                        color=[method_b_medals.loc[comp_name].color for comp_name in method_b_res.index],
                        width=4,
                    ),
                ),
                hovertext=[
                    f"{comp_name}:<br>"
                    f"  - {method_a_label}: {method_a_res.loc[comp_name][BaseTableOfResultsColname.PRIV_QUANTILE].item():.1f}<br>"
                    f"  - {method_b_label}: {method_b_res.loc[comp_name][BaseTableOfResultsColname.PRIV_QUANTILE].item():.1f}"
                    for comp_name in method_a_res.index
                ],
                showlegend=False,
            )
        )

        add_legend_spacing(fig=fig, legendgroup=None, legend="legend")

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    symbol=symbol, size=14, color=darken_color(NoMedal.color, 1.25),
                    line=dict(color=NoMedal.color, width=4)
                ),
                name=label,
                legend="legend"
            )
        )

    add_legend_spacing(fig=fig, legendgroup=None, legend="legend")

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=30, color=award_medals_color),
            name="Awards medals",
            legend="legend",
            showlegend=True,
        )
    )

    for medal in [GoldMedal, SilverMedal, BronzeMedal]:
        add_legend_spacing(fig=fig, legendgroup=None, legend="legend2")
        for inner_color, border_color, name in zip(
                ["white", darken_color(medal.color, 1.25)], [medal.color, "white"],
                [f"{medal.name.value} for {method_b_label}", f"{medal.name.value} for {method_a_label}"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        symbol="square",
                        size=14,
                        color=inner_color,  # darken_color(medal.color, 1.25),
                        line=dict(
                            color=border_color,
                            width=4,
                        ),
                    ),
                    name=name,
                    legendgroup=None,
                    legend="legend2",
                )
            )

    priv_q_col = BaseTableOfResultsColname.PRIV_QUANTILE.value
    method_a_quantiles = a_clean_results.loc[b_clean_results.index][priv_q_col].values
    method_b_quantiles = b_clean_results[priv_q_col].values
    n_method_a_victories = (method_a_quantiles > method_b_quantiles).sum()
    n_method_b_victories = (method_a_quantiles < method_b_quantiles).sum()
    n_competitions = len(competitions)

    # Add text on the top left and bottom right
    for x_pos, y_pos, text, xanchor, yanchor, n_victories in zip(
            [0, 100], [112, 0], [f'{method_b_label} > {method_a_label}', f'{method_a_label} > {method_b_label}'],
            ['left', 'right'],
            ['top', 'bottom'], [n_method_b_victories, n_method_a_victories]
    ):
        fig.add_annotation(
            text=text + f"<br>({n_victories} / {n_competitions} competitions)",
            x=x_pos, y=y_pos,
            showarrow=False,
            font=dict(size=14, color='black', family="STIXGeneral, Times New Roman, serif", weight=800),
            xanchor=xanchor, yanchor=yanchor
        )

    fig.update_layout(
        legend=dict(
            y=0.95,
            title=dict(
                font=dict(family="STIXGeneral, Times New Roman, serif", weight=1000, size=16),
                text="Competition types"
            ),
            font=dict(family="STIXGeneral, Times New Roman, serif", weight=800, size=14),
            bordercolor="white",
            borderwidth=2,
            indentation=10
        )
    )

    fig.update_layout(
        legend2=dict(
            y=0.,
            title=dict(
                font=dict(family="STIXGeneral, Times New Roman, serif", weight=1000, size=16),
                text="Medal types"
            ),
            font=dict(family="STIXGeneral, Times New Roman, serif", weight=800, size=14),
            bordercolor="white",
            borderwidth=2,
            indentation=10
        )
    )

    fig.update_layout(
        xaxis_title=f"Private Quantile (%) of {method_a_label}",
        yaxis_title=f"Private Quantile (%) of {method_b_label}",
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=700,
        width=900,
        xaxis=dict(
            title_font=dict(color='black', family="STIXGeneral, Times New Roman, serif", weight=1000),
        ),
        yaxis=dict(
            title_font=dict(color='black', family="STIXGeneral, Times New Roman, serif", weight=1000),
        ),
    )

    return fig, n_method_a_victories, n_method_b_victories


def compute_elo_mmr(
        competition_ids: list[CompetitionID], extended_leaderboards: dict[CompetitionID, pd.DataFrame],
        participants: list[str]
) -> dict[str, Player]:
    """
    Compute Elo-MMR score of participants given a sorted list of leaderboards
    Args:
        participants: list of participant names to consider
        extended_leaderboards: map of competition ID to leaderboards.
        competition_ids: list of competition IDs. Elo will be computed following the order of the IDs

    Returns:
        players_to_elo_records: dictionary mapping player name to the Elo-Player object.

    """
    elo_mmr = EloMMR()
    players = {p: Player() for p in participants}

    for comp_id in tqdm(competition_ids, desc=f"Computing Elo-MMR over {len(players)} players"):
        leaderboard = extended_leaderboards[comp_id]
        standings = []
        for i in range(len(leaderboard)):
            p = leaderboard.iloc[i].UserName
            if p not in players:
                raise RuntimeError(f"{p} not in `participants`")
            standings.append(
                (
                    players[p],
                    leaderboard.iloc[i][KagLBColname.MIN_RANK.value],
                    leaderboard.iloc[i][KagLBColname.MAX_RANK.value]
                )
            )
        elo_mmr.round_update(standings)

    return players


def plot_elo_mmr_evol(
        competitions: list[Competition], extended_leaderboards_w_us: dict[CompetitionID, pd.DataFrame],
        players: dict[str, Player],
        method_name: str, original_leaderboards: dict[CompetitionID, pd.DataFrame], original_quantiles: pd.DataFrame,
        dates: list[datetime] = None
) -> go.Figure:
    """

    Args:
        competitions: list of competitions over which ELO was computed
        extended_leaderboards_w_us: map competition ID to extended_leaderboards that includes the method to evaluate.
        players: map player name to the Elo-Player object.
        method_name: name of the method assessed
        original_leaderboards: original leaderboards of the competitions
        original_quantiles: original quantiles of the method
        dates: list of competition date (if None, dates of latest leaderboard entry)

    Returns:

    """
    our_quantiles = {}
    for competition in competitions:
        leaderboard = extended_leaderboards_w_us[competition.competition_id]
        our_rank = leaderboard[leaderboard[KagLBColname.USER_NAME.value] == method_name][KagLBColname.MIN_RANK.value]
        our_quantiles[competition.comp_name] = (len(leaderboard) - our_rank.item()) / len(leaderboard)

    txt_comp_ids = list(filter(lambda c: c.has_txt_input, competitions))
    img_comp_ids = list(filter(lambda c: c.has_img_input, competitions))
    tab_comp_ids = list(filter(lambda c: c.is_fully_tabular(), competitions))
    medal_awarding_comp_id = next(filter(lambda c: c.award_medals, competitions))

    if dates is None:
        dates = []
        for competition in competitions:
            dates.append(extended_leaderboards_w_us[competition.competition_id].LastSubmissionDate.max())

    year_positions = []
    unique_years = []
    year_annotation_shifts = []

    current_year = dates[0].year
    for i in range(1, len(dates)):
        year = dates[i].year
        if year == current_year:
            continue
        delta = (dates[i] - dates[i - 1]).days
        unique_years.append(year)
        current_year = year
        year_positions.append(i - 1 + delta / 365)
        year_annotation_shifts.append(len(year_annotation_shifts) % 2)

    def get_competition_color_based_on_type(comp: Competition) -> str:
        if comp.has_txt_input:
            return "#20B2AA"
        elif comp.has_img_input:
            return "#F4A300"
        return "#4682B4"

    def get_line_width(comp: Competition) -> int:
        if comp.award_medals:
            return 3
        return 0

    mu_values = [e.mu for e in players[method_name].event_history]
    original_quantile = original_quantiles[BaseTableOfResultsColname.PRIV_QUANTILE.value]
    hovertext = [
        f"ELO: {int(mu_values[i])}<br>" \
        f"Competition: {comp.comp_name}<br>" \
        f"Quantile: {our_quantiles[comp.comp_name] * 100:.1f}%<br>" \
        f"Num. players: {len(extended_leaderboards_w_us[comp.competition_id])}<br>" \
        f"Original quantile: {original_quantile.loc[comp.comp_name].item():.1f}% " \
        f"(out of {len(original_leaderboards[comp.competition_id])})"
        for i, comp in enumerate(competitions)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(mu_values)),
            y=mu_values,
            mode='lines',
            line=dict(color='grey', width=2),  # Line color and width
            opacity=0.6,  # Set opacity here
            name=None,
            showlegend=False  # This trace will not appear in the legend
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(mu_values)),
            y=mu_values,
            mode='markers',
            name=None,
            hovertext=hovertext,
            showlegend=False,  # This trace will not appear in the legend
            marker=dict(
                size=10,
                color=[get_competition_color_based_on_type(comp=comp) for comp in competitions],
                opacity=0.8,
                line=dict(width=[get_line_width(comp) for comp in competitions], color='black')
            )
        )
    )

    # Add vertical lines corresponding to the years and annotations
    for year_position, year, year_annotation_shift in zip(year_positions, unique_years, year_annotation_shifts):
        # Add the vertical line
        fig.add_vline(
            x=year_position,
            line=dict(color="black", width=2, dash="dot"),  # Customize line style as needed
            opacity=0.7,
            name=str(year),
        )

        # Add the annotation
        fig.add_annotation(
            x=year_position + 1.3,  # Shift slightly to the right to avoid overlap with the line
            y=max(mu_values) + year_annotation_shift * (max(mu_values) - min(mu_values)) * .05,
            # Position the annotation at the top of the plot
            text=str(year),
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            opacity=0.7
        )

    for comp_id, label in zip([txt_comp_ids[0], img_comp_ids[0], tab_comp_ids[0]], ["NLP", "CV", "TAB"]):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=f"{label} competition",
                marker=dict(
                    size=10,
                    color=get_competition_color_based_on_type(comp_id),
                    opacity=0.8,
                    line=dict(width=0, color='black')
                )
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=f"Medal-awarding competition",
            marker=dict(
                size=10,
                color="white",
                line=dict(width=get_line_width(medal_awarding_comp_id), color='black')
            )
        )
    )

    # Add title and labels
    fig.update_layout(
        title=f'Evolution of {method_name} ELO-score Over Time',
        xaxis_title='Competition last submission date.',
        yaxis_title='Elo Score Value',
        template='plotly',
        width=1300,
        height=800,
        plot_bgcolor="white",  # Explicitly set the plot area background to white
        paper_bgcolor="white",  # Set the overall figure background to white
        showlegend=True,
        xaxis=dict(
            tickvals=np.arange(len(competitions)),  # Positions for ticks
            ticktext=[comp.comp_name for comp in competitions],  # Competition names as labels
            tickangle=90  # Optional: Rotate the x-axis labels for better readability
        )
    )

    fig.update_layout(
        legend=dict(
            x=1,  # Position the legend to the right of the plot (1 is the far-right edge)
            y=.9,  # Position the legend at the top of the plot (1 is the top edge)
            xanchor='right',  # Anchor the legend to the right edge of the plot
            yanchor='top',  # Anchor the legend to the top edge of the plot
            bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: background color for the legend
            font=dict(size=12, color='black')  # Optional: font size and color
        )
    )

    return fig


def plot_elo_hist(
        players_elo: dict[str, Player], usr_levels_dict: dict[str, type[KaggleLevel]], method_name: str,
        method_elo: float, dark_mode: bool
) -> go.Figure:
    """

    Args:
        players_elo: map from player name to player ELO object
        usr_levels_dict: map from player name to player Kaggle level
        method_name: name of the assessed method
        method_elo: elo score of the method
        dark_mode: whether it is in light or dark mode

    Returns:
        figure of the ELO histogram
    """
    elo_scores = pd.DataFrame.from_dict({p: {"elo": players_elo[p]} for p in players_elo}).T

    user_level_elo_dict = {}
    missing_usr_levels = set()
    for usr in elo_scores.index:
        if usr == method_name:
            continue
        user_level_elo_dict[usr] = {"elo": elo_scores.loc[usr].item()}
        if usr not in usr_levels_dict:
            level = BaseKaggler
        else:
            level = usr_levels_dict[usr]
            assert level != UnknownLevel
            if level in [Contributor, Novice]:
                level = BaseKaggler
        user_level_elo_dict[usr]["level"] = level.name.value
    print(f"Missing users: {len(missing_usr_levels)}")
    save_w_pickle(obj=missing_usr_levels, path=os.path.join(ROOT_PROJECT, "missing_usr_levels.pkl"))

    user_level_elo_df = pd.DataFrame.from_dict(user_level_elo_dict).T

    avgs_per_levels = user_level_elo_df.groupby("level")["elo"].mean().to_dict()

    def percentile(n: int) -> Callable[[np.ndarray], floating[...]]:
        def percentile_(x: np.ndarray) -> floating[...]:
            return np.percentile(x, n)

        percentile_.__name__ = f'perc_{n}'
        return percentile_

    perc_per_levels_df = user_level_elo_df.groupby("level")["elo"].agg([percentile(i) for i in [0, 25, 50, 75, 100]])

    # Calculate quantiles
    sorted_elo = sorted(user_level_elo_df.elo.values.flatten())
    quantiles = np.percentile(sorted_elo, np.arange(0, 101, 1))  # 0th to 100th percentile
    # Find the quantile for value v
    quantile_index = np.searchsorted(quantiles, method_elo)
    if quantile_index % 10 == 1:
        quant = "st"
    elif quantile_index % 10 == 2:
        quant = "nd"
    elif quantile_index % 10 == 3:
        quant = "rd"
    else:
        quant = "th"

    my_annotation = f"{method_name} ({method_elo})<br>{quantile_index:.0f}{quant} percentile"

    contrast_color = "white" if dark_mode else "black"

    fig = px.histogram(
        user_level_elo_df,
        x="elo",
        y=None,
        color="level",
        color_discrete_map={lev.name.value: lev.color for lev in KAGGLE_LEVELS if
                            lev not in [Novice, Contributor, UnknownLevel]},
        category_orders={
            "level": sorted(
                [lev.name.value for lev in KAGGLE_LEVELS if lev not in [Novice, Contributor, UnknownLevel]],
                key=lambda lev_name: KaggleLevel.get_kaggle_level_from_str(lev_name).value
            )
        }
    )

    extra_elo = None
    for level in avgs_per_levels:
        if level == UnknownLevel.name.value:
            continue
        elo_data = user_level_elo_df[user_level_elo_df["level"] == level].elo.values
        print(level, len(elo_data))
        k_level = KaggleLevel.get_kaggle_level_from_str(level)
        annotation = level
        y = - (k_level.value + 1) * 22
        x_start = perc_per_levels_df.loc[level].perc_25.item()
        x_end = perc_per_levels_df.loc[level].perc_75.item()
        x_median = perc_per_levels_df.loc[level].perc_50.item()
        color = k_level.color
        extra_elo = elo_data[np.logical_or(elo_data < x_start, (elo_data > x_end))]
        fig.add_shape(
            type="line",
            x0=x_start, x1=x_end,
            y0=y, y1=y,
            line=dict(color=color, width=4, dash="solid"),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_median, x_end], y=[y, y, y], mode="markers",
                marker=dict(
                    symbol=["line-ns", "diamond", "line-ns"], size=15, color=color, line_width=2,
                    line_color=color
                ),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=extra_elo, y=[y for _ in range(len(extra_elo))], mode="markers",
                marker=dict(size=5, color=color, opacity=.2), showlegend=False
            )
        )

        fig.add_annotation(
            x=x_end + 50,
            y=y + 11,
            text=annotation,
            showarrow=False,
            font=dict(size=15, color=color, weight=700),
            xanchor="left"
        )

    y_top = 220
    fig.add_shape(
        type="line",
        x0=method_elo, x1=method_elo,
        y0=-(Grandmaster.value + 1) * 22 - 15, y1=y_top,
        line=dict(color="red", width=4, dash="dash"),
    )
    fig.add_annotation(
        x=method_elo,
        y=y_top + 11,
        text=my_annotation,
        showarrow=False,  # Whether to show an arrow pointing to the text
        font=dict(size=15, color="red", weight=700),
        xanchor="center", yanchor="bottom"
    )

    elo_values = np.unique(user_level_elo_df.elo.values.flatten())
    p5 = min(elo_values) + 10
    p25 = p5 + 400
    samples = np.random.uniform(p5, p25, 100)
    qs = [np.percentile(samples, p) for p in [25, 50, 75]]
    y = 180
    fig.add_trace(
        go.Scatter(
            x=qs,
            y=[y for _ in qs],
            mode="lines+markers",
            line=dict(color=contrast_color, width=4),  # Line style
            marker=dict(
                size=15, symbol=["line-ns", "diamond", "line-ns"], color=contrast_color, line_width=2,
                line_color="black"
            ),
            showlegend=False
        )
    )
    for i in range(3):
        fig.add_annotation(
            x=qs[i], y=y + 25, text=f"Q{i + 1}", showarrow=False,
            font=dict(size=15, color=contrast_color, weight=700), xanchor="center"
        )
    extra_points = samples[np.logical_or(samples < qs[0], samples > qs[-1])]
    fig.add_trace(
        go.Scatter(
            x=extra_points,
            y=[y for _ in range(len(extra_elo))],
            mode="markers",
            marker=dict(size=5, color=contrast_color, opacity=.2),
            showlegend=False
        )
    )

    fig.update_layout(
        xaxis_title="Elo-MMR Score",
        yaxis_title=f"Count",
        bargap=0.1,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        width=1200,
        height=600,
        xaxis=dict(
            tickfont=dict(size=14, family="STIXGeneral, Times New Roman, serif", color=contrast_color),
            title_font=dict(size=22, family="STIXGeneral, Times New Roman, serif", weight=1000, color=contrast_color),
        ),
        yaxis=dict(
            tickfont=dict(size=14, family="STIXGeneral, Times New Roman, serif", color=contrast_color),
            title_font=dict(size=22, family="STIXGeneral, Times New Roman, serif", color=contrast_color, weight=1000),
            tickmode="array",
            tickvals=np.arange(0, 250, 50),
        ),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=15, family="STIXGeneral, Times New Roman, serif", color=contrast_color, weight=700),
            title="Kaggle Level"
        )
    )

    return fig


def hash_pandas_df(df: pd.DataFrame) -> str:
    """ Hash a pandas dataframe reproducibly

    Args:
        df: content of dataframe

    Returns:
        hash of the dataframe
    """
    # Sort by index if there's an 'id' column
    if 'id' in df.columns and df['id'].is_unique:
        df = df.sort_values(by='id')
    elif df[df.columns[0]].is_unique:
        df = df.sort_values(by=df.columns[0])

    # Create a string representation of each column's content and hash it
    col_hashes = []
    for col in sorted(df.columns):
        # Attempt to convert column to float if possible
        try:
            col_data = df[col].astype(float)
        except (ValueError, TypeError):
            col_data = df[col]

        col_str = ' -@- '.join(map(str, col_data.tolist()))

        h = hashlib.sha256(col_str.encode()).hexdigest()
        col_hashes.append(h)

    # Create a final hash
    final_hash_input = ' -|- '.join(col_hashes)
    final_hash = hashlib.sha256(final_hash_input.encode()).hexdigest()

    return final_hash


def get_hash_submission_file(submission_file: Path) -> Path:
    return submission_file.parent / ("hash" + submission_file.name.replace(".csv", ".txt"))

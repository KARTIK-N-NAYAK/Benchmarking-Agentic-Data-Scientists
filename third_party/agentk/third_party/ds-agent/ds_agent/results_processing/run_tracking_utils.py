from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import NewType, ClassVar

from pydantic import BaseModel

from ds_agent.competition_ids import CompetitionID
from ds_agent.utils import StringColors, string_w_color, format_timedelta, ListableEnum
from ds_agent.utils_kaggle import Medal


class ExpLLM(str, ListableEnum):
    LLM_PLAYGROUND_QWEN2_5_72B = "Qwen2.5-72b"
    DEEPSEEK_R1 = "Deepseek-R1"


class ProgressStatusType(Enum):

    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """ Generate the next value when not given. """
        return " ".join(name.split("_"))

    NO_NEED_RUNNING = auto()
    RUNNING = auto()
    TO_CHECK_PROBABLY_SUCCESS = auto()
    TO_CHECK_PROBABLY_FAILURE = auto()
    FINISHED_SUCCESS = auto()
    FINISHED_FAILURE = auto()
    TO_SUBMIT = auto()
    NOT_STARTED = auto()
    NO_SETUP_DIR = auto()
    NO_RAMP_DIR = auto()
    TOO_MANY_RAMP_DIRS = auto()
    RAMP_NOT_OVER = auto()
    CI_FOR_REGRESSION = auto()
    TOO_MANY_FOLDERS = auto()
    MISSING_VERBOSE_LOG = auto()
    MISSING_REACT_EXPL_CONFIG = auto()
    TIME_LIMIT_MISMATCH = auto()
    INTERMEDIATE_NODE_PROBLEM = auto()
    TO_REGENERATE_NODES = auto()
    TOO_MANY_MAIN_PIPELINE = auto()
    NO_MAIN_PIPELINE = auto()
    MISSING_COT_TO_START_FROM = auto()
    RAN_DESPITE_MISSING_COT = auto()
    CANNOT_RUN_NO_SETUP = "CANNOT RUN (NO SETUP)"
    CANNOT_READ_JOURNAL = auto()
    MISSING_RAMP_SUBMISSION_FILE = auto()
    MISSING_RAMP_SUMMARY = auto()
    RELEASE_DATE_ERROR = auto()
    SUBMISSION_NOT_FOUND = auto()
    BLEND_COMPONENT_MISSING = auto()
    SINGLE_BLEND_COMPONENT = auto()

    @classmethod
    @lru_cache
    def max_name_length(cls) -> int:
        return max(len(status.value) for status in cls)


class TrackingMessage(BaseModel):
    tracking_type: ProgressStatusType
    header: str
    color: StringColors
    body: str
    min_message_length: int = 50

    def __str__(self) -> str:
        header = f"[{self.header} {self.tracking_type.value}]"
        m = string_w_color(text=header, color=self.color, bold=True, min_length=self.min_message_length)
        m += f" {self.body}"
        return m


class ProgressStatusKey(BaseModel):
    seed: str
    competition_id: CompetitionID

    def __hash__(self) -> int:
        return hash((self.seed, self.competition_id))


class ProgressStatus(BaseModel, ABC):
    type_name: ClassVar[ProgressStatusType]
    color: ClassVar[StringColors]
    exp_fullname: str
    seed: str
    comp_id: CompetitionID
    min_exp_name_length: int = 20
    tracking_messages: list[TrackingMessage] = []
    extra_hover_text: str | None = None

    def build_header(self, color: StringColors | None) -> str:
        m = ""
        m += string_w_color(text=f"[SEED {self.seed}]", color=color, bold=True, min_length=10)
        m += string_w_color(
            text=f"[{self.exp_fullname}]", color=color, bold=True, min_length=self.min_exp_name_length
        )
        m += string_w_color(
            text=f"[{self.type_name.value}]", color=color, bold=True, min_length=ProgressStatusType.max_name_length()
        )
        return m

    @abstractmethod
    def build_body(self) -> str:
        return f" {self.comp_id.value}"

    def hover_txt(self) -> str:
        t = self.build_body()
        if self.extra_hover_text is not None and len(self.extra_hover_text) > 0:
            t += " -- " + self.extra_hover_text
        return t.replace(" -- ", "<br>")

    def to_key(self) -> ProgressStatusKey:
        return ProgressStatusKey(seed=self.seed, competition_id=self.comp_id)

    def build_messages(self) -> str:
        m = ""
        for message in self.tracking_messages:
            m += str(message) + "\n"
        return m

    def report_txt(self) -> str:
        return self.build_header(color=self.color) + " " + self.build_body() + self.build_messages()

    def __hash__(self) -> int:
        return hash(self.seed) + hash(self.type_name) + hash(self.exp_fullname) + hash(self.comp_id)


class ProgressStatusSimpleBody(ProgressStatus):
    def build_body(self) -> str:
        return self.comp_id.value


class ProgressStatusWithWorkspace(ProgressStatus):
    workspace: Path | str

    def build_body(self) -> str:
        return self.comp_id.value + " -- " + str(self.workspace)


class ProgressStatusNoNeedRunning(ProgressStatusSimpleBody):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.NO_NEED_RUNNING
    color: ClassVar[StringColors] = StringColors.BLACK


class ProgressStatusRunning(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.RUNNING
    color: ClassVar[StringColors] = StringColors.ORANGE


class ProgressStatusToCheckProbablySuccess(ProgressStatusWithWorkspace):
    delta: timedelta
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TO_CHECK_PROBABLY_SUCCESS
    color: ClassVar[StringColors] = StringColors.PURPLE

    def build_body(self) -> str:
        return self.comp_id.value + f" -- started {format_timedelta(self.delta)} ago -- " + str(self.workspace)


class ProgressStatusToCheckProbablyFailure(ProgressStatusWithWorkspace):
    delta: timedelta
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TO_CHECK_PROBABLY_FAILURE
    color: ClassVar[StringColors] = StringColors.YELLOW

    def build_body(self) -> str:
        return self.comp_id.value + f" -- started {format_timedelta(self.delta)} ago -- " + str(self.workspace)


class ProgressStatusToRegenerateNode(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TO_REGENERATE_NODES
    color: ClassVar[StringColors] = StringColors.BLUE


class ProgressStatusFinishedSuccess(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.FINISHED_SUCCESS
    color: ClassVar[StringColors] = StringColors.FOREST
    submission_paths: list[str]
    submission_quantiles: list[float]
    medal: Medal | None = None

    def build_body(self) -> str:
        body = ""
        if self.medal is not None:
            body += f"[{self.medal.name.value}] "
        body += self.comp_id.value + " -- " + str(self.workspace)
        return body

    def hover_txt(self) -> str:
        str_workspace = str(self.workspace)
        t = self.build_body().replace(" -- ", "<br>")
        for sub_path, sub_quantile in zip(self.submission_paths, self.submission_quantiles):
            t += f"<br>  - {sub_path.replace(str_workspace, './').replace('//', '/')} : {sub_quantile:.1f}"
        return t


class ProgressStatusFinishedFailure(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.FINISHED_FAILURE
    color: ClassVar[StringColors] = StringColors.RED
    generated_submissions: bool
    message: str | None = None
    message_path: Path | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.message_path is not None:
            self.read_message()

    def read_message(self) -> None:
        """ Reads the content from the message_path and sets the message attribute. """
        if self.message_path.exists():
            with open(self.message_path, 'r') as f:
                self.message = f.read().strip()
        else:
            raise FileNotFoundError(f"The file at {self.message_path} does not exist.")

    def build_body(self) -> str:
        body = super().build_body()
        if self.generated_submissions:
            body += f" -- Generated faulty submissions"
        else:
            body += f" -- Generated no submission"
        if self.message is not None and len(self.message) > 0:
            body += " -- \n" + self.message
        return body


class ProgressStatusToSubmit(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TO_SUBMIT
    color: ClassVar[StringColors] = StringColors.MAGENTA


class ProgressStatusNotStarted(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.NOT_STARTED
    color: ClassVar[StringColors] = StringColors.BROWN


class ProgressStatusNoCoTToStartFrom(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.MISSING_COT_TO_START_FROM
    color: ClassVar[StringColors] = StringColors.GREY


class ProgressStatusRanDespiteMissingCoT(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.RAN_DESPITE_MISSING_COT
    color: ClassVar[StringColors] = StringColors.TURQUOISE


class ProgressStatusMissingSetup(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.CANNOT_RUN_NO_SETUP
    color: ClassVar[StringColors] = StringColors.BEIGE


class ProgressStatusCannotReadJournal(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.CANNOT_READ_JOURNAL
    color: ClassVar[StringColors] = StringColors.NAVY


class ProgressStatusNoSetupDir(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.NO_SETUP_DIR
    color: ClassVar[StringColors] = StringColors.YELLOW


class ProgressStatusNoRAMPDir(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.NO_RAMP_DIR
    color: ClassVar[StringColors] = StringColors.YELLOW


class ProgressStatusTooManyRAMPDirs(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TOO_MANY_RAMP_DIRS
    color: ClassVar[StringColors] = StringColors.YELLOW


class ProgressStatusRAMPNotOver(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.RAMP_NOT_OVER
    color: ClassVar[StringColors] = StringColors.BLUE


class ProgressStatusMissingRAMPSubmission(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.MISSING_RAMP_SUBMISSION_FILE
    color: ClassVar[StringColors] = StringColors.PURPLE


class ProgressStatusBlendComponentMissing(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.BLEND_COMPONENT_MISSING
    color: ClassVar[StringColors] = StringColors.LIGHT_BLUE


class ProgressStatusMissingRAMPSummary(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.MISSING_RAMP_SUMMARY
    color: ClassVar[StringColors] = StringColors.BLUE


class ProgressStatusToManyMainPipelines(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TOO_MANY_MAIN_PIPELINE
    color: ClassVar[StringColors] = StringColors.YELLOW


class ProgressStatusNoMainPipelines(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.NO_MAIN_PIPELINE
    color: ClassVar[StringColors] = StringColors.SALMON


class ProgressStatusCIForRegression(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.CI_FOR_REGRESSION
    color: ClassVar[StringColors] = StringColors.YELLOW


class ProgressStatusTooManyFolders(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TOO_MANY_FOLDERS
    color: ClassVar[StringColors] = StringColors.BLACK


class ProgressStatusMissingVerboseLog(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.MISSING_VERBOSE_LOG
    color: ClassVar[StringColors] = StringColors.CYAN


class ProgressStatusMissingReactExplConfig(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.MISSING_REACT_EXPL_CONFIG
    color: ClassVar[StringColors] = StringColors.TURQUOISE


class ProgressStatusMissingTimeLimitMismatchConfig(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.TIME_LIMIT_MISMATCH
    color: ClassVar[StringColors] = StringColors.PINK


class ProgressStatusIntermediateNodeProblem(ProgressStatusWithWorkspace):
    type_name: ClassVar[ProgressStatusType] = ProgressStatusType.INTERMEDIATE_NODE_PROBLEM
    color: ClassVar[StringColors] = StringColors.GREY


ProgressElements = NewType(name="ProgressElements", tp=dict[str, dict[ProgressStatusKey, ProgressStatus]])


def print_report_elements(progress_elements: ProgressElements, show_missing_seeds: bool) -> None:
    progress_to_skip = (
        ProgressStatusFinishedSuccess, ProgressStatusFinishedFailure, ProgressStatusNoNeedRunning,
        ProgressStatusNoCoTToStartFrom
    )
    for exp_name in progress_elements:
        messages = []
        for p in progress_elements[exp_name].values():
            for tracking_message in p.tracking_messages:
                messages.append(str(tracking_message))
            if isinstance(p, progress_to_skip):
                continue
            if not show_missing_seeds and isinstance(p, ProgressStatusNotStarted):
                continue
            messages.append(p.report_txt())
        for m in sorted(messages):
            print(m)

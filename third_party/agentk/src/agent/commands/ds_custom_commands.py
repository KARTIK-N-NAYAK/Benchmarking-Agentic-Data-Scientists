import abc
import itertools
import os
from abc import ABC
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partialmethod, partial
from pathlib import Path
from typing import Annotated, Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from tqdm import tqdm

from agent.agents import LLMAgent, safe_parsing_chat_completion, Agent
from agent.commands import ExecutePlannedAction, DecisionFlow
from agent.commands import Flow
from agent.commands import LoopFlow
from agent.commands import SequentialFlow
from agent.commands.core import Command, HumanTakeoverCommand, MemKeyCopy
from agent.commands.core import DoNothing
from agent.commands.core import UseTool
from agent.commands.flows import PARSE_FUNC_MAP
from agent.commands.utils_commands import CodePassedLoopChoiceCmd, CodePassedPostLoopChoiceCmd, \
    MultiTrialPostLoopChoiceCmd, AlternateCodeCommand
from agent.memory import MemKey
from agent.memory import Memory
from agent.tasks.datascience_task.utils import FileMap
from agent.tasks.feature_engineering import FeatureEngineeringStageNames
from agent.tools.RAG_tool import RAG
from agent.tools.python_interpreter import PythonInterpreterWithBlanks
from agent.utils.utils import extract_submission_as_json
from ds_agent.utils import set_pd_options, reset_pd_options, get_df_stats
from third_party.data_science.env import DSAction, DNNSubmissionState, ClassicalTabSubmissionState
from third_party.data_science.env import DSActionHypKeys
from third_party.data_science.env import DataScienceSubmissionCard
from third_party.data_science.env_stages import DataScienceStageNames


class PlanFixedAction(Command):
    name: str = "act"
    description: str = "Set planned action to a fix value (no call to LLM)"

    action: Any

    def func(self, agent, *args, **kwargs) -> None:
        agent.memory.store(content=self.action, tags={MemKey.NEXT_PLANNED_ACTION})


class PlanDSAction(Command, abc.ABC):
    name: str
    stage_name: DataScienceStageNames
    description: str = "Set planned action to a fix value (no call to LLM)"

    def func(self, agent, *args, **kwargs) -> None:
        agent.memory.store(
            content=DSAction(stage_name=self.stage_name, hyps=self.get_hyps(memory=agent.memory)),
            tags={MemKey.NEXT_PLANNED_ACTION},
        )

    @abc.abstractmethod
    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        raise NotImplementedError()


class NoHypsPlanDSAction(PlanDSAction):
    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {}


class ThinkAndCodePlanDSAction(PlanDSAction):
    """To use to set the DS action after a thinking and coding step.

    Will store the summary and the generated code blank
    """

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {
            DSActionHypKeys.SUMMARY_STEP: memory.retrieve(tags=MemKey.SUMMARY_STEP),
            DSActionHypKeys.CODE_BLANK: memory.retrieve(tags=MemKey.CODE_BLANK),
        }


class ThinkAndCodeFailedPlanDSAction(PlanDSAction):
    """To use to set the DS action after a thinking and coding step.

    Will store the summary and the generated code blank
    """

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {
            DSActionHypKeys.FAILED: True,
        }


class TrainSubmissionPlanDSAction(PlanDSAction):
    """To use to set the DS action after a thinking and coding step.

    Will store the summary and the generated code blank
    """

    name: str = "train_submission"
    stage_name: DataScienceStageNames = DataScienceStageNames.TRAIN_SUBMISSION
    description: str = "Train submission"

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {DSActionHypKeys.SUBMISSION_SUMMARY: memory.retrieve(tags=MemKey.SUBMISSION_SUMMARY)}


class ExecuteBagPlanDSAction(PlanDSAction):
    """To use to set the DS action after a thinking and coding step.

    Will store the summary and the generated code blank
    """

    name: str = "bag_submissions"
    stage_name: DataScienceStageNames = DataScienceStageNames.BAG_SUBMISSIONS
    description: str = "Bag submissions"

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {
            DSActionHypKeys.SUBMISSION_SUMMARY: memory.retrieve(tags=MemKey.SUBMISSION_SUMMARY),
            DSActionHypKeys.SUMMARY_STEP: memory.retrieve(tags=MemKey.SUMMARY_STEP),
            DSActionHypKeys.CODE_BLANK: memory.retrieve(tags=MemKey.CODE_BLANK),
        }


class SummarizeThinkAndCode(HumanTakeoverCommand):
    """Summarize think and code, and remove code-related keys from the agent memory."""

    response_parser_id: str = "extract_summary_as_json"
    max_retries: int = 5
    human_takeover: bool = False

    output_keys: dict[str, MemKey] = {"output_mem_key": MemKey.SUMMARY_STEP}

    predefined_response: str | None = None

    def func(self, agent: LLMAgent, ask_template: str, output_mem_key: MemKey):
        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=ask_template,
            parse_func=PARSE_FUNC_MAP[self.response_parser_id],
            format_error_message='Use the specified JSON structure:\n```json\n{\n\t"summary": "<summary>"\n}\n```',
            max_retries=self.max_retries,
            human_takeover=self.human_takeover,
            predefined_response=self.predefined_response
        )

        agent.memory.store(response, output_mem_key)


class SummarizeOverall(HumanTakeoverCommand):
    """Summarize overall submission steps."""

    response_parser_id: str = "extract_summary_as_json"

    output_keys: dict[str, MemKey] = {MemKey.SUBMISSION_SUMMARY.value: MemKey.SUBMISSION_SUMMARY}

    max_retries: int = 5
    human_takeover: bool = False

    def func(self, agent, *args: Any, **kwargs: Any):
        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates['ask_template'],
            parse_func=PARSE_FUNC_MAP[self.response_parser_id],
            format_error_message='Use the specified JSON structure:\n```json\n{\n\t"summary": "<summary>"\n}\n```',
            max_retries=self.max_retries,
            human_takeover=self.human_takeover
        )
        agent.memory.store(response, self.output_keys[MemKey.SUBMISSION_SUMMARY.value])


class FixedAct(SequentialFlow):
    """Simply execute a given action."""

    def __init__(self, name: str, description: str, action: Any = None):
        """Initializes the SequentialFlow object with a sequence of commands or sub-flows, a name,
        and a description.

        Args:
            name: The name of the flow. Defaults to "sequential_flow".
            description: A brief description of the flow. Defaults to "A sequence of actions".
            action: action to execute (if None, action will be set to the name)
        """
        if action is None:
            action = name
        super().__init__(
            name=name, description=description, sequence=[PlanFixedAction(action=action), ExecutePlannedAction()]
        )


class FixedDSAct(SequentialFlow):
    """Simply execute a given action."""

    def __init__(self, action_planner: PlanDSAction):
        """Initializes the SequentialFlow object with a sequence of commands or sub-flows, a name,
        and a description.

        Args:
            action_planner: a command that just plans the next DS action
        """
        super().__init__(
            name=action_planner.name,
            description=action_planner.description,
            sequence=[action_planner, ExecutePlannedAction()],
        )


class FixedDSActFromName(FixedDSAct):
    stage_name: DataScienceStageNames

    def __init__(self) -> None:
        aux_stage_name = self.stage_name

        class AuxPlanDSAction(NoHypsPlanDSAction):
            name: str = aux_stage_name.value
            stage_name: DataScienceStageNames = aux_stage_name

        action_planner = AuxPlanDSAction()
        super().__init__(action_planner=action_planner)


class AdoptDNNFramework(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.ADOPT_DNN_FRAMEWORK


class AddSubmissionAct(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.ADD_SUBMISSION


class AddDNNSubmissionAct(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.ADD_DNN_SUBMISSION


class AddClassicalSubmissionAct(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.ADD_CLASSICAL_TAB_SUBMISSION


class AdoptClassicalMLAct(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.ADOPT_CLASSICAL_ML_FRAMEWORK


class AddTrainCode(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.ADD_TRAIN_CODE


class TrainSubmissionCode(SequentialFlow):
    """Train and summarize the submission."""

    def __init__(
            self,
            summary_parse_func_id: str,
            overall_summary_template: str,
            max_retries: int = 5,
            human_takeover_step: int = 10,
    ):
        summary_cmd = SummarizeOverall(
            name="summarize_overall",
            description="Summarize the entire submission steps into one summary.",
            response_parser_id=summary_parse_func_id,
            required_prompt_templates={"ask_template": overall_summary_template},
            max_retries=max_retries,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0,
        )
        act_cmd = TrainSubmissionPlanDSAction()
        sequence = [summary_cmd, act_cmd, ExecutePlannedAction()]
        super().__init__(
            name=DataScienceStageNames.TRAIN_SUBMISSION.value,
            sequence=sequence,
            description="Summarize overall submission and train submission",
        )


class TabEmbedAct(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.DNN_TAB_EMBEDDING


class PlanTabFEAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TAB_FEATURE_ENGINEERING.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TAB_FEATURE_ENGINEERING


class FailedPlanTabFEAction(ThinkAndCodeFailedPlanDSAction):
    name: str = f"Failed {DataScienceStageNames.TAB_FEATURE_ENGINEERING.value}"
    stage_name: DataScienceStageNames = DataScienceStageNames.TAB_FEATURE_ENGINEERING


class PlanClassicalTabFEAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.CLASSICAL_TAB_FE.value
    stage_name: DataScienceStageNames = DataScienceStageNames.CLASSICAL_TAB_FE


class PlanTabPreprocessedEmbedAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TAB_EMBED_PREPROCESSED.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TAB_EMBED_PREPROCESSED


class PlanTabRegressionTargetScalerAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TABLE_REGRESSION_TARGET_TRANSFORM.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TABLE_REGRESSION_TARGET_TRANSFORM


class PlanTabHeadAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TABLE_HEAD.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TABLE_HEAD


class ImgEmbedAct(FixedDSActFromName):
    stage_name: DataScienceStageNames = DataScienceStageNames.IMAGE_EMBEDDING


class PlanTrainImgTransformAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TRAIN_IMAGE_TRANSFORM.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TRAIN_IMAGE_TRANSFORM


class PlanTestImgTransformAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TEST_IMAGE_TRANSFORM.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TEST_IMAGE_TRANSFORM


class PlanImgModelAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.IMAGE_MODELLING.value
    stage_name: DataScienceStageNames = DataScienceStageNames.IMAGE_MODELLING


class PlanTxtEmbedAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.TEXT_EMBEDDING.value
    stage_name: DataScienceStageNames = DataScienceStageNames.TEXT_EMBEDDING


class PlanBagAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.BAG_SUBMISSIONS.value
    stage_name: DataScienceStageNames = DataScienceStageNames.BAG_SUBMISSIONS


class PlanFEAction(ThinkAndCodePlanDSAction):
    name: str = FeatureEngineeringStageNames.CLASSICAL_TAB_FE.value
    stage_name: FeatureEngineeringStageNames = FeatureEngineeringStageNames.CLASSICAL_TAB_FE


class PlanModelTrainingAction(ThinkAndCodePlanDSAction):
    name: str = FeatureEngineeringStageNames.MODEL_TRAINING.value
    stage_name: FeatureEngineeringStageNames = FeatureEngineeringStageNames.MODEL_TRAINING


class PlanModelSelectBestAction(ThinkAndCodePlanDSAction):
    name: str = FeatureEngineeringStageNames.SELECT_BEST_MODEL.value
    stage_name: FeatureEngineeringStageNames = FeatureEngineeringStageNames.SELECT_BEST_MODEL


class PlanFEColumnTypesAction(ThinkAndCodePlanDSAction):
    name: str = FeatureEngineeringStageNames.GENERATE_TAB_COLUMN_TYPES.value
    stage_name: FeatureEngineeringStageNames = FeatureEngineeringStageNames.GENERATE_TAB_COLUMN_TYPES


class PlanClassImbalanceAction(ThinkAndCodePlanDSAction):
    name: str = DataScienceStageNames.CLASS_IMBALANCE.value
    stage_name: DataScienceStageNames = DataScienceStageNames.CLASS_IMBALANCE


class ReinitKeyCommand(DoNothing):
    def __init__(self, deleted_keys: list[MemKey], **data: Any):
        super().__init__(**data)
        self.deleted_keys = deleted_keys


class SummarizedThink(HumanTakeoverCommand):
    name: str = "summarize_think_cot"
    description: str = "Summarized Plan for Think in case using a reasoning model"

    required_prompt_templates: dict[str, str] = {}
    parse_func_id: str = "extract_plan_as_json"
    predefined_response: str | None = None

    def func(self, agent, *args: Any, **kwargs: Any):
        if agent.memory.retrieve(MemKey.SUMMARIZE_COT):
            parse_func = PARSE_FUNC_MAP["extract_summary_as_json"]
            format_error_message = ('Use the specified JSON structure:\n'
                                    '```json\n{\n\t"summary": "<summary>"\n}\n```')
        else:
            parse_func = lambda x: x
            format_error_message = ""
        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["ask_template"],
            parse_func=parse_func,
            format_error_message=format_error_message,
            prompt_kwargs={"memory": agent.memory},
            max_retries=5,
            human_takeover=True,
            predefined_response=self.predefined_response
        )
        agent.memory.store(response, MemKey.THOUGHT)


class ThinkAndCode(SequentialFlow, abc.ABC):
    SUCCESS_END: str = "SUCCESS END"
    FAILED_END: str = "FAILED END"

    def __init__(
            self,
            name: str,
            description: str,
            code_with_blanks: str,
            workspace_path: str,
            think_prompt_template: str,
            code_prompt_template: str,
            rag_for_think_template: str | None,
            rag_for_code_template: str | None,
            pre_loop_cmds: list[Command | Flow],
            loop_max_repetitions: int,
            flow_loop_template: str,
            choice_parse_func_id: str,
            summary_parse_func_id: str,
            summary_ask_template: str,
            allow_early_break: bool = True,
            max_retries: int = 5,
            human_takeover_step: int = 10,
            stop_after_success: bool = False,
            loop_max_repetitions_memkey_tag: MemKey | None = None,
            default_plan_path: str | None = None,
            default_code_path: str | None = None,
            default_summary_path: str | None = None,
            path_to_python: str = "./third_party/agent_k_python_path.txt",
    ) -> None:
        """
        Args:
            path_to_python: file containing path to python exe
            human_takeover_step: Number of loops to do for human to replace LLM in the loop
            stop_after_success: whether to let the Agent redo the coding even when it successfully generates one
            loop_max_repetitions_memkey_tag: if the loop around coding terminates due to too many retries, this
                memkey will contain `True`
        """
        self.workspace_path = workspace_path
        self.default_plan_path = default_plan_path
        self.default_code_path = default_code_path
        self.default_summary_path = default_summary_path
        self.code_with_blanks = code_with_blanks
        self.path_to_python = path_to_python
        self.code_prompt_template = code_prompt_template

        sequence = [ReinitKeyCommand(deleted_keys=[MemKey.RAG_RETRIEVAL])]
        sequence.extend(pre_loop_cmds)
        sequence.extend(self.get_pre_loop_cmds())

        # Delete keys used for managing the loop stopping criterion and success assessment
        keys_to_delete = [MemKey.THINK_AND_CODE_END_CHOICE, MemKey.CODE, MemKey.CODE_OUTPUT, MemKey.CODE_ERROR]

        if loop_max_repetitions_memkey_tag is not None:
            keys_to_delete.append(loop_max_repetitions_memkey_tag)
        sequence.append(DoNothing(deleted_keys=keys_to_delete))

        loop_body_seq = []

        # Think before code
        if rag_for_think_template is not None:  # get info from RAG for thinking
            loop_body_seq.append(UseTool(tool=RAG, prompt_template=rag_for_think_template))
        think_cmd = SummarizedThink(
            required_prompt_templates={"ask_template": think_prompt_template}, predefined_response=self.default_plan
        )
        loop_body_seq.append(think_cmd)

        # Code
        if rag_for_code_template is not None:  # get info from RAG for coding
            UseTool(tool=RAG, prompt_template=rag_for_code_template)
        code_tool = PythonInterpreterWithBlanks(
            path_to_python=path_to_python, code_with_blanks=code_with_blanks, workspace_path=self.workspace_path,
        )
        loop_body_seq.append(
            UseTool(tool=code_tool, prompt_template=code_prompt_template, predefined_response=self.default_code)
        )

        # Handle whether to stop after the code properly runs
        memory_choice_tag_val = None
        self.stop_after_success = stop_after_success
        if self.stop_after_success:
            code_passed_loop_choice = CodePassedLoopChoiceCmd(check_code_passed=self.check_code_passed)
            loop_body_seq.append(code_passed_loop_choice)
            memory_choice_tag_val = code_passed_loop_choice.choice_key
        loop_body = SequentialFlow(sequence=loop_body_seq)
        self.loop_flow = LoopFlow(
            loop_body=loop_body,
            max_repetitions=loop_max_repetitions,
            max_repetitions_memkey_tag=loop_max_repetitions_memkey_tag,
            max_retries=max_retries,
            prompt_template=flow_loop_template,
            name=name,
            description=description,
            parse_func_id=choice_parse_func_id,
            allow_early_break=allow_early_break,
            human_takeover_step=human_takeover_step,
            memory_choice_tag_val=memory_choice_tag_val
        )

        sequence.append(self.loop_flow)
        sequence.extend(self.get_post_loop_cmds())

        # At this point either the code is successfully generated or the loop terminated after the max_repetitions...
        # If the code was generated successfully: summarize and terminate
        # Otherwise: terminate with some signal

        check_loop_passed = lambda agent: self.is_code_passed(
            agent=agent, max_repetitions_memkey_tag=loop_max_repetitions_memkey_tag
        )
        set_end_flow_choices_cmd = CodePassedPostLoopChoiceCmd(
            success_end_choice=self.SUCCESS_END,
            fail_end_choice=self.FAILED_END,
            check_loop_passed=check_loop_passed
        )
        sequence.append(set_end_flow_choices_cmd)

        fail_end_flow = self.get_fail_end_flow()

        success_end_flow_seq: list[Flow | Command] = []
        # Add summarization
        summarize_cmd = SummarizeThinkAndCode(
            name="summarizer",
            description="Summarize the thinking and coding phase",
            response_parser_id=summary_parse_func_id,
            required_prompt_templates={"ask_template": summary_ask_template},
            max_retries=max_retries,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0,
            predefined_response=self.default_summary
        )
        success_end_flow_seq.append(summarize_cmd)
        success_end_flow_seq.extend(self.get_post_summary_cmds())

        success_end_flow_seq.append(FixedDSAct(action_planner=self.get_action_planner()))
        success_end_flow = SequentialFlow(sequence=success_end_flow_seq, name=self.SUCCESS_END)

        end_flow = DecisionFlow(
            choices=[success_end_flow, fail_end_flow],
            description="Depends whether the code was properly created",
            prompt_template=None,
            max_retries=1,
            memory_choice_tag_val=MemKey.THINK_AND_CODE_END_CHOICE
        )

        sequence.append(end_flow)
        super().__init__(name=name, description=description, sequence=sequence)

    @abc.abstractmethod
    def get_action_planner(self) -> PlanDSAction:
        raise NotImplementedError()

    @staticmethod
    def get_default_response(file_path: str | None) -> str:
        allow_default_response = False
        if str(os.getenv("ALLOW_DEFAULT_RESPONSE")) in ["True", "true", "1"]:
            allow_default_response = True
        if file_path is not None and allow_default_response:
            default_response = LLMAgent.get_llm_answers(read_answer_from_file_path=file_path)[0]  # get first element
        else:
            default_response = None
        return default_response

    @property
    def default_plan(self) -> str:
        return self.get_default_response(file_path=self.default_plan_path)

    @property
    def default_code(self) -> str:
        return self.get_default_response(file_path=self.default_code_path)

    @property
    def default_summary(self) -> str:
        return self.get_default_response(file_path=self.default_summary_path)

    def check_code_passed(self, agent: LLMAgent) -> None:
        """ Do custom check of code pass """
        return None

    def default_check_code_passed(self, agent: LLMAgent) -> None:
        code_error = agent.memory.retrieve(MemKey.CODE_ERROR)
        if code_error is None or code_error == "":
            code_passed = True
        else:
            code_passed = False
        agent.memory.store(code_passed, MemKey.CODE_PASSED)

    def get_post_summary_cmds(self) -> list[Flow | Command]:
        """ Add commands after the summarization step """
        return []

    def get_pre_loop_cmds(self) -> list[Flow | Command]:
        """ Add commands before the think and code loop starts """
        return []

    def get_fail_end_flow(self) -> Flow | Command | None:
        """ Flow that should be run if the agent fail to generate code """

        class FailedEndCommand(Command):
            name: str = self.FAILED_END
            description: str = "Do nothing"

        return FailedEndCommand()

    def is_code_passed(self, agent: LLMAgent, max_repetitions_memkey_tag: MemKey | None) -> bool:
        if self.stop_after_success:
            loop_choice = agent.memory.retrieve(MemKey.LOOP_CODE_PASSED_CHOICE)[0]
            if loop_choice == LoopFlow.TERMINATE_LOOP_CHOICE:
                return True
            elif loop_choice == LoopFlow.CONTINUE_LOOP_CHOICE:
                return False
            else:
                raise ValueError(loop_choice)

        if max_repetitions_memkey_tag is not None:
            max_repetition_hit = agent.memory.retrieve(max_repetitions_memkey_tag)
            return not max_repetition_hit  # if max repetitions was hit, we consider it a failure
        return True

    def get_post_loop_cmds(self) -> list[Command | Flow]:
        """ Add commands after the think and code loop ends """
        return []


class SelectSubmission(Command):
    """Asks the LLM to select a submission out of a list of submissions."""

    response_parser_id: str = "extract_submission_as_json"
    output_keys: dict[str, MemKey] = {"submission_name": MemKey.SELECTED_SUBMISSION}

    def func(self, agent: LLMAgent, ask_template: str, submission_name: MemKey):
        choices = [sub.name for sub in agent.memory.retrieve(MemKey.SUBMISSION_LIST)]
        prompt_kwargs = {"memory": agent.memory, "submissions": agent.memory.retrieve(tags=MemKey.SUBMISSION_LIST)}
        response = agent.safe_choose_from_options(
            ask_template=ask_template,
            parse_func=extract_submission_as_json,
            format_error_message=('Use the specified JSON structure:\n'
                                  '```json\n{\n\t"submission": "<submission_choice>"\n}\n```'),
            options=choices,
            prompt_kwargs=prompt_kwargs,
            max_retries=5,
            human_takeover=True
        )
        agent.memory.store(response, submission_name)


class StoreTabTrainInMemory(Command):
    """Keep a view of table in the memory."""

    name: str = "store_tab_train"
    description: str = " Store example rows of tab train map in memory"

    output_keys: dict[str, MemKey] = {memkey.value: memkey for memkey in
                                      [MemKey.TAB_TRAIN_COLUMNS, MemKey.TAB_TRAIN_INFO, MemKey.ACTIVE_TABLE_VIEW]}

    view_n_rows: int = 5

    @staticmethod
    def get_summary_stats(df: pd.DataFrame, max_columns: int = 100, unique_threshold: int = 30) -> pd.DataFrame:
        """ Generates summary statistics for a DataFrame with optional filtering for large datasets."""

        summary = {}

        columns = df.columns

        for col in columns:
            if df[col].nunique() <= unique_threshold:
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary[col] = {
                        'mean': df[col].mean(),
                        'std_dev': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'unique_count': df[col].nunique(),
                        'most_common': df[col].mode()[0] if not df[col].mode().empty else None,
                        'number of missing values': df[col].isnull().sum()
                    }
                else:
                    summary[col] = {
                        'unique_count': df[col].nunique(),
                        'most_common': df[col].mode()[0] if not df[col].mode().empty else None,
                        'number of missing values': df[col].isnull().sum()
                    }

        # Limit the summary output to max_columns
        summary_df = pd.DataFrame(summary)
        if len(summary_df.columns) > max_columns:
            summary_df = summary_df.iloc[:, :max_columns]

        return summary_df

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        workspace_path: str | bytes = os.path.join(
            agent.task.prepared_setup_dir, agent.task.task_id,
            agent.task.prepared_version
        )
        df_tab_train_path = os.path.join(workspace_path, 'train_tab_input_map.csv')
        df_tab_train_target_path = os.path.join(workspace_path, 'train_tab_target_map.csv')
        df_train = pd.read_csv(df_tab_train_path, index_col="id")
        df_target = pd.read_csv(df_tab_train_target_path, index_col="id")
        if len(df_train.columns) > 50:
            self.view_n_rows = 2
        random_sample = df_train.sample(n=self.view_n_rows, random_state=42)
        df_summary = self.get_summary_stats(df_train)

        active_table_view = ''
        active_table_view += f'#### Shape of the training data: {df_train.shape}\n'
        active_table_view += f'#### Summary of the training data:\n'
        active_table_view += df_summary.to_string()
        active_table_view += '\n'
        active_table_view += f'#### View of training data :\n'
        active_table_view += random_sample.to_string()
        active_table_view += '\n'
        active_table_view += f'#### View of target data:\n'
        active_table_view += df_target.sample(n=self.view_n_rows, random_state=42).to_string()
        active_table_view += '\n\n'
        agent.memory.store(active_table_view, self.output_keys[MemKey.ACTIVE_TABLE_VIEW.value])


class SendSubmissionPlanDSAction(PlanDSAction):
    name: str = "send_submission"
    stage_name: DataScienceStageNames = DataScienceStageNames.SEND_SUBMISSION
    description: str = "Send selected submission file"

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {DSActionHypKeys.SUBMISSION_NAME: memory.retrieve({MemKey.SELECTED_SUBMISSION: 1.0})}


class SendSubmission(SequentialFlow):
    """Ask the LLM to choose which submission file to submit and then acts to submit to kaggle."""

    def __init__(
            self,
            parse_func_id: str,
            select_submission_template: str,
    ):
        select_cmd = SelectSubmission(
            name="select_submission",
            description="Select the submission to send out of a list of already created submissions",
            response_parser_id=parse_func_id,
            required_prompt_templates={"ask_template": select_submission_template},
        )
        act_cmd = SendSubmissionPlanDSAction()
        sequence = [select_cmd, act_cmd, ExecutePlannedAction()]
        super().__init__(
            name=DataScienceStageNames.SEND_SUBMISSION.value,
            sequence=sequence,
            description="Select and send submission",
        )


class ModifySubmissionStatusCmd(Command):
    """Modify the current submission status."""

    name: str = "StatusModifyer"
    description: str = "Modify current submission status"

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        tag = agent.memory.mem_keys.CURRENT_SUBMISSION
        current_submission = agent.memory.retrieve(tags={tag: 1.0})
        self.modify_current_submission(current_submission=current_submission)
        agent.memory.store(current_submission, tags=tag)

    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        pass


class TabFEStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.table_fe.is_pending = True


class ClassicalTabFEStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, ClassicalTabSubmissionState)
        current_submission.submission_state.table_fe.is_pending = True


class TabEmbedPreprocessedStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.table_model.is_pending = True


class TabRegressionTargetScalerStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.regression_target_transform.is_pending = True


class TabHeadStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.table_head.is_pending = True


class TabularModelStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.table_model.is_pending = True


class ImgModelStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.img_model.is_pending = True


class TrainImgTransformStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.train_img_transform.is_pending = True


class TestImgTransformStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.test_img_transform.is_pending = True


class TxtEmbedStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.txt_embed.is_pending = True


class BagStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        pass


class ClassImbalanceStatusCmd(ModifySubmissionStatusCmd):
    def modify_current_submission(self, current_submission: DataScienceSubmissionCard) -> None:
        assert isinstance(current_submission.submission_state, DNNSubmissionState)
        current_submission.submission_state.class_imbalance.is_pending = True


class MultiTrialThinkAndCode(Flow):

    def __init__(self, name: str, description: str, think_and_code_flow: ThinkAndCode, max_outer_repetitions: int):
        super().__init__(name, description)
        self.name = name
        self.description = description
        self.seq_flow_body = self.get_pre_loop_cmds()

        self.loop_body_seq = [think_and_code_flow]
        self.loop_body_seq.append(
            MultiTrialPostLoopChoiceCmd(success_message=ThinkAndCode.SUCCESS_END, fail_message=ThinkAndCode.FAILED_END)
        )  # put in the memory whether to continue or terminate
        self.loop_body = SequentialFlow(self.loop_body_seq)
        self.loop_flow = LoopFlow(
            loop_body=self.loop_body,
            max_repetitions=max_outer_repetitions,
            allow_early_break=True,
            max_retries=1,
            prompt_template=None,
            memory_choice_tag_val=MemKey.MULTI_TRIAL_THINK_AND_CODE_LOOP_CHOICE
        )
        self.seq_flow_body.append(self.loop_flow)
        self.seq_flow = SequentialFlow(self.seq_flow_body)

    def reset(self) -> None:
        """Resets the flow to its initial state.

        This method should be overridden by subclasses.
        """
        self.seq_flow.reset()

    def step(self, agent: Agent) -> Command | Flow | None:
        """Executes a step in the flow and moves towards the next command or sub-flow.

        Args:
            agent: An agent that carries out the commands or sub-flows.

        This method should be overridden by subclasses.
        """
        return self.seq_flow.step(agent=agent)

    def get_pre_loop_cmds(self) -> list[Flow | Command]:
        """ Add commands before the think and code loop starts """
        return []


class ThinkAndCodeDNNTabFE(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TAB_FEATURE_ENGINEERING.value,
        description="Pre-process tabular features.",
        loop_max_repetitions=35,
        pre_loop_cmds=[],
        loop_max_repetitions_memkey_tag=MemKey.THINK_AND_CODE_MAX_REPETITIONS,
        stop_after_success=True
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanTabFEAction()

    def check_code_passed(self, agent: LLMAgent) -> None:
        code_output = agent.memory.retrieve(MemKey.CODE_OUTPUT)
        if code_output is None:
            code_passed = False
        else:
            if "Number of features: " in "".join(code_output.split("\n")[-2:]):
                code_passed = True
            else:
                code_passed = False
        agent.memory.store(code_passed, MemKey.CODE_PASSED)

    def get_post_loop_cmds(self) -> list[Flow | Command]:
        """ Flow after main loop """

        class PostLoopCommand(Command):
            path_to_python: str
            code_with_blanks: str
            workspace_path: str
            code_prompt_template: str
            check_code_passed: Callable
            name: str = "post_loop_command"
            description: str = "Post Loop Command"

            def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
                code_interpreter_command = AlternateCodeCommand(code_memkey=MemKey.CODE_FE)
                code_interpreter_command(agent=agent)
                code_response = code_interpreter_command.available_code
                if code_response is not None:
                    code_tool = PythonInterpreterWithBlanks(
                        path_to_python=self.path_to_python, code_with_blanks=self.code_with_blanks,
                        workspace_path=self.workspace_path,
                    )
                    tool = UseTool(
                        tool=code_tool, prompt_template=self.code_prompt_template,
                        predefined_response=code_response
                    )
                    tool(agent=agent)
                    code_passed_loop_choice = CodePassedLoopChoiceCmd(check_code_passed=self.check_code_passed)
                    code_passed_loop_choice(agent=agent)

        return [
            PostLoopCommand(
                path_to_python=self.path_to_python, code_with_blanks=self.code_with_blanks,
                workspace_path=self.workspace_path, code_prompt_template=self.code_prompt_template,
                check_code_passed=self.check_code_passed
            )
        ]


class MultiTrialThinkAndCodeDNNTabFE(MultiTrialThinkAndCode):
    memory = Memory()
    outer_repetition = 1 if memory.retrieve(MemKey.CODE_FE) else 3
    __init__ = partialmethod(
        MultiTrialThinkAndCode.__init__,
        name=DataScienceStageNames.TAB_FEATURE_ENGINEERING.value,
        description="Pre-process tabular features.",
        max_outer_repetitions=outer_repetition
    )

    def get_pre_loop_cmds(self) -> list[Flow | Command]:
        """ Add commands before the think and code loop starts """
        return [StoreTabTrainInMemory(), TabFEStatusCmd()]


class ThinkAndCodeClassicalTabFE(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.CLASSICAL_TAB_FE.value,
        description="Classical ML tabular features engineering.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[StoreTabTrainInMemory(), ClassicalTabFEStatusCmd()],
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanClassicalTabFEAction()


class ThinkAndCodeFE(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=FeatureEngineeringStageNames.CLASSICAL_TAB_FE.value,
        description="Dedicated features engineering.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[StoreTabTrainInMemory()],
        stop_after_success=True
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanFEAction()

    def check_code_passed(self, agent: LLMAgent) -> None:
        code_output = agent.memory.retrieve(MemKey.CODE_OUTPUT)
        if code_output is None:
            code_passed = False
        else:
            if "Feature engineered data successfully saved" in "".join(code_output.split("\n")[-2:]):
                code_passed = True
            else:
                code_passed = False
        agent.memory.store(code_passed, MemKey.CODE_PASSED)


class ThinkAndCodeTrain(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=FeatureEngineeringStageNames.MODEL_TRAINING.value,
        description="Model training.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[],
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanModelTrainingAction()


class ThinkAndCodeSelectBest(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=FeatureEngineeringStageNames.SELECT_BEST_MODEL.value,
        description="Select best model performance.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[],
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanModelSelectBestAction()


class ThinkAndCodeColumnTypes(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=FeatureEngineeringStageNames.GENERATE_TAB_COLUMN_TYPES.value,
        description="Column type generation.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[StoreTabTrainInMemory()],
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanFEColumnTypesAction()


class ThinkAndCodeTabPreprocessedEmbed(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TAB_EMBED_PREPROCESSED.value,
        description="Implement an embedder of the pre-processed tabular features.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[TabEmbedPreprocessedStatusCmd()],
        stop_after_success=True,
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanTabPreprocessedEmbedAction()

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)


class GetTabRegressionInfo(Command):
    """Keep a view of table in the memory."""
    workspace_path: str
    name: str = "store_regression_table_info"
    description: str = " Store example rows and stats of regression targets in memory"

    output_keys: dict[str, MemKey] = {memkey.value: memkey for memkey in
                                      [MemKey.REGRESSION_TARGET_VIEW, MemKey.REGRESSION_TARGET_STATS]}
    view_n_rows: int = 5
    max_columns: int = 20
    width: int = 300

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        set_pd_options(max_rows=None, max_columns=self.max_columns, float_format="{:20,.2f}".format, width=self.width)
        tab_target_path = os.path.join(self.workspace_path, FileMap.TRAIN_TABULAR_TARGET.value)
        tab_target_df = pd.read_csv(tab_target_path, index_col="id")
        regression_df = tab_target_df[[c for c in tab_target_df.columns if c.endswith("_regression")]]

        assert regression_df.shape[-1] > 0, regression_df
        random_inds = np.random.choice(
            np.arange(len(regression_df)), size=min(self.view_n_rows, len(regression_df)), replace=False
        )
        sample_rows = regression_df.iloc[random_inds]
        table_view = f"```Shape {regression_df.shape}\n{sample_rows}\n```"
        stats_view = f"```{get_df_stats(df=regression_df, sorted_by='std')}\n```"
        agent.memory.store(content=table_view, tags=self.output_keys[MemKey.REGRESSION_TARGET_VIEW.value])
        agent.memory.store(content=stats_view, tags=self.output_keys[MemKey.REGRESSION_TARGET_STATS.value])
        reset_pd_options()


class ThinkAndCodeTabRegressionTargetTransform(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TABLE_REGRESSION_TARGET_TRANSFORM.value,
        description="Implement regression target transform and inverse transform.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[],
        stop_after_success=True,
    )

    def check_code_passed(self, agent: LLMAgent) -> None:
        code_output = agent.memory.retrieve(MemKey.CODE_OUTPUT)
        if code_output is None:
            code_passed = False
        else:
            if "Could perform fit and transform" in "".join(code_output.split("\n")):
                code_passed = True
            else:
                code_passed = False
        agent.memory.store(code_passed, MemKey.CODE_PASSED)

    def get_action_planner(self) -> PlanDSAction:
        return PlanTabRegressionTargetScalerAction()

    def get_post_summary_cmds(self) -> list[Flow | Command]:
        class DeleteRegressionTableViewKey(Command):
            name: str = "delete_regression_tab_view_key"
            description: str = "Delete keys not needed anymore"
            deleted_keys: Annotated[
                list[MemKey], "Memory keys that will be deleted. Deletion happens at the end of the function call."
            ] = [MemKey.REGRESSION_TARGET_VIEW, MemKey.REGRESSION_TARGET_STATS]

        return [DeleteRegressionTableViewKey()]

    def get_pre_loop_cmds(self) -> list[Flow | Command]:
        return [TabRegressionTargetScalerStatusCmd(), GetTabRegressionInfo(workspace_path=self.workspace_path)]


class ThinkAndCodeTabHead(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TABLE_HEAD.value,
        description="Implement a tabular decoder and the associated losses.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[TabHeadStatusCmd()],
        stop_after_success=True
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanTabHeadAction()

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)


class ThinkAndCodeImgModelling(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.IMAGE_MODELLING.value,
        description="Implement an embedder of the images.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[ImgModelStatusCmd()],
        stop_after_success=True
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanImgModelAction()

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)


class ThinkAndCodeImgTransform(ThinkAndCode, ABC):

    def check_code_passed(self, agent: LLMAgent) -> None:
        code_output = agent.memory.retrieve(MemKey.CODE_OUTPUT)
        if code_output is None:
            code_passed = False
        else:
            if "Transformed image tensor shape:" in "".join(code_output.split("\n")):
                code_passed = True
            else:
                code_passed = False
        agent.memory.store(code_passed, MemKey.CODE_PASSED)


class GetDatasetStatisticsCmd(HumanTakeoverCommand):
    name: str = "get_dataset_statistics"
    description: str = "Get dataset statistics for image transforms"
    output_keys: dict[str, MemKey] = {"img_data_statistics": MemKey.IMG_DATA_STATISTICS}
    max_retries: int = 5
    human_takeover: bool = False
    max_workers: int = 32
    rescale_channels: bool = True

    @staticmethod
    def open_and_calculate_stats(path, rescale=True) -> np.array:
        with Image.open(path) as img:
            # width, height = img.size
            # total_pixels = np.product(img.size)
            if np.array(img).shape[-1] > 3:
                img = img.convert('RGB')

            s = ImageStat.Stat(img)
            mean_per_channel = s.mean
            var_per_channel = s.var

            if rescale:
                mean_per_channel = np.array(mean_per_channel) / 255
                # Variance scales quadratically with input
                var_per_channel = np.array(mean_per_channel) / 65025  # (255^2)

            if len(mean_per_channel) == 1:
                mean_per_channel = np.repeat(mean_per_channel, 3)
            if len(var_per_channel) == 1:
                var_per_channel = np.repeat(var_per_channel, 3)

            return np.concatenate([np.array(img.size), mean_per_channel, var_per_channel])

    def format_stats(self, stats: Iterable[float]) -> str:
        """Trim significant figures and convert to string"""
        stats_rounded = [np.format_float_positional(v, precision=3, fractional=False, trim="-") for v in stats]
        return f"[{', '.join(stats_rounded)}]"

    def func(self, agent: LLMAgent, img_data_statistics: MemKey):
        saved_img_data_statistics = agent.memory.retrieve(MemKey.IMG_DATA_STATISTICS)
        if saved_img_data_statistics is None:
            prepared_setup_dir = Path(agent.task.env.get_src_path())
            train_img_input_map_path = prepared_setup_dir / FileMap.TRAIN_IMAGE_INPUT.value

            df = pd.read_csv(train_img_input_map_path, index_col=0)
            if self.max_workers == 0:
                print("Calculating image statistics (single-threaded)...")
                agg_df = df.map(self.open_and_calculate_stats)
                stat_sum = agg_df.to_numpy().sum()
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    stat_sum = np.zeros(2 + 3 + 3)

                    for stats in tqdm(
                            executor.map(
                                partial(self.open_and_calculate_stats, rescale=self.rescale_channels),
                                itertools.chain.from_iterable(df.itertuples(index=False))
                            ), desc="Calculating image statistics (multi-threaded)", total=df.size
                    ):
                        stat_sum += stats

            stat_mean = stat_sum / df.size

            size, mean_per_channel, var_per_channel = np.split(stat_mean, [2, 5])
            std_per_channel = np.sqrt(var_per_channel)

            stats_dict = {
                "mean per channel": self.format_stats(mean_per_channel),
                "standard deviation per channel": self.format_stats(std_per_channel),
                "mean pixel resolution [width, height]": self.format_stats(size),
            }
            agent.memory.store(stats_dict, img_data_statistics)


class ThinkAndCodeTrainImgTransform(ThinkAndCodeImgTransform):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TRAIN_IMAGE_TRANSFORM.value,
        description="Implement transform for train images.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[TrainImgTransformStatusCmd(), GetDatasetStatisticsCmd(max_workers=0)],
        stop_after_success=True,
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanTrainImgTransformAction()

    def get_post_loop_cmds(self) -> list[Command | Flow]:
        """
        Save train image transform code
        """
        return [MemKeyCopy(original_key=MemKey.CODE_BLANK, copy_key=MemKey.CODE_TRAIN_IMG_TRANSFORM)]

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)


class ThinkAndCodeTestImgTransform(ThinkAndCodeImgTransform):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TEST_IMAGE_TRANSFORM.value,
        description="Implement transform for test images.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[TestImgTransformStatusCmd()],
        stop_after_success=True,
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanTestImgTransformAction()

    def get_post_loop_cmds(self) -> list[Command | Flow]:
        """
        Delete saved train image transform code
        """
        keys_to_delete = [MemKey.CODE_TRAIN_IMG_TRANSFORM]
        return [DoNothing(deleted_keys=keys_to_delete)]

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)


class ThinkAndCodeTextEmbed(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.TEXT_EMBEDDING.value,
        description="Implement an embedder of the text.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[TxtEmbedStatusCmd()],
        stop_after_success=True
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanTxtEmbedAction()

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)


class SelectBagSubmissions(Command):
    """Asks the LLM to select a submission out of a list of submissions."""

    response_parser_id: str = "extract_submission_as_json"
    output_keys: dict[str, MemKey] = {"selected_submissions_bag": MemKey.SELECTED_SUBMISSIONS_BAG}

    def func(self, agent: LLMAgent, ask_template: str, selected_submissions_bag: MemKey):
        choices = [sub.name for sub in agent.memory.retrieve(MemKey.SUBMISSION_LIST)]
        prompt_kwargs = {"memory": agent.memory, "submissions": agent.memory.retrieve(tags=MemKey.SUBMISSION_LIST)}
        response = agent.safe_choose_from_options(
            ask_template=ask_template,
            parse_func=extract_submission_as_json,
            format_error_message=('Use the specified JSON structure:\n'
                                  '```json\n{\n\t"submission": "<submission_choice_1>, <submission_choice_2>"\n}\n```'),
            options=choices,
            prompt_kwargs=prompt_kwargs,
            max_retries=5,
            human_takeover=True
        )
        agent.memory.store(response, selected_submissions_bag)


class SelectBlendSubmissions(Command):
    """Asks the LLM to select a submission out of a list of submissions."""

    response_parser_id: str = "extract_submission_as_json"
    output_keys: dict[str, MemKey] = {MemKey.SELECTED_SUBMISSIONS_BLEND.value: MemKey.SELECTED_SUBMISSIONS_BLEND}

    def func(self, agent: LLMAgent, ask_template: str, selected_submissions_blend: MemKey):
        choices = [sub.name for sub in agent.memory.retrieve(MemKey.SUBMISSION_LIST)]
        prompt_kwargs = {"memory": agent.memory, "submissions": agent.memory.retrieve(tags=MemKey.SUBMISSION_LIST)}
        response = agent.safe_parsing_chat_completion(
            ask_template=ask_template,
            parse_func=extract_submission_as_json,
            format_error_message=('Use the specified JSON structure:\n'
                                  '```json\n{\n\t"submission": "<submission_choice_1>, <submission_choice_2>"\n}\n```'),
            # options=choices,
            prompt_kwargs=prompt_kwargs,
            max_retries=5,
            human_takeover=True
        )
        agent.memory.store(response, selected_submissions_blend)


class SendBlendSubmissionPlanDSAction(PlanDSAction):
    name: str = "send_blend_submission"
    stage_name: DataScienceStageNames = DataScienceStageNames.BLEND_SUBMISSIONS
    description: str = "Send selected submissions files"

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {DSActionHypKeys.SUBMISSIONS_LIST: memory.retrieve({MemKey.SELECTED_SUBMISSIONS_BLEND: 1.0})}


class SendBlendSubmissions(SequentialFlow):
    """Ask the LLM to choose which submission file to submit and then acts to submit to kaggle."""

    def __init__(
            self,
            parse_func_id: str,
            select_submission_template: str,
    ):
        select_cmd = SelectBlendSubmissions(
            name="select_submission",
            description="Select the submission to send out of a list of already created submissions",
            parse_func_id=parse_func_id,
            required_prompt_templates={"ask_template": select_submission_template},
        )
        act_cmd = SendBlendSubmissionPlanDSAction()
        sequence = [select_cmd, act_cmd, ExecutePlannedAction()]
        super().__init__(
            name=DataScienceStageNames.BLEND_SUBMISSIONS.value,
            sequence=sequence,
            description="Select and send submissions for blend",
        )


class SendBagSubmissionPlanDSAction(PlanDSAction):
    name: str = "send_bag_submission"
    stage_name: DataScienceStageNames = DataScienceStageNames.BAG_SUBMISSIONS
    description: str = "Send selected submissions files"

    def get_hyps(self, memory: Memory) -> dict[DSActionHypKeys, Any]:
        return {
            DSActionHypKeys.SUBMISSIONS_LIST: memory.retrieve({MemKey.SELECTED_SUBMISSIONS_BAG: 1.0}),
            DSActionHypKeys.SUMMARY_STEP: memory.retrieve(tags=MemKey.SUMMARY_STEP),
            DSActionHypKeys.CODE_BLANK: memory.retrieve(tags=MemKey.CODE_BLANK),
        }


class SendBagSubmissions(SequentialFlow):
    """Ask the LLM to choose which submission file to submit and then acts to submit to kaggle."""

    def __init__(
            self,
            parse_func_id: str,
            select_submission_template: str,
            workspace_path: str,
            flow_loop_template: str,
            choice_parse_func_id: str,
            rag_for_think_template: str,
            think_prompt_template: str,
            rag_for_code_template: str,
            code_prompt_template: str,
            code_with_blanks: str,
            summary_ask_template: str,
            summary_parse_func_id: str
    ):
        select_cmd = SelectBagSubmissions(
            name="select_submission",
            description="Select the submission to send out of a list of already created submissions",
            parse_func_id=parse_func_id,
            required_prompt_templates={"ask_template": select_submission_template},
        )
        gen_code = ThinkAndCodeBag(
            workspace_path=workspace_path,
            flow_loop_template=flow_loop_template,
            choice_parse_func_id=choice_parse_func_id,
            rag_for_think_template=rag_for_think_template,
            think_prompt_template=think_prompt_template,
            rag_for_code_template=rag_for_code_template,
            code_prompt_template=code_prompt_template,
            code_with_blanks=code_with_blanks,
            summary_ask_template=summary_ask_template,
            summary_parse_func_id=summary_parse_func_id
        )
        act_cmd = SendBagSubmissionPlanDSAction()
        sequence = [select_cmd, gen_code, act_cmd, ExecutePlannedAction()]
        super().__init__(
            name=DataScienceStageNames.BAG_SUBMISSIONS.value,
            sequence=sequence,
            description="Select and send submissions for bag",
        )


class ThinkAndCodeBag(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.GEN_BAG_CODE.value,
        description="Implement a function to bag submissions.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[BagStatusCmd()],
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanBagAction()


class ThinkAndCodeClassImbalance(ThinkAndCode):
    __init__ = partialmethod(
        ThinkAndCode.__init__,
        name=DataScienceStageNames.CLASS_IMBALANCE.value,
        description="Handle class imbalance.",
        loop_max_repetitions=-1,
        pre_loop_cmds=[ClassImbalanceStatusCmd()],
        stop_after_success=True
    )

    def get_action_planner(self) -> PlanDSAction:
        return PlanClassImbalanceAction()

    @staticmethod
    def get_default_response(file_path: str | None) -> str:
        allow_ci_handling = False
        allow_default_response = False
        if str(os.getenv("ALLOW_DEFAULT_RESPONSE")) in ["True", "true", "1"]:
            allow_default_response = True
        if str(os.getenv("USE_CI_HANDLING")) in ["True", "true", "1"]:
            allow_ci_handling = True

        if allow_default_response or not allow_ci_handling:
            default_response = LLMAgent.get_llm_answers(read_answer_from_file_path=file_path)[0]  # get first element
        else:
            default_response = None

        return default_response

    @property
    def default_plan(self) -> str:
        return self.get_default_response(file_path=self.default_plan_path)

    @property
    def default_code(self) -> str:
        return self.get_default_response(file_path=self.default_code_path)

    @property
    def default_summary(self) -> str:
        return self.get_default_response(file_path=self.default_summary_path)

    def check_code_passed(self, agent: LLMAgent) -> None:
        return self.default_check_code_passed(agent)

import abc
import io
import json
import os
from functools import partial
from typing import Any, Tuple

import pandas as pd

from agent.agents import LLMAgent, safe_parsing_chat_completion
from agent.commands import ExecutePlannedAction, Flow, LoopFlow, SequentialFlow, Think
from agent.commands.core import Command, HumanTakeoverCommand, UseTool
from agent.commands.flows import PARSE_FUNC_MAP
from agent.commands.utils_commands import CodePassedLoopChoiceCmd
from agent.memory import MemKey, Memory
from agent.parsers.parser import ParsingError
from agent.tasks.data_preprocessing import CodeTemplateKeys
from agent.tasks.datascience_task.utils import FileMap
from agent.tools.fetch_tool import FetchTool
from agent.utils.utils import (
    catch_error_wrap,
    check_code_safety,
    extract_column_names_and_values_as_json,
    extract_json,
    extract_json_with_bools,
    extract_python,
    run_python_code,
)
from ds_agent.rag import DB_FAISS, VectorFaissDB
from third_party.data_preprocessing.env import (
    DataPrepAction,
    DataPrepColumnTypesStageName,
    DataPrepCreateStageName,
    DataPrepMetricStageName,
    DataPrepPlan,
    DataPrepPositiveClassStageName,
    DataPrepStageParam,
    DataPrepSubmissionFormatStageName,
    DataPrepSubmissionFormatAltStageName,
    DataPrepTestStageName,
    DataPrepStageName
)
from third_party.data_science.utils import get_raw_data_root_dir, recursive_chmod


class PlanDPAction(Command, abc.ABC):
    name: str
    stage_name: DataPrepStageName
    description: str = "Set planned action to a fix value (no call to LLM)"
    output_keys: dict[str, MemKey] = {MemKey.NEXT_PLANNED_ACTION.value: MemKey.NEXT_PLANNED_ACTION}

    def func(self, agent, *args, **kwargs) -> None:
        agent.memory.store(
            content=DataPrepAction(
                stage_name=self.stage_name,
                params=self.get_params(agent_memory=agent.memory)
            ),
            tags={self.output_keys[MemKey.NEXT_PLANNED_ACTION.value]},
        )

    @abc.abstractmethod
    def get_params(self, agent_memory: Memory) -> dict[DataPrepStageParam, Any]:
        raise NotImplementedError()


class NoParamsPlanDPAction(PlanDPAction):
    stage_name: DataPrepStageName
    name: str = "no_params_data_prep_action"
    description: str = "Do nothing"

    def get_params(self, agent_memory: Memory) -> dict[DataPrepStageParam, Any]:
        return {}


class ThinkAndCodePlanDPAction(PlanDPAction):
    stage_name: DataPrepStageName
    name: str = "think_code_and_summarize_planned_data_prep_action"
    description: str = "Create and store DataPrepAction after Think, Code and Summarize."

    deleted_keys: list[MemKey] = [
        MemKey.CODE, MemKey.CODE_OUTPUT, MemKey.CODE_ERROR, MemKey.CODE_RAN,
        MemKey.CODE_PASSED, MemKey.ACTIVE_TABLE_VIEW
    ]

    def get_params(self, agent_memory: Memory) -> dict[DataPrepStageParam, Any]:
        return {
            DataPrepStageParam.CODE: agent_memory.retrieve(agent_memory.mem_keys.CODE),
            DataPrepStageParam.CODE_OUTPUT: agent_memory.retrieve(agent_memory.mem_keys.CODE_OUTPUT),
            DataPrepStageParam.CODE_SUMMARY: agent_memory.retrieve(agent_memory.mem_keys.CODE_SUMMARY),
        }


class RAMPMetricPlanDPAction(PlanDPAction):
    stage_name: DataPrepMetricStageName
    name: str = "think_code_and_summarize_planned_data_prep_action"
    description: str = "Create and store DataPrepAction after Think, Code and Summarize."

    def get_params(self, agent_memory: Memory) -> dict[DataPrepStageParam, Any]:
        return {DataPrepStageParam.RAMP_METRIC_SELECTED: True}


class RAMPMetricActionFlow(SequentialFlow):
    """ Simply returns an action to the environment, no interaction with the Agent """

    def __init__(self) -> None:
        stage_name = DataPrepMetricStageName()
        super().__init__(
            sequence=[RAMPMetricPlanDPAction(stage_name=stage_name), ExecutePlannedAction()],
            name=stage_name.to_str()
        )


class StageGroupUnitTestPlanDPAction(PlanDPAction):
    stage_name: DataPrepTestStageName
    name: str = "stage_group_unit_test_planned_data_prep_action"
    description: str = "Run unit test associated with a group of stages"

    def get_params(self, agent_memory: Memory) -> dict[DataPrepStageParam, Any]:
        return {}


class StageGroupUnitTestActionFlow(SequentialFlow):
    """ Simply returns an action to the environment, no interaction with the Agent """

    def __init__(
            self,
            split: str,
            modality: str = None,
            map: bool = False,
            transform: bool = False,
            dataloader: bool = False,
    ):
        stage_name = DataPrepTestStageName(
            split=split,
            modality=modality,
            map=map,
            transform=transform,
            dataloader=dataloader
        )
        super().__init__(
            sequence=[StageGroupUnitTestPlanDPAction(stage_name=stage_name), ExecutePlannedAction()],
            name=stage_name.to_str()
        )


class ColTypesPassedLoopChoiceCmd(HumanTakeoverCommand):
    name: str = "col_types_pass_loop_choice"
    description: str = "Add choice to terminate or continue depending on the col types pass status."

    input_keys: dict[str, MemKey] = {
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED
    }
    output_keys: dict[str, MemKey] = {MemKey.LOOP_CODE_PASSED_CHOICE.value: MemKey.LOOP_CODE_PASSED_CHOICE}

    max_steps: int = 1
    current_step: int = 0

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        """ Put 'Continue' or 'Terminate' in choices depending on the status of the code """
        code_passed = agent.memory.retrieve(self.input_keys[MemKey.CODE_PASSED.value])
        assert code_passed is not None
        if code_passed:
            if self.current_step < self.max_steps:
                choices = [LoopFlow.CONTINUE_LOOP_CHOICE, LoopFlow.TERMINATE_LOOP_CHOICE]
                self.current_step += 1
            else:
                choices = [LoopFlow.TERMINATE_LOOP_CHOICE]
                self.current_step = 0
        else:
            choices = [LoopFlow.CONTINUE_LOOP_CHOICE]
        agent.memory.store(content=choices, tags=self.output_keys[MemKey.LOOP_CODE_PASSED_CHOICE.value])

    @property
    def choice_key(self) -> MemKey:
        return list(self.output_keys.values())[0]


class CommentColTypesCmd(HumanTakeoverCommand):
    name: str = "comment_on_generated_column_types"
    description: str = "Crate a comment of the generated column types."

    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.CODE.value: MemKey.CODE,
        MemKey.CODE_OUTPUT.value: MemKey.CODE_OUTPUT,
        MemKey.CODE_ERROR.value: MemKey.CODE_ERROR,
        MemKey.CODE_RAN.value: MemKey.CODE_RAN,
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED,
        MemKey.VALID_COLUMN_TYPES.value: MemKey.VALID_COLUMN_TYPES
    }
    output_keys: dict[str, MemKey] = {MemKey.COMMENT_COLUMN_TYPES.value: MemKey.COMMENT_COLUMN_TYPES}
    max_retries: int = 5
    parse_func_id: str = "extract_comment_as_json"
    human_takeover: bool = False

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        comment = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["comment_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message='Your response did not follow the required format'
                                 '\n```json\n{\n\t"comment": "<comments>"\n}\n```. Correct it now.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=comment, tags=self.output_keys[MemKey.COMMENT_COLUMN_TYPES.value])


class ThinkAndCodeDataMap(SequentialFlow):
    def __init__(
            self,
            loop_flow_prompt_template: str,
            table_view_prompt_template: str,
            error_instruct_prompt_template: str,
            think_prompt_template: str,
            code_prompt_template: str,
            summarize_prompt_template: str,
            modality: str,
            input_or_target: str,
            split: str,
            use_table_view: bool | None = False,
            use_error_instructions: bool | None = False,
            flow_name: str | None = None,
            fill_template: bool = True,
            max_repetitions: int = -1,
            max_retries: int = 5,
            parse_func_id: str = "break_word_split",
            human_takeover_step: int | None = None,
            specialized_llm_name: str | None = None,
    ):
        """
        Flow = SequentialFlow           Loop and act
                - LoopFlow:             loop to create until code can run
                   SequentialFlow:
                     - TableView        [optional] choice to add view of some tables to prompt
                     - ErrorInstruct    [optional] convert past errors into new instructions
                     - Think:           write plan to code & correct errors
                     - Code:            generate and run code
                - Summarize             write a summary of generated code
                - PlanDPAction
                - ExecutePlannedAction

        Args:
            loop_flow_prompt_template:
            think_prompt_template:
            code_prompt_template:
            summarize_prompt_template:
            modality:
            input_or_target:
            split:
            flow_name:
        """
        specification = f"{modality}_{input_or_target}_{split}"
        map_specification = f"map_{specification}"
        assert map_specification in CodeTemplateKeys.list(), \
            (f"Specification {map_specification} not in CodeTemplateKeys "
             f"when initializing ThinkAndCodeDataMap")
        if flow_name is None:
            flow_name = f"create_{map_specification}"

        table_view_cmd = GetTableView(
            required_prompt_templates={"table_view_template": table_view_prompt_template},
            max_retries=max_retries
        )
        error_instruct_cmd = ErrorInstruct(
            required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
            max_retries=max_retries, code_prompt_template=code_prompt_template
        )
        think_cmd = ThinkOfCode(
            required_prompt_templates={"ask_template": think_prompt_template}
        )
        create_data_map_cmd = CreateDataMapOrTransform(
            required_prompt_templates={"code_prompt_template": code_prompt_template},
            specification=map_specification,
            fill_template=fill_template,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0,
            specialized_llm_name=specialized_llm_name,
        )
        code_passed_loop_choice = CodePassedLoopChoiceCmd()

        inner_cmd_seq = [think_cmd, create_data_map_cmd, code_passed_loop_choice]
        if use_error_instructions:
            inner_cmd_seq = [error_instruct_cmd] + inner_cmd_seq
        if use_table_view:
            inner_cmd_seq = [table_view_cmd] + inner_cmd_seq
        inner_sequential_flow = SequentialFlow(
            sequence=inner_cmd_seq,
            name="think_and_code_data_map",
            description="Plan and code to create data map",
        )
        loop_flow = LoopFlow(
            loop_body=inner_sequential_flow,
            max_repetitions=max_repetitions,
            allow_early_break=True,
            max_retries=max_retries,
            prompt_template=loop_flow_prompt_template,
            name=f"{flow_name}_loop",
            description="Loop until data map is created and code passes",
            parse_func_id=parse_func_id,
            memory_choice_tag_val=code_passed_loop_choice.choice_key,
            human_takeover_step=human_takeover_step,
        )

        summarize_cmd = SummarizeCode(
            required_prompt_templates={"summarize_prompt_template": summarize_prompt_template},
            human_takeover=human_takeover_step is not None and human_takeover_step > 0
        )

        plan_action_cmd = ThinkAndCodePlanDPAction(
            stage_name=DataPrepCreateStageName(
                modality=modality,
                split=split,
                input=(input_or_target == "input"),
                target=(input_or_target == "target"),
                map=True,
                transform=False,
            ),
        )

        super().__init__(
            sequence=[loop_flow, summarize_cmd, plan_action_cmd, ExecutePlannedAction()],
            name=flow_name,
            description=f"{loop_flow.description}, then Summarize step and finally Act."
        )


class ThinkAndCodeDataTransform(SequentialFlow):
    def __init__(
            self,
            loop_flow_prompt_template: str,
            table_view_prompt_template: str,
            error_instruct_prompt_template: str,
            think_prompt_template: str,
            code_prompt_template: str,
            summarize_prompt_template: str,
            modality: str,
            use_table_view: bool | None = False,
            use_error_instructions: bool | None = False,
            target_columns_transform_prompt_template: str = None,
            flow_name: str = None,
            max_repetitions: int = -1,
            max_retries: int = 5,
            parse_func_id: str = "break_word_split",
            human_takeover_step: int | None = None,
            specialized_llm_name: str | None = None,
    ):
        """
        Flow = SequentialFlow           Loop, Summarize and Act
                - LoopFlow:             loop to create until code can run
                   SequentialFlow:
                     - ErrorInstruct    [optional] convert past errors into new instructions
                     - TableView        [optional] choice to add view of some tables to prompt
                     - Think:           write plan to code & correct errors
                     - Code:            generate and run code
                - Summarize             write a summary of generated code
                - PlanDPAction
                - ExecutePlannedAction

        Args:
            loop_flow_prompt_template:
            think_prompt_template:
            code_prompt_template:
            summarize_prompt_template:
            modality:
            flow_name:
        """
        specification = f"{modality}_target_train"
        tf_specification = f"transform_{specification}"
        assert tf_specification in CodeTemplateKeys.list(), \
            (f"Specification {tf_specification} not in CodeTemplateKeys "
             f"when initializing ThinkAndCodeDataTransform")
        if flow_name is None:
            flow_name = f"create_{tf_specification}"

        if modality == "tab":
            column_target_cmd = CreateColumnTargetTransform(
                tf_specification=tf_specification,
                required_prompt_templates={
                    "target_columns_transform_template": target_columns_transform_prompt_template
                },
                max_retries=max_retries,
                human_takeover=human_takeover_step is not None and human_takeover_step > 0,
            )

            plan_action_cmd = NoParamsPlanDPAction(
                stage_name=DataPrepCreateStageName(
                    modality=modality,
                    split="train",
                    input=False,
                    target=True,
                    map=False,
                    transform=True,
                ),
            )

            sequence_flow = [column_target_cmd, plan_action_cmd, ExecutePlannedAction()]

        else:
            table_view_cmd = GetTableView(
                required_prompt_templates={"table_view_template": table_view_prompt_template},
                max_retries=max_retries
            )
            error_instruct_cmd = ErrorInstruct(
                required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
                max_retries=max_retries, code_prompt_template=code_prompt_template
            )
            think_cmd = ThinkOfCode(
                required_prompt_templates={"ask_template": think_prompt_template}
            )
            create_data_map_cmd = CreateDataMapOrTransform(
                required_prompt_templates={"code_prompt_template": code_prompt_template},
                specification=tf_specification,
                fill_template=False,
                human_takeover=human_takeover_step is not None and human_takeover_step > 0,
                specialized_llm_name=specialized_llm_name,
            )

            code_passed_loop_choice = CodePassedLoopChoiceCmd()

            inner_cmd_seq = [think_cmd, create_data_map_cmd, code_passed_loop_choice]
            if use_error_instructions:
                inner_cmd_seq = [error_instruct_cmd] + inner_cmd_seq
            if use_table_view:
                inner_cmd_seq = [table_view_cmd] + inner_cmd_seq
            inner_sequential_flow = SequentialFlow(
                sequence=inner_cmd_seq,
                name="think_and_code_data_transform",
                description="Plan and code to create data transform",
            )
            loop_flow = LoopFlow(
                loop_body=inner_sequential_flow,
                max_repetitions=max_repetitions,
                allow_early_break=True,
                max_retries=max_retries,
                prompt_template=loop_flow_prompt_template,
                name=f"{flow_name}_loop",
                description="Loop until data map is created and code passes",
                parse_func_id=parse_func_id,
                memory_choice_tag_val=code_passed_loop_choice.choice_key,
                human_takeover_step=human_takeover_step,
            )

            summarize_cmd = SummarizeCode(
                required_prompt_templates={"summarize_prompt_template": summarize_prompt_template},
                human_takeover=human_takeover_step is not None and human_takeover_step > 0
            )

            plan_action_cmd = ThinkAndCodePlanDPAction(
                stage_name=DataPrepCreateStageName(
                    modality=modality,
                    split="train",
                    input=False,
                    target=True,
                    map=False,
                    transform=True,
                ),
            )
            sequence_flow = [loop_flow, summarize_cmd, plan_action_cmd, ExecutePlannedAction()]

        super().__init__(
            sequence=sequence_flow,
            name=flow_name,
            description="create target transforms, then Summarize step and finally Act."
        )


class CreatePositiveClass(SequentialFlow):
    def __init__(
            self,
            positive_class_prompt_template: str,
            human_takeover_step: int | None = None,
            max_retries: int = 10,
    ):
        get_positive_class_cmd = GetPositiveClass(
            required_prompt_templates={"positive_class_template": positive_class_prompt_template},
            human_takeover=human_takeover_step is not None and human_takeover_step > 0,
            max_retries=max_retries,
        )
        plan_action_cmd = NoParamsPlanDPAction(stage_name=DataPrepPositiveClassStageName())
        super().__init__(
            sequence=[get_positive_class_cmd, plan_action_cmd, ExecutePlannedAction()],
            name="create_positive_class",
            description="creates the positive class JSON if necessary for that task."
        )


class ThinkOfCode(Think):
    input_keys: dict[str, MemKey] = {MemKey.DATA_PREP_PLAN.value: MemKey.DATA_PREP_PLAN}

    def func(self, agent, *args: Any, **kwargs: Any):
        # retrieve code snippets from related stages of the Plan
        plan = agent.memory.retrieve(self.input_keys[MemKey.DATA_PREP_PLAN.value])
        stages_code_dict = plan.get_stages_code()

        if agent.memory.retrieve(MemKey.SUMMARIZE_COT):
            parse_func = PARSE_FUNC_MAP["extract_plan_as_json"]
            format_error_message = ('Use the specified JSON structure:\n'
                                    '```json\n{\n\t"plan": "<plan>"\n}\n```')
        else:
            parse_func = lambda x: x
            format_error_message = ""
        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["ask_template"],
            parse_func=parse_func,
            format_error_message=format_error_message,
            prompt_kwargs={
                "memory": agent.memory,
                "stages_code_dict": stages_code_dict,
            },
            max_retries=5,
            human_takeover=True
        )
        agent.memory.store(response, {self.output_keys["output_mem_key"]})


class GetTableView(HumanTakeoverCommand):
    """
    Ask the Agent if it's interested in seeing information about any tables from the raw data before planning.
    """
    name: str = 'get_table_view'
    description: str = 'Ask LLM which tables it wishes to view information about'
    output_keys: dict[str, MemKey] = {MemKey.ACTIVE_TABLE_VIEW.value: MemKey.ACTIVE_TABLE_VIEW}
    max_retries: int = 5
    human_takeover: bool = False
    parse_func_id: str = "extract_paths_as_json"

    @staticmethod
    def get_table_view(table_path: str, n_rows: int = 2) -> str:
        raw_table_view = ""

        _table_view = FetchTool.df_formated_head_view(
            input_df_path=table_path, n_rows=n_rows, max_columns=100, width=100,
        )
        raw_table_view += '```\n' + _table_view + '\n```\n'

        return raw_table_view

    @staticmethod
    def get_raw_table_info(table_path: str) -> str:
        raw_table_info = ""
        try:
            if ".csv" in table_path:
                table = pd.read_csv(table_path)
            elif ".tsv" in table_path:
                table = pd.read_csv(table_path, sep='\t')
            elif ".json" in table_path:
                try:
                    table = pd.read_json(table_path)
                except ValueError:
                    table = json.load(open(table_path, 'r'))
                    table = json.dumps(table, indent=2)
            else:
                table = None

            if table is not None and isinstance(table, pd.DataFrame):
                buffer = io.StringIO()
                table.info(buf=buffer)
                s = buffer.getvalue()
                _table_info = s
                for c in table.columns[1:]:
                    if (table[c].dtype == 'object' and isinstance(table[c].iloc[0], str)
                            and len(table[c].unique().tolist()) < 200):
                        _table_info += f"\n- column {c} contains strings with values in {table[c].unique().tolist()}"
                raw_table_info += "\n\n" + _table_info
                raw_table_info = '```\n' + raw_table_info + '\n```\n'

            elif table is not None and isinstance(table, dict):
                raw_table_info = str(table)
                if len(raw_table_info.split('\n')) > 100:
                    raw_table_info_lines = raw_table_info.split('\n')[:100]
                    raw_table_info_lines += ["...", "}"]
                    raw_table_info = '\n'.join(raw_table_info_lines)
                raw_table_info = '```\n' + raw_table_info + '\n```\n'

            return raw_table_info

        except ValueError:
            print(f"Skipping view of {table_path} as it it probably not a table.", flush=True)

    def func(self, agent, *args: Any, **kwargs: Any):
        paths = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates['table_view_template'],
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message='Correct those paths now.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover()
        )
        # by this stage, we assume the response is a dict {key=table_name: value=path_to_table}
        if len(paths) > 0:
            active_table_view = ''
            for path in paths:
                if os.path.splitext(path)[1] == '.txt':
                    with open(path, 'r') as f:
                        text_lines = f.readlines()
                    n_lines_to_show = 10
                    active_table_view += "\n".join(text_lines[:n_lines_to_show])
                    active_table_view += "\n..." if len(text_lines) > n_lines_to_show else "\n"
                else:
                    _table_view = self.get_table_view(table_path=path)
                    _table_info = self.get_raw_table_info(table_path=path)
                    if _table_view != '' and _table_info != '':
                        active_table_view += f'#### View of table `{path}`:\n'
                        active_table_view += _table_view
                        active_table_view += _table_info
                        active_table_view += '\n\n'
            agent.memory.store(active_table_view, self.output_keys[MemKey.ACTIVE_TABLE_VIEW.value])


class ErrorInstruct(HumanTakeoverCommand):
    """
    Ask the Agent to interpret past errors and write a new instruction out of it in order to avoid doing the same
    mistake in the future. These are continuously extended until the unit test passes.
    """
    name: str = 'error_instruct'
    description: str = 'Transform past errors in new instructions'
    input_keys: dict[str, MemKey] = {
        MemKey.CODE_RAN.value: MemKey.CODE_RAN,
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED,
        MemKey.UNIT_TEST_RAN.value: MemKey.UNIT_TEST_RAN,
        MemKey.UNIT_TEST_PASSED.value: MemKey.UNIT_TEST_PASSED,
        MemKey.GROUP_UNIT_TEST_RAN.value: MemKey.GROUP_UNIT_TEST_RAN,
        MemKey.GROUP_UNIT_TEST_PASSED.value: MemKey.GROUP_UNIT_TEST_PASSED,
        MemKey.RAG_KEY.value: MemKey.RAG_KEY,
        MemKey.TASK_ID.value: MemKey.TASK_ID
    }
    output_keys: dict[str, MemKey] = {
        MemKey.ERROR_INSTRUCT.value: MemKey.ERROR_INSTRUCT,
        MemKey.CODE_RAG_EXAMPLES.value: MemKey.CODE_RAG_EXAMPLES,
        MemKey.UNIT_TEST_RAG_EXAMPLES.value: MemKey.UNIT_TEST_RAG_EXAMPLES
    }
    code_prompt_template: str
    max_retries: int = 5
    human_takeover: bool = False
    parse_func_id: str = "extract_instruction_as_json"

    def func(self, agent, *args: Any, **kwargs: Any):
        error_instructions = agent.memory.retrieve(MemKey.ERROR_INSTRUCT)  # cannot be in `input_keys` as it raises err
        code_ran = agent.memory.retrieve(self.input_keys[MemKey.CODE_RAN.value])
        code_passed = agent.memory.retrieve(self.input_keys[MemKey.CODE_PASSED.value])
        unit_test_ran = agent.memory.retrieve(self.input_keys[MemKey.UNIT_TEST_RAN.value])
        unit_test_passed = agent.memory.retrieve(self.input_keys[MemKey.UNIT_TEST_PASSED.value])
        group_unit_test_ran = agent.memory.retrieve(self.input_keys[MemKey.GROUP_UNIT_TEST_RAN.value])
        group_unit_test_passed = agent.memory.retrieve(self.input_keys[MemKey.GROUP_UNIT_TEST_PASSED.value])

        _code_error, _unit_test_error, _group_unit_test_error = False, False, False
        if code_ran and not code_passed:
            _code_error = True
        if not _code_error and unit_test_ran and not unit_test_passed:
            _unit_test_error = True
        if not _code_error and not _unit_test_error and group_unit_test_ran and not group_unit_test_passed:
            _group_unit_test_error = True

        if _code_error or _unit_test_error or _group_unit_test_error:
            response = safe_parsing_chat_completion(
                agent=agent,
                ask_template=self.required_prompt_templates['error_instruct_template'],
                parse_func=PARSE_FUNC_MAP[self.parse_func_id],
                format_error_message='Your response did not follow the required format'
                                     '\n```json\n{\n\t"instruction": "<instruction>"\n}\n```. Correct it now.',
                max_retries=self.max_retries,
                human_takeover=self.check_trigger_human_takeover()
            )

            # Retrieve relevant examples
            if DB_FAISS.started:
                rag_key = agent.memory.retrieve(self.input_keys[MemKey.RAG_KEY.value])
                if rag_key:
                    task_id = agent.memory.retrieve(self.input_keys[MemKey.TASK_ID.value])
                    retrieved_example = DB_FAISS.retrieve_top_template_error_example(
                        query=rag_key, template_file=self.code_prompt_template, heldout_metadata={"task_id": task_id}
                    )
                    if retrieved_example is not None:
                        if _code_error:
                            agent.memory.store(retrieved_example, self.output_keys[MemKey.CODE_RAG_EXAMPLES.value])
                        else:
                            agent.memory.store(retrieved_example, self.output_keys[MemKey.UNIT_TEST_RAG_EXAMPLES.value])

            if len(response) > 0:
                if error_instructions is not None:
                    error_instructions += f'\n- {response}'
                else:
                    error_instructions = f"\n- {response}"
                agent.memory.store(error_instructions, self.output_keys[MemKey.ERROR_INSTRUCT.value])


class GetPositiveClass(HumanTakeoverCommand):
    name: str = "get_positive_class"
    description: str = (
        "Write positive class(es) if necessary."
    )
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.TASK_CATEGORY.value: MemKey.TASK_CATEGORY,
        MemKey.DATA_PREP_PLAN.value: MemKey.DATA_PREP_PLAN,
        MemKey.WORKSPACE_PATH.value: MemKey.WORKSPACE_PATH,
    }
    output_keys: dict[str, MemKey] = {
        MemKey.POSITIVE_CLASS.value: MemKey.POSITIVE_CLASS,
        MemKey.TARGET_CLASSES.value: MemKey.TARGET_CLASSES
    }
    human_takeover: bool = False
    max_retries: int = 1

    def func(self, agent: LLMAgent, *args, **kwargs) -> None:
        task_category = agent.memory.retrieve(tags=self.input_keys[MemKey.TASK_CATEGORY.value])
        workspace_path = agent.memory.retrieve(tags=self.input_keys[MemKey.WORKSPACE_PATH.value])
        plan = agent.memory.retrieve(tags=self.input_keys[MemKey.DATA_PREP_PLAN.value])
        if (
                "classification" in task_category and
                "binary" in task_category and
                plan.is_full_tabular
        ):
            train_tab_target_map_path = os.path.join(workspace_path, 'train_tab_target_map.csv')
            df = pd.read_csv(train_tab_target_map_path)
            targets_dic = {}
            for col in df.columns:
                if col != 'id':
                    targets_dic[col] = list(df[col].unique())
            targets_dic_to_str = ""
            for k in targets_dic:
                targets_dic_to_str += f"- target column name: {k}, target_column_values: {targets_dic[k]}\n"
            agent.memory.store(content=targets_dic_to_str, tags=self.output_keys[MemKey.TARGET_CLASSES.value])
            parsing_func = partial(extract_column_names_and_values_as_json, path_to_df=train_tab_target_map_path)
            positive_class = safe_parsing_chat_completion(
                agent=agent,
                ask_template=self.required_prompt_templates["positive_class_template"],
                parse_func=parsing_func,
                format_error_message='Make sure your response follows the required format.',
                max_retries=self.max_retries,
                human_takeover=self.check_trigger_human_takeover()
            )
            agent.memory.store(content=positive_class, tags=self.output_keys[MemKey.POSITIVE_CLASS.value])
            with open(os.path.join(workspace_path, "metadata", "positive_class.json"), "w") as fp:
                json.dump(obj=positive_class, fp=fp, indent=True)


class GetTaskCategory(HumanTakeoverCommand):
    name: str = "get_task_category"
    description: str = (
        "Write the task category."
    )
    required_prompt_templates: dict[str, str]
    output_keys: dict[str, MemKey] = {
        MemKey.TASK_CATEGORY.value: MemKey.TASK_CATEGORY
    }
    workspace_path: str
    human_takeover: bool = False
    max_retries: int = 1
    parse_func_id: str = 'extract_task_type_as_json'

    def func(self, agent: LLMAgent, *args, **kwargs) -> None:
        task_category = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["task_category_template"],
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message='Your response did not follow the required format'
                                 '\n```json\n{\n\t"task_type": "<task_type>"\n}\n```. Correct it now.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover()
        )
        agent.memory.store(content=task_category, tags=self.output_keys[MemKey.TASK_CATEGORY.value])
        task_category_json = {"task_type": task_category}
        json.dump(
            obj=task_category_json,
            fp=open(os.path.join(self.workspace_path, "metadata/task_category.json"), "w"),
            indent=True
        )


class WriteTabularTargetsStructure(HumanTakeoverCommand):
    name: str = "create_tab_targets_struct"
    description: str = "Create tabular targets struct for main pipeline"
    output_keys: dict[str, MemKey] = {MemKey.TAB_TARGETS_STRUCT.value: MemKey.TAB_TARGETS_STRUCT}
    workspace_path: str
    max_retries: int = 5
    human_takeover: bool = False

    def func(self, agent: LLMAgent, *args, **kwargs) -> None:
        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["tab_targets_struct_prompt_template"],
            parse_func=extract_python,
            format_error_message=('Use the specified PYTHON structure:\n'
                                  '```python\nTARGET_STRUCT = [\n\t(<dimension>, <list>),\n]\n```'),
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover()
        )
        agent.memory.store(content=response, tags=self.output_keys[MemKey.TAB_TARGETS_STRUCT.value])
        with open(os.path.join(self.workspace_path, "tab_targets_struct.py"), "w") as f:
            f.write(response)


class FetchAndSummarizeData(SequentialFlow):
    def __init__(
            self,
            fetch_tool_prompt_template: str,
            detect_sample_submission_prompt_template: str,
            summarize_required_prompt_templates: dict[str, str],
            workspace_path: str,
            raw_data_dir_name: str,
            task_url: str,
            is_local_task: bool,
            user_details: str = "./third_party/data_preprocessing/kaggle_login_details.json",
            path_to_saved_responses: str | None = None,
            sample_submission_file: str | None = None,
            max_retries: int = 5,
            human_takeover_step: int = None,
            skip_summarization: bool = False,
    ):
        """
        Fetch data from Kaggle

        Args:
            raw_data_dir_name: name of the folder where the raw data will be saved
            sample_submission_file: name of the file containing the sample submission
        """
        raw_data_dir = str(get_raw_data_root_dir() / raw_data_dir_name)

        # ---------- Fetch ----------
        use_fetch_tool = UseTool(
            name="fetch_and_scrap",
            description="Fetch data from Kaggle and scrap website for raw descriptions",
            prompt_template=fetch_tool_prompt_template,
            tool=FetchTool(
                task_url=task_url,
                workspace_path=workspace_path,
                is_local_task=is_local_task,
                raw_data_dir=raw_data_dir,
                user_details=user_details,
                sample_submission_file=sample_submission_file
            )
        )
        # ---------- Detect Sample Submission with LLM if necessary ----------
        detect_sample_submission = DetectSampleSubmissionFile(
            required_prompt_templates={
                'detect_sample_submission_prompt_template': detect_sample_submission_prompt_template
            },
            workspace_path=workspace_path,
            max_retries=max_retries,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0,
            parse_func_id='extract_detected_file_as_json',
            sample_submission_file=sample_submission_file,
        )
        # ---------- Summarize ----------
        summarize_command = SummarizeRawDescription(
            workspace_path=workspace_path,
            raw_data_dir=raw_data_dir,
            required_prompt_templates=summarize_required_prompt_templates,
            path_to_saved_responses=path_to_saved_responses
        )
        # ----- Prevent writing in raw ----
        remove_write_permission_command = RemoveWritePermission(
            raw_data_dir=raw_data_dir,
        )

        # ---------- combine all ----------
        if skip_summarization:
            sequence = [
                use_fetch_tool,
                detect_sample_submission,
                remove_write_permission_command
            ]
        else:
            sequence = [
                use_fetch_tool,
                detect_sample_submission,
                summarize_command,
                remove_write_permission_command
            ]
        super().__init__(
            name="fetch_and_summarize_data",
            description="fetch data, scrap metadata and summarize",
            sequence=sequence,
        )


class DetectSampleSubmissionFile(HumanTakeoverCommand):
    name: str = "detect_sample_submission_file"
    description: str = ("Detect with the LLM the file that is closest to the sample submission "
                        "if the file was not able to be detected otherwise")
    input_keys: dict[str, MemKey] = {
        MemKey.DETECT_SAMPLE_SUBMISSION_WITH_LLM.value: MemKey.DETECT_SAMPLE_SUBMISSION_WITH_LLM,
        MemKey.RAW_DATA_DIR.value: MemKey.RAW_DATA_DIR,
        MemKey.TASK_ID.value: MemKey.TASK_ID,
    }
    output_keys: dict[str, MemKey] = {
        MemKey.HAS_SAMPLE_SUBMISSION.value: MemKey.HAS_SAMPLE_SUBMISSION,
        MemKey.SAMPLE_SUBMISSION_HEAD.value: MemKey.SAMPLE_SUBMISSION_HEAD,
        MemKey.RAW_TARGETS_COLUMN_NAMES.value: MemKey.RAW_TARGETS_COLUMN_NAMES,
        MemKey.RAW_ID_COLUMN_NAME.value: MemKey.RAW_ID_COLUMN_NAME,
        MemKey.RAW_TABLE_VIEW.value: MemKey.RAW_TABLE_VIEW,
        MemKey.RAW_TABLE_INFO.value: MemKey.RAW_TABLE_INFO,
        MemKey.RAW_DATA_VIEW.value: MemKey.RAW_DATA_VIEW,
    }
    workspace_path: str
    max_retries: int = 5
    human_takeover: bool = False
    parse_func_id: str = 'extract_detected_file_as_json'
    sample_submission_file: str | None = None

    def func(self, agent: LLMAgent, *args, **kwargs) -> None:
        detect_sample_submission_with_llm = agent.memory.retrieve(
            self.input_keys[MemKey.DETECT_SAMPLE_SUBMISSION_WITH_LLM.value]
        )
        raw_data_dir = agent.memory.retrieve(self.input_keys[MemKey.RAW_DATA_DIR.value])
        if not detect_sample_submission_with_llm:
            return

        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["detect_sample_submission_prompt_template"],
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message=('Use the specified JSON structure:\n'
                                  '```json\n{\n\t"detected_file": "<detected_file>",\n\t}\n```'),
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover()
        )
        _, has_sample_submission, id_name, target_names, sample_submission_head = FetchTool.review_sample_submission(
            raw_data_dir=raw_data_dir,
            workspace_path=self.workspace_path,
            task_id=agent.memory.retrieve(self.input_keys[MemKey.TASK_ID.value]),
            sample_submission_file=self.sample_submission_file,
            detected_file=response,
        )
        raw_table_view = FetchTool.get_raw_table_view(
            download_dir=raw_data_dir,
            raw_table_view_path=os.path.join(self.input_keys[MemKey.RAW_DATA_DIR.value], "raw_table_view.txt"),
            detected_file=response,
            fetched_path=os.path.join(raw_data_dir, 'fetched_raw_data_done.json'),
        )
        raw_table_info = FetchTool.get_raw_table_info(
            download_dir=raw_data_dir,
            raw_table_info_path=os.path.join(raw_data_dir, "raw_table_view.txt"),
            detected_file=response,
            fetched_path=os.path.join(raw_data_dir, 'fetched_raw_data_done.json'),
        )
        raw_data_view = FetchTool.get_raw_data_view(
            download_dir=raw_data_dir,
            raw_data_view_path=os.path.join(raw_data_dir, "raw_table_view.txt"),
            detected_file=response,
        )
        agent.memory.store(content=sample_submission_head, tags=self.output_keys[MemKey.SAMPLE_SUBMISSION_HEAD.value])
        agent.memory.store(content=has_sample_submission, tags=self.output_keys[MemKey.HAS_SAMPLE_SUBMISSION.value])
        agent.memory.store(content=target_names, tags=self.output_keys[MemKey.RAW_TARGETS_COLUMN_NAMES.value])
        agent.memory.store(content=id_name, tags=self.output_keys[MemKey.RAW_ID_COLUMN_NAME.value])
        agent.memory.store(content=raw_table_view, tags=self.output_keys[MemKey.RAW_TABLE_VIEW.value])
        agent.memory.store(content=raw_table_info, tags=self.output_keys[MemKey.RAW_TABLE_INFO.value])
        agent.memory.store(content=raw_data_view, tags=self.output_keys[MemKey.RAW_DATA_VIEW.value])


class ThinkAndCodeMetric(SequentialFlow):
    def __init__(
            self,
            loop_flow_prompt_template: str,
            table_view_prompt_template: str,
            error_instruct_prompt_template: str,
            think_prompt_template: str,
            code_prompt_template: str,
            summarize_prompt_template: str,
            use_table_view: bool | None = False,
            use_error_instructions: bool | None = False,
            max_repetitions: int = -1,
            max_retries: int = 5,
            parse_func_id: str = "break_word_split",
            human_takeover_step: int | None = None,
            specialized_llm_name: str | None = None,
    ):
        table_view_cmd = GetTableView(
            required_prompt_templates={"table_view_template": table_view_prompt_template},
            max_retries=max_retries
        )
        error_instruct_cmd = ErrorInstruct(
            required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
            max_retries=max_retries, code_prompt_template=code_prompt_template
        )
        think_cmd = ThinkOfCode(
            required_prompt_templates={"ask_template": think_prompt_template}
        )
        create_metric_cmd = CreateMetric(
            required_prompt_templates={"code_prompt_template": code_prompt_template},
            fill_template=False,
            max_retries=max_retries,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0,
            specialized_llm_name=specialized_llm_name,
        )
        code_passed_loop_choice = CodePassedLoopChoiceCmd()

        inner_cmd_seq = [think_cmd, create_metric_cmd, code_passed_loop_choice]
        if use_error_instructions:
            inner_cmd_seq = [error_instruct_cmd] + inner_cmd_seq
        if use_table_view:
            inner_cmd_seq = [table_view_cmd] + inner_cmd_seq
        inner_sequential_flow = SequentialFlow(
            sequence=inner_cmd_seq,
            name="think_and_code_metric",
            description="Plan and code to create metric function",
        )
        loop_flow = LoopFlow(
            loop_body=inner_sequential_flow,
            max_repetitions=max_repetitions,
            allow_early_break=True,
            max_retries=max_retries,
            prompt_template=loop_flow_prompt_template,
            name="metric_loop",
            description="Loop until data map is created and code passes",
            parse_func_id=parse_func_id,
            memory_choice_tag_val=code_passed_loop_choice.choice_key,
            human_takeover_step=human_takeover_step,
        )

        summarize_cmd = SummarizeCode(
            required_prompt_templates={"summarize_prompt_template": summarize_prompt_template},
            human_takeover=human_takeover_step is not None and human_takeover_step > 0

        )

        plan_action_cmd = ThinkAndCodePlanDPAction(stage_name=DataPrepMetricStageName())

        super().__init__(
            sequence=[loop_flow, summarize_cmd, plan_action_cmd, ExecutePlannedAction()],
            name="create_and_summarize_metric_function",
            description=f"{loop_flow.description}, then Summarize step and finally Act."
        )


class ThinkAndCodeColumnTypes(SequentialFlow):
    def __init__(
            self,
            loop_flow_prompt_template: str,
            error_instruct_prompt_template: str,
            think_prompt_template: str,
            code_prompt_template: str,
            comment_prompt_template: str,
            summarize_prompt_template: str,
            use_error_instructions: bool | None = False,
            max_repetitions: int = -1,
            max_retries: int = 5,
            parse_func_id: str = "break_word_split",
            human_takeover_step: int | None = None,
            fill_template: bool = False,
    ):
        store_tab_train_cmd = StoreTabTrainInMemory()
        error_instruct_cmd = ErrorInstruct(
            required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
            max_retries=max_retries, code_prompt_template=code_prompt_template
        )
        think_cmd = ThinkOfCode(
            required_prompt_templates={"ask_template": think_prompt_template}
        )
        create_column_types_cmd = CreateColumnTypes(
            required_prompt_templates={"code_prompt_template": code_prompt_template},
            fill_template=fill_template,
            max_retries=max_retries,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0
        )
        code_passed_loop_choice = ColTypesPassedLoopChoiceCmd(max_steps=1)
        store_col_types_cmd = StoreColTypesInMemory()
        comment_column_types_cmd = CommentColTypesCmd(
            required_prompt_templates={"comment_prompt_template": comment_prompt_template}
        )
        inner_cmd_seq = [
            think_cmd,
            create_column_types_cmd,
            code_passed_loop_choice,
            store_col_types_cmd,
            comment_column_types_cmd
        ]
        if use_error_instructions:
            inner_cmd_seq = [error_instruct_cmd] + inner_cmd_seq
        inner_sequential_flow = SequentialFlow(
            sequence=inner_cmd_seq,
            name="think_and_code_column_types",
            description="Plan and code to create column types function",
        )
        loop_flow = LoopFlow(
            loop_body=inner_sequential_flow,
            max_repetitions=max_repetitions,
            allow_early_break=True,
            max_retries=max_retries,
            prompt_template=loop_flow_prompt_template,
            name="column_types_loop",
            description="Loop until column types json is created and code passes",
            parse_func_id=parse_func_id,
            memory_choice_tag_val=code_passed_loop_choice.choice_key,
            human_takeover_step=human_takeover_step,
        )

        summarize_cmd = SummarizeCode(
            required_prompt_templates={"summarize_prompt_template": summarize_prompt_template},
            human_takeover=human_takeover_step is not None and human_takeover_step > 0

        )
        stage_name = DataPrepColumnTypesStageName()
        plan_action_cmd = ThinkAndCodePlanDPAction(stage_name=stage_name)

        super().__init__(
            sequence=[store_tab_train_cmd, loop_flow, summarize_cmd, plan_action_cmd, ExecutePlannedAction()],
            name=stage_name.to_str(),
            description=f"{loop_flow.description}, then Summarize step and finally Act."
        )


class ThinkAndCodeSubmissionFormat(SequentialFlow):
    def __init__(
            self,
            loop_flow_prompt_template: str,
            table_view_prompt_template: str,
            error_instruct_prompt_template: str,
            think_prompt_template: str,
            code_prompt_template: str,
            summarize_prompt_template: str,
            use_table_view: bool | None = False,
            use_error_instructions: bool | None = False,
            flow_name: str = None,
            max_repetitions: int = -1,
            max_retries: int = 5,
            parse_func_id: str = "break_word_split",
            human_takeover_step: int | None = None,
            code_filename_specification: str = "submission_format",
    ):
        if flow_name is None:
            flow_name = "create_submission_format"

        store_inv_transform_metadata_cmd = StoreInverseTransformInMemory()
        table_view_cmd = GetTableView(
            required_prompt_templates={"table_view_template": table_view_prompt_template},
            max_retries=max_retries
        )
        error_instruct_cmd = ErrorInstruct(
            required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
            max_retries=max_retries,
            code_prompt_template=code_prompt_template
        )
        think_cmd = ThinkOfCode(
            required_prompt_templates={"ask_template": think_prompt_template}
        )
        create_submission_format_cmd = CreateSubmissionFormat(
            required_prompt_templates={"code_prompt_template": code_prompt_template},
            specification=code_filename_specification,
            fill_template=False,
            max_retries=max_retries,
            human_takeover=human_takeover_step is not None and human_takeover_step > 0
        )
        code_passed_loop_choice = CodePassedLoopChoiceCmd()
        inner_cmd_seq = [
            store_inv_transform_metadata_cmd, think_cmd, create_submission_format_cmd,
            code_passed_loop_choice
        ]
        if use_error_instructions:
            inner_cmd_seq = [error_instruct_cmd] + inner_cmd_seq
        if use_table_view:
            inner_cmd_seq = [table_view_cmd] + inner_cmd_seq
        inner_sequential_flow = SequentialFlow(
            sequence=inner_cmd_seq,
            name="think_and_code_submission_format",
            description="Plan and code to create submission_format function",
        )
        loop_flow = LoopFlow(
            loop_body=inner_sequential_flow,
            max_repetitions=max_repetitions,
            allow_early_break=True,
            max_retries=max_retries,
            prompt_template=loop_flow_prompt_template,
            name=f"{flow_name}_loop",
            description="Loop until data map is created and code passes",
            parse_func_id=parse_func_id,
            human_takeover_step=human_takeover_step,
            memory_choice_tag_val=code_passed_loop_choice.choice_key
        )

        summarize_cmd = SummarizeCode(
            required_prompt_templates={"summarize_prompt_template": summarize_prompt_template},
            human_takeover=human_takeover_step is not None and human_takeover_step > 0
        )
        if code_filename_specification == "submission_format_alt":
            plan_action_cmd = ThinkAndCodePlanDPAction(stage_name=DataPrepSubmissionFormatAltStageName())
        else:
            plan_action_cmd = ThinkAndCodePlanDPAction(stage_name=DataPrepSubmissionFormatStageName())

        super().__init__(
            sequence=[loop_flow, summarize_cmd, plan_action_cmd, ExecutePlannedAction()],
            name=flow_name,
            description=f"{loop_flow.description}, then Summarize step and finally Act."
        )


class SelectMetric(Flow):
    def __init__(
            self,
            prompt_template: str,
            workspace_path: str,
            metric_coding_flow: Flow,
            max_retries: int = 1,
            max_tokens: int | None = None,
            parse_func_id: str = "extract_metric_as_json",
            human_takeover_step: int | None = 0
    ):
        """A flow inspired from the DecisionFlow and used to select the metric among the ones available
        or ask the LLM to code it"""

        name = "select_metric"
        description = "A flow to select the metric to use in the challenge"
        self.workspace_path = workspace_path
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.parse_function = PARSE_FUNC_MAP[parse_func_id]
        self.metric_coding_flow = metric_coding_flow
        self._choice_taken = None
        self.human_takeover_step = human_takeover_step
        super().__init__(name=name, description=description, prompt_template=prompt_template)

    def reset(self) -> None:
        """Reset the DecisionFlow and all it sub-flows to their initial state."""
        self._choice_taken = None
        self.metric_coding_flow.reset()

    def step(self, agent: LLMAgent) -> Command | Flow | None:
        """Selects among the available metrics. If None uses the LLM to code the metric"""

        # stopping criterion
        if self._choice_taken is not None:
            if isinstance(self._choice_taken, Flow):
                return self._choice_taken.step(agent)
            return None

        # check if the task involves only tabular data
        has_img_input = os.path.exists(os.path.join(self.workspace_path, FileMap.TRAIN_IMAGE_INPUT.value))
        has_img_target = os.path.exists(os.path.join(self.workspace_path, FileMap.TRAIN_IMAGE_TARGET.value))
        has_txt_input = os.path.exists(os.path.join(self.workspace_path, FileMap.TRAIN_TEXT_INPUT.value))
        has_txt_target = os.path.exists(os.path.join(self.workspace_path, FileMap.TRAIN_TEXT_TARGET.value))

        if not (has_img_input or has_img_target or has_txt_input or has_txt_target):
            from agent.tasks.datascience_task import ramp_utils

            metrics = ramp_utils.RAMP_METRICS + ['none']

            # Select the Metric
            # This is based on the DecisionFlow
            # ---------------------------
            assert len(metrics) > 0
            # Even if self.metrics are not flows, this is ok, as in the DecisionFlow they use a list of string as well

            selected_command = None
            try:
                selected_command = agent.safe_choose_from_options(
                    ask_template=self.prompt_template,
                    parse_func=self.parse_function,
                    format_error_message='Please answer using appropriate format.',
                    options=metrics,
                    prompt_kwargs={"memory": agent.memory, "metrics": metrics},
                    max_retries=self.max_retries,
                    human_takeover=self.human_takeover_step > 0,
                    max_tokens=self.max_tokens,
                )
            except ParsingError as e:
                print(e)

            selected_metric = None
            if selected_command is not None:
                selected_metric = next((metric for metric in metrics if metric == selected_command), None)

            if selected_metric not in ["none", None]:
                agent.memory.store(content={"metric": selected_metric}, tags=MemKey.SELECTED_METRIC)
                metric_path = os.path.join(
                    agent.task.workspace_path, FileMap.METADATA.value, FileMap.RAMP_METRIC_NAME.value
                )
                with open(metric_path, "w") as f:
                    json.dump({"metric": selected_metric}, f)
                self._choice_taken = choice = RAMPMetricActionFlow()
                return choice.step(agent)

            code_templates = agent.memory.retrieve(MemKey.CODE_TEMPLATES)
            code_templates['metric'] = code_templates['ramp_metric']
            agent.memory.store(code_templates, tags=MemKey.CODE_TEMPLATES)

        # Call the flow that codes the metric
        print("Score did not correspond to any of the available ones. Generating score function...")
        self._choice_taken = choice = self.metric_coding_flow
        return choice.step(agent)


class SummarizeRawDescription(HumanTakeoverCommand):
    """
    DSAgent: Summarize raw description into more concise and useful information.
    """
    workspace_path: str
    raw_data_dir: str
    name: str = "summarize_raw_description"
    description: str = "Summarize raw descriptions to extract more useful and concise information"
    path_to_saved_responses: str | None = None

    required_prompt_templates: dict[str, str | None]
    input_keys: dict[str, MemKey] = {
        MemKey.RAW_TASK_DESCRIPTION.value: MemKey.RAW_TASK_DESCRIPTION,
        MemKey.RAW_DATA_DESCRIPTION.value: MemKey.RAW_DATA_DESCRIPTION,
        MemKey.RAW_DATA_VIEW.value: MemKey.RAW_DATA_VIEW,
        MemKey.RAW_METRIC_DESCRIPTION.value: MemKey.RAW_METRIC_DESCRIPTION,
        MemKey.RAW_TABLE_INFO.value: MemKey.RAW_TABLE_INFO
    }
    output_keys: dict[str, str] = {
        memkey.value: memkey for memkey in
        [
            MemKey.SUMMARIZED_TASK_DESCRIPTION,
            MemKey.SUMMARIZED_DATA_DESCRIPTION,
            MemKey.SUMMARIZED_METRIC_DESCRIPTION,
            MemKey.SUMMARIZED_SUBMISSION_FORMAT,
            MemKey.SUMMARIZED_INPUTS_MODALITY_MAPS,
            MemKey.SUMMARIZED_TARGETS_MODALITY_MAPS,
            MemKey.SUMMARIZED_TARGETS_MODALITY_TRANSFORMS
        ]
    }

    def _summarize(self, agent: LLMAgent, ask_template: str, prompt_kwargs: dict[str, Any]) -> str:
        if agent.memory.retrieve(MemKey.SUMMARIZE_COT):
            parse_func = PARSE_FUNC_MAP["extract_summary_as_json"]
            format_error_message = ('Use the specified JSON structure:\n'
                                    '```json\n{\n\t"summary": "<summary>"\n}\n```')
        else:
            parse_func = lambda x: x
            format_error_message = ""
        response = safe_parsing_chat_completion(
            agent=agent,
            ask_template=ask_template,
            parse_func=parse_func,
            format_error_message=format_error_message,
            max_retries=5,
            human_takeover=True,
            prompt_kwargs=prompt_kwargs,
            path_to_saved_responses=self.path_to_saved_responses
        )
        return response

    def func(self, agent, *args, **kwargs) -> None:
        raw_task_description = agent.memory.retrieve(self.input_keys[MemKey.RAW_TASK_DESCRIPTION.value])
        raw_data_description = agent.memory.retrieve(self.input_keys[MemKey.RAW_DATA_DESCRIPTION.value])
        raw_metric_description = agent.memory.retrieve(self.input_keys[MemKey.RAW_METRIC_DESCRIPTION.value])
        raw_table_info = agent.memory.retrieve(self.input_keys[MemKey.RAW_TABLE_INFO.value])
        raw_data_view = agent.memory.retrieve(self.input_keys[MemKey.RAW_DATA_VIEW.value])

        # ------------------------------------------------
        # ---------- summarize task description ----------
        # ------------------------------------------------
        summarized_task = self._summarize(
            agent,
            ask_template=self.required_prompt_templates["task_template"],
            prompt_kwargs={
                "raw_task_description": raw_task_description,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0})
            }
        )
        agent.memory.store(summarized_task, {self.output_keys[MemKey.SUMMARIZED_TASK_DESCRIPTION.value]})
        with open(os.path.join(self.workspace_path, "metadata/task_description.txt"), 'w') as f:
            f.writelines(summarized_task)

        # ------------------------------------------------
        # ---------- summarize data description ----------
        # ------------------------------------------------
        summarized_data = self._summarize(
            agent=agent,
            ask_template=self.required_prompt_templates["data_template"],
            prompt_kwargs={
                "raw_data_description": raw_data_description,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0}),
                "has_sample_submission": agent.memory.retrieve({MemKey.HAS_SAMPLE_SUBMISSION: 1.0}),
                "raw_id_column_name": agent.memory.retrieve({MemKey.RAW_ID_COLUMN_NAME: 1.0}),
                "raw_target_column_names": agent.memory.retrieve({MemKey.RAW_TARGETS_COLUMN_NAMES: 1.0}),
                "sample_submission_head": agent.memory.retrieve({MemKey.SAMPLE_SUBMISSION_HEAD: 1.0}),
                "raw_table_info": raw_table_info,
                "raw_data_view": raw_data_view,
                "raw_data_dir": self.raw_data_dir
            }
        )
        # append raw_data_view to summary
        raw_data_view_str = (
            f"### View of the top-level directory {self.raw_data_dir}:"
            f"\nafter having downloaded and extracted the raw data from the source, "
            f"these are the data you can use to solve this task:"
            f"\n```"
            f"\n{raw_data_view}"
            f"\n```"
        )
        summarized_data = raw_data_view_str + summarized_data
        agent.memory.store(summarized_data, {self.output_keys[MemKey.SUMMARIZED_DATA_DESCRIPTION.value]})
        with open(os.path.join(self.workspace_path, "metadata/data_description.txt"), 'w') as f:
            f.writelines(summarized_data)

        # --------------------------------------------------
        # ---------- summarize metric description ----------
        # --------------------------------------------------
        summarized_metric = self._summarize(
            agent=agent,
            ask_template=self.required_prompt_templates["metric_template"],
            prompt_kwargs={
                "raw_metric_description": raw_metric_description,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0}),
                "has_sample_submission": agent.memory.retrieve({MemKey.HAS_SAMPLE_SUBMISSION: 1.0}),
                "raw_id_column_name": agent.memory.retrieve({MemKey.RAW_ID_COLUMN_NAME: 1.0}),
                "raw_target_column_names": agent.memory.retrieve({MemKey.RAW_TARGETS_COLUMN_NAMES: 1.0}),
                "sample_submission_head": agent.memory.retrieve({MemKey.SAMPLE_SUBMISSION_HEAD: 1.0}),
                "raw_data_dir": agent.memory.retrieve({MemKey.RAW_DATA_DIR: 1.0}),
                "raw_data_view": agent.memory.retrieve({MemKey.RAW_DATA_VIEW: 1.0}),
                "raw_table_info": agent.memory.retrieve({MemKey.RAW_TABLE_INFO: 1.0}),
                "summarized_data_description": agent.memory.retrieve({MemKey.SUMMARIZED_DATA_DESCRIPTION: 1.0}),
            }
        )
        agent.memory.store(summarized_metric, {self.output_keys[MemKey.SUMMARIZED_METRIC_DESCRIPTION.value]})
        with open(os.path.join(self.workspace_path, "metadata/metric_description.txt"), 'w') as f:
            f.writelines(summarized_metric)

        # ----------------------------------------------------------------------------------------------------
        # ---------- summarize submission format (useful in particular for target transforms later) ----------
        # ----------------------------------------------------------------------------------------------------
        summarized_submission_format = self._summarize(
            agent=agent,
            ask_template=self.required_prompt_templates["submission_template"],
            prompt_kwargs={
                "raw_task_description": raw_task_description,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0}),
                "has_sample_submission": agent.memory.retrieve({MemKey.HAS_SAMPLE_SUBMISSION: 1.0}),
                "raw_id_column_name": agent.memory.retrieve({MemKey.RAW_ID_COLUMN_NAME: 1.0}),
                "raw_target_column_names": agent.memory.retrieve({MemKey.RAW_TARGETS_COLUMN_NAMES: 1.0}),
                "sample_submission_head": agent.memory.retrieve({MemKey.SAMPLE_SUBMISSION_HEAD: 1.0}),
                "raw_data_dir": agent.memory.retrieve({MemKey.RAW_DATA_DIR: 1.0}),
                "raw_data_view": agent.memory.retrieve({MemKey.RAW_DATA_VIEW: 1.0}),
                "raw_table_info": agent.memory.retrieve({MemKey.RAW_TABLE_INFO: 1.0}),
                "summarized_data_description": agent.memory.retrieve({MemKey.SUMMARIZED_DATA_DESCRIPTION: 1.0}),
            }
        )
        agent.memory.store(summarized_submission_format, {self.output_keys[MemKey.SUMMARIZED_SUBMISSION_FORMAT.value]})
        with open(os.path.join(self.workspace_path, "metadata/submission_format.txt"), 'w') as f:
            f.writelines(summarized_submission_format)

        # ------------------------------------------------------------
        # ---------- summarize modality maps for the inputs ----------
        # ------------------------------------------------------------
        summarized_inputs_modality_maps = self._summarize(
            agent=agent,
            ask_template=self.required_prompt_templates["input_modality_maps_template"],
            prompt_kwargs={
                "summarized_data_description": summarized_data,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0}),
                "has_sample_submission": agent.memory.retrieve({MemKey.HAS_SAMPLE_SUBMISSION: 1.0}),
                "raw_id_column_name": agent.memory.retrieve({MemKey.RAW_ID_COLUMN_NAME: 1.0}),
                "raw_target_column_names": agent.memory.retrieve({MemKey.RAW_TARGETS_COLUMN_NAMES: 1.0}),
                "sample_submission_head": agent.memory.retrieve({MemKey.SAMPLE_SUBMISSION_HEAD: 1.0}),
                "raw_table_info": raw_table_info,
                "raw_data_dir": self.raw_data_dir,
                "summarized_task_description": summarized_task
            }
        )
        agent.memory.store(
            summarized_inputs_modality_maps,
            {self.output_keys[MemKey.SUMMARIZED_INPUTS_MODALITY_MAPS.value]}
        )
        with open(os.path.join(self.workspace_path, "metadata/input_modality_maps_description.txt"), 'w') as f:
            f.writelines(summarized_inputs_modality_maps)

        # -------------------------------------------------------------
        # ---------- summarize modality maps for the targets ----------
        # -------------------------------------------------------------
        summarized_targets_modality_maps = self._summarize(
            agent=agent,
            ask_template=self.required_prompt_templates["target_modality_maps_template"],
            prompt_kwargs={
                "summarized_data_description": summarized_data,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0}),
                "has_sample_submission": agent.memory.retrieve({MemKey.HAS_SAMPLE_SUBMISSION: 1.0}),
                "raw_id_column_name": agent.memory.retrieve({MemKey.RAW_ID_COLUMN_NAME: 1.0}),
                "raw_target_column_names": agent.memory.retrieve({MemKey.RAW_TARGETS_COLUMN_NAMES: 1.0}),
                "sample_submission_head": agent.memory.retrieve({MemKey.SAMPLE_SUBMISSION_HEAD: 1.0}),
                "raw_table_info": raw_table_info,
                "raw_data_dir": self.raw_data_dir
            }
        )
        agent.memory.store(
            summarized_targets_modality_maps,
            {self.output_keys[MemKey.SUMMARIZED_TARGETS_MODALITY_MAPS.value]}
        )
        with open(os.path.join(self.workspace_path, "metadata/target_modality_maps_description.txt"), 'w') as f:
            f.writelines(summarized_targets_modality_maps)

        # ---------------------------------------------------
        # ---------- summarize modality transforms ----------
        # ---------------------------------------------------
        summarized_modality_transforms = self._summarize(
            agent=agent,
            ask_template=self.required_prompt_templates["modality_transforms_template"],
            prompt_kwargs={
                "summarized_data_description": summarized_data,
                "summarized_submission_format": summarized_submission_format,
                "summarized_targets_modality_maps": summarized_targets_modality_maps,
                "fetched_raw_data": agent.memory.retrieve({MemKey.FETCHED_RAW_DATA: 1.0}),
                "has_sample_submission": agent.memory.retrieve({MemKey.HAS_SAMPLE_SUBMISSION: 1.0}),
                "raw_id_column_name": agent.memory.retrieve({MemKey.RAW_ID_COLUMN_NAME: 1.0}),
                "raw_target_column_names": agent.memory.retrieve({MemKey.RAW_TARGETS_COLUMN_NAMES: 1.0}),
                "sample_submission_head": agent.memory.retrieve({MemKey.SAMPLE_SUBMISSION_HEAD: 1.0}),
                "raw_table_info": raw_table_info,
                "raw_data_dir": self.raw_data_dir
            }
        )
        agent.memory.store(
            summarized_modality_transforms, {self.output_keys[MemKey.SUMMARIZED_TARGETS_MODALITY_TRANSFORMS.value]}
        )
        with open(os.path.join(self.workspace_path, FileMap.MODALITY_TRANSFORMS_DESCRIPTION.value), 'w') as f:
            f.writelines(summarized_modality_transforms)


class RemoveWritePermission(HumanTakeoverCommand):
    """
    DSAgent: Summarize raw description into more concise and useful information.
    """
    raw_data_dir: str
    name: str = "remove_write_permission"
    description: str = "Remove the permission to write / delete anything from raw data folder"

    required_prompt_templates: dict[str, str | None] = {}
    input_keys: dict[str, MemKey] = {}
    output_keys: dict[str, str] = {}

    def func(self, agent, *args, **kwargs) -> None:
        try:
            recursive_chmod(path=self.raw_data_dir, mode=0o555)
        except PermissionError as e:
            print(e)
            print("Skipping `chmod` for this directory", self.raw_data_dir)


class SummarizeCode(HumanTakeoverCommand):
    name: str = "summarize_generated_code"
    description: str = "Crate a summary of the generated code."

    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.CODE.value: MemKey.CODE,
        MemKey.CODE_OUTPUT.value: MemKey.CODE_OUTPUT,
        MemKey.CODE_ERROR.value: MemKey.CODE_ERROR,
        MemKey.CODE_RAN.value: MemKey.CODE_RAN,
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED
    }
    output_keys: dict[str, MemKey] = {MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY}
    max_retries: int = 5
    parse_func_id: str = "extract_summary_as_json"
    human_takeover: bool = False

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        summary = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["summarize_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message='Your response did not follow the required format'
                                 '\n```json\n{\n\t"summary": "<summary>"\n}\n```. Correct it now.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=summary, tags=self.output_keys[MemKey.CODE_SUMMARY.value])


class SummarizeUnitTestError(HumanTakeoverCommand):
    name: str = "summarize_unit_test_error"
    description: str = "Crate a summary of an error that occurred after running a unit test."

    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        "unit_test_output": MemKey.UNIT_TEST_OUTPUT,
        "unit_test_error": MemKey.UNIT_TEST_ERROR,
        "unit_test_ran": MemKey.UNIT_TEST_RAN,
        "unit_test_passed": MemKey.UNIT_TEST_PASSED,
    }
    output_keys: dict[str, MemKey] = {"uni_test_error_summary": MemKey.UNIT_TEST_ERROR_SUMMARY}
    max_retries: int = 5
    parse_func_id: str = "extract_summary_as_json"
    human_takeover: bool = False

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        summary = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["summarize_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message='Your response did not follow the required format'
                                 '\n```json\n{\n\t"summary": "<summary>"\n}\n```. Correct it now.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=summary, tags=self.output_keys["code_summary"])


class CreateDataPrepPlan(HumanTakeoverCommand):
    name: str = "create_data_prep_plan"
    description: str = "Create DataPrepPlan from summarized data descriptions"

    required_prompt_templates: dict[str, str]

    input_keys: dict[str, MemKey] = {
        memkey.value: memkey for memkey in [
            MemKey.SUMMARIZED_TASK_DESCRIPTION, MemKey.SUMMARIZED_SUBMISSION_FORMAT, MemKey.RAW_DATA_DESCRIPTION,
            MemKey.SUMMARIZED_METRIC_DESCRIPTION, MemKey.SUMMARIZED_SUBMISSION_FORMAT,
            MemKey.SUMMARIZED_DATA_DESCRIPTION,
            MemKey.SUMMARIZED_INPUTS_MODALITY_MAPS,
            MemKey.SUMMARIZED_TARGETS_MODALITY_MAPS,
            MemKey.SUMMARIZED_TARGETS_MODALITY_TRANSFORMS, MemKey.UNIT_TESTS,
            MemKey.WORKSPACE_PATH, MemKey.TEMPLATES_RELATIVE_PATH,
            MemKey.PATH_TO_PYTHON
        ]
    }

    output_keys: dict[str, MemKey] = {
        MemKey.DATA_PREP_PLAN.value: MemKey.DATA_PREP_PLAN,
        MemKey.INPUT_RAG_EXAMPLES.value: MemKey.INPUT_RAG_EXAMPLES,
        MemKey.TARGET_RAG_EXAMPLES.value: MemKey.TARGET_RAG_EXAMPLES,
        MemKey.METRIC_RAG_EXAMPLES.value: MemKey.METRIC_RAG_EXAMPLES,
        MemKey.SUBMISSION_FORMAT_RAG_EXAMPLES.value: MemKey.SUBMISSION_FORMAT_RAG_EXAMPLES,

    }
    workspace_path: str = None
    max_retries: int = 5
    stage_max_retries: int = 5
    human_takeover: bool = False
    db_faiss_k: int = 0
    path_to_saved_responses: str | None = None

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        input_template = self.required_prompt_templates["input_stages"]
        target_template = self.required_prompt_templates["target_stages"]
        transform_template = self.required_prompt_templates["transform_stages"]

        input_stages_prompt_kwargs = {
            "task_description": agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_TASK_DESCRIPTION.value]),
            "input_modalities": agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_INPUTS_MODALITY_MAPS.value]),
        }
        input_stages_json = safe_parsing_chat_completion(
            agent=agent,
            ask_template=input_template,
            prompt_kwargs=input_stages_prompt_kwargs,
            parse_func=extract_json_with_bools,
            format_error_message='Your response did not follow the JSON format \n```json\n...\n```\n'
                                 'Correct it now and use python boolean elements only in the values of the JSON.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
            path_to_saved_responses=self.path_to_saved_responses
        )

        target_stages_prompt_kwargs = {
            "task_description": agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_TASK_DESCRIPTION.value]),
            "target_modalities": agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_TARGETS_MODALITY_MAPS.value]),
        }
        target_stages_json = safe_parsing_chat_completion(
            agent=agent,
            ask_template=target_template,
            prompt_kwargs=target_stages_prompt_kwargs,
            parse_func=extract_json_with_bools,
            format_error_message='Your response did not follow the JSON format \n```json\n...\n```\n'
                                 'Correct it now and use python boolean elements only in the values of the JSON.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
            path_to_saved_responses=self.path_to_saved_responses
        )
        expected_keys_set = {"tabular_targets_needed", "image_targets_needed", "text_targets_needed"}
        assert set(target_stages_json) == expected_keys_set, (target_stages_json, expected_keys_set)

        # do not check for transform only tabular targets are needed
        if target_stages_json["image_targets_needed"] or target_stages_json["text_targets_needed"]:
            transform_stages_prompt_kwargs = {
                "task_description": agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_TASK_DESCRIPTION.value]),
                "submission_format": agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_SUBMISSION_FORMAT.value]),
                "transform_modalities": agent.memory.retrieve(
                    self.input_keys[MemKey.SUMMARIZED_TARGETS_MODALITY_TRANSFORMS.value]
                ),
            }
            transform_stages_json = safe_parsing_chat_completion(
                agent=agent,
                ask_template=transform_template,
                prompt_kwargs=transform_stages_prompt_kwargs,
                parse_func=extract_json_with_bools,
                format_error_message='Your response did not follow the JSON format \n```json\n...\n```\n'
                                     'Correct it now and use python boolean elements only in the values of the JSON.',
                max_retries=self.max_retries,
                human_takeover=self.check_trigger_human_takeover(),
                path_to_saved_responses=self.path_to_saved_responses
            )
        else:
            transform_stages_json = {
                'tabular_targets_transform_needed': True,
                'image_targets_transform_needed': False,
                'text_targets_transform_needed': False
            }

        # Create plan and env in task
        stages_dict = {}
        stages_dict.update(input_stages_json)
        stages_dict.update(target_stages_json)
        stages_dict.update(transform_stages_json)
        plan = DataPrepPlan.create_plan_from_dict(
            stage_name_dict=stages_dict,
            unit_tests=agent.memory.retrieve(self.input_keys[MemKey.UNIT_TESTS.value]),
            workspace_path=agent.memory.retrieve(self.input_keys[MemKey.WORKSPACE_PATH.value]),
            templates_relative_path=agent.memory.retrieve(self.input_keys[MemKey.TEMPLATES_RELATIVE_PATH.value]),
            path_to_python=agent.memory.retrieve(self.input_keys[MemKey.PATH_TO_PYTHON.value]),
        )
        agent.memory.store(plan, self.output_keys[MemKey.DATA_PREP_PLAN.value])
        agent.task.reset_env(plan=plan, stage_max_retries=self.stage_max_retries)

        if DB_FAISS.started and self.db_faiss_k > 0:
            input_modality = "nlp" if plan.has_txt_input else "image" if plan.has_img_input else "tabular"
            target_modality = "nlp" if plan.has_txt_target else "image" if plan.has_img_target else "tabular"

            query = agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_DATA_DESCRIPTION.value])
            input_docs = DB_FAISS.retrieve_input_top_k_documents(query, k=self.db_faiss_k, modality=input_modality)
            target_docs = DB_FAISS.retrieve_target_top_k_documents(
                query, k=self.db_faiss_k,
                modality=target_modality
            )

            formatted_input_examples = VectorFaissDB.format_docs_to_str(input_docs)
            formatted_target_examples = VectorFaissDB.format_docs_to_str(target_docs)

            agent.memory.store(formatted_input_examples, self.output_keys[MemKey.INPUT_RAG_EXAMPLES.value])
            agent.memory.store(formatted_target_examples, self.output_keys[MemKey.TARGET_RAG_EXAMPLES.value])

            metric_query = agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_METRIC_DESCRIPTION.value])
            metric_docs = DB_FAISS.retrieve_metric_top_k_documents(metric_query, k=self.db_faiss_k)
            formatted_metric_examples = VectorFaissDB.format_docs_to_str(metric_docs)
            agent.memory.store(formatted_metric_examples, self.output_keys[MemKey.METRIC_RAG_EXAMPLES.value])

            submission_format_query = agent.memory.retrieve(
                self.input_keys[MemKey.SUMMARIZED_SUBMISSION_FORMAT.value]
            )
            submission_format_docs = DB_FAISS.retrieve_submission_format_top_k_documents(
                submission_format_query,
                k=self.db_faiss_k
            )
            formatted_submission_format_examples = VectorFaissDB.format_docs_to_str(submission_format_docs)
            agent.memory.store(
                formatted_submission_format_examples,
                self.output_keys[MemKey.SUBMISSION_FORMAT_RAG_EXAMPLES.value]
            )


def fill_template_inplace(
        agent_code: list[str],
        templates_relative_path: str,
        code_template: str,
        fim_hole_token: str,
) -> Tuple[str, str]:
    """
    Fills the code template with the code snippets from the agent.
    If not same number of code snippets provided as number of holes to fill, returns an error message and no code.
    """
    with open(os.path.join(templates_relative_path, code_template), 'r') as f:
        code_template = f.read()
    code_template_split = code_template.split(f'# {fim_hole_token}')

    err = ""
    if len(code_template_split) != len(agent_code) + 1:
        n_holes = len(code_template_split) - 1
        err = (f"Number of holes to fill in template ({n_holes}) "
               f"doesn't match number of codes provided by agent ({len(agent_code)}).\n"
               f"Please retry and output {n_holes} code snippets, i.e.\n")
        example = [f"```python\ncode{i + 1}\n```" for i in range(n_holes)]
        example = "\n\n".join(example) + "\n\n..."
        err += example
        print(err)

    if err != "":
        return "", err
    else:
        code = code_template_split[0]
        for mc, ac in zip(code_template_split[1:], agent_code):
            code += ac + mc

        return code, err


def generate_and_run_code(
        agent: LLMAgent,
        input_keys: dict[str, Any],
        code_prompt_template: str,
        code_filename: str,
        code_template_key: str,
        max_retries: int = 5,
        fill_template: bool = True,
        human_takeover: bool | None = False,
        prompt_kwargs: dict[str, Any] = None,
        specialized_llm_name: str | None = None,
) -> dict[str, Any]:
    """
    Call the LLM to generate the code completing the associated template, run the generated code from the workspace,
     and finally return the output or any errors encountered along the way.
    """
    return_dict: dict[str, str | bool | None] = {
        "code": None,
        "code_output": None,
        "code_error": None,
        "code_ran": False,
        "code_passed": False,
    }

    # retrieve code snippets from related stages of the Plan
    plan = agent.memory.retrieve(input_keys["data_prep_plan"])
    stages_code_dict = plan.get_stages_code()
    generate_code_prompt_kwargs = {"stages_code_dict": stages_code_dict}

    if prompt_kwargs:
        generate_code_prompt_kwargs.update(prompt_kwargs)

    agent_code = safe_parsing_chat_completion(
        agent=agent,
        ask_template=code_prompt_template,
        prompt_kwargs=generate_code_prompt_kwargs,
        parse_func=extract_python,
        format_error_message="Format Error: Your code was not in the correct format\n```python\ncode\n```\n",
        max_retries=max_retries,
        human_takeover=human_takeover,
        specialized_llm_name=specialized_llm_name,
    )

    if fill_template:
        code, error = fill_template_inplace(
            agent_code=[agent_code],
            templates_relative_path=agent.memory.retrieve(input_keys["templates_relative_path"]),
            code_template=agent.memory.retrieve(input_keys["code_templates"])[code_template_key],
            fim_hole_token=agent.memory.retrieve(input_keys["fim_tokens"])["fim_hole_token"],
        )
        if len(code) == 0:
            return_dict["code"] = f"```python\n{code}\n```"
            return_dict["code_ran"] = True
            return_dict["code_error"] = error
            return return_dict
    else:
        # agent returns whole code directly
        code = agent_code

    check_code_safety(code)

    workspace_path = agent.memory.retrieve(input_keys["workspace_path"])
    path_to_python = agent.memory.retrieve(input_keys["path_to_python"])
    code_path = os.path.join(workspace_path, code_filename)
    code_output_path = os.path.join(workspace_path, "_code_output.txt")
    code_error_path = os.path.join(workspace_path, "_code_error.txt")
    code_warnings_path = os.path.join(workspace_path, "_code_warnings.txt")
    aux_code_error_path = os.path.join(workspace_path, "_aux_code_error.txt")

    # save code to python file
    wrapped_code = catch_error_wrap(code=code, code_error_path=code_error_path, code_warnings_path=code_warnings_path)
    with open(code_path, "w") as f:
        f.writelines(wrapped_code)

    # Run code
    code_output, code_warnings, code_error = run_python_code(
        workspace_path=workspace_path,
        path_to_python=path_to_python,
        code_path=code_path,
        code_output_path=code_output_path,
        code_warnings_path=code_warnings_path,
        code_error_path=code_error_path,
        aux_code_error_path=aux_code_error_path,
    )

    return_dict["code"] = f"```python\n{code}\n```"
    return_dict["code_ran"] = True

    if code_error == "":
        print("Code ran successfully.", flush=True)
        if code_warnings != "":
            print(code_warnings)
        return_dict["code_output"] = code_output
        return_dict["code_error"] = None
        return_dict["code_passed"] = True
    else:
        print(f"Error in code:\n\n{code_error}", flush=True)
        return_dict["code_output"] = None
        return_dict["code_error"] = code_error
        return_dict["code_passed"] = False

    return return_dict


class CreateCodeCommand(HumanTakeoverCommand, abc.ABC):
    name: str = "create_code"
    description: str = "Generate code abstract base class."

    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.PATH_TO_PYTHON.value: MemKey.PATH_TO_PYTHON,
        MemKey.WORKSPACE_PATH.value: MemKey.WORKSPACE_PATH,
        MemKey.TEMPLATES_RELATIVE_PATH.value: MemKey.TEMPLATES_RELATIVE_PATH,
        MemKey.CODE_TEMPLATES.value: MemKey.CODE_TEMPLATES,
        MemKey.FIM_TOKENS.value: MemKey.FIM_TOKENS,
        MemKey.DATA_PREP_PLAN.value: MemKey.DATA_PREP_PLAN,
        MemKey.TURN_OFF_RAG_LOCALLY_ON_FAILURE.value: MemKey.TURN_OFF_RAG_LOCALLY_ON_FAILURE,
        MemKey.N_FAILURES_WITH_RAG_ACTIVE.value: MemKey.N_FAILURES_WITH_RAG_ACTIVE,
    }
    output_keys: dict[str, MemKey] = {
        MemKey.CODE.value: MemKey.CODE,
        MemKey.CODE_RAN.value: MemKey.CODE_RAN,
        MemKey.CODE_OUTPUT.value: MemKey.CODE_OUTPUT,
        MemKey.CODE_ERROR.value: MemKey.CODE_ERROR,
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED,
        MemKey.RAG_KEY.value: MemKey.RAG_KEY,
        MemKey.RAG_LOCALLY_ACTIVE.value: MemKey.RAG_LOCALLY_ACTIVE,
    }

    specification: str
    fill_template: bool
    max_retries: int = 5
    human_takeover: bool = False
    specialized_llm_name: str | None = None

    @property
    def code_filename(self):
        return f"code_{self.specification}.py"

    def generate_and_run_kwargs(self, agent: LLMAgent, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Extra kwargs for the generate_and_run call to be overridden by subclasses"""
        return dict()

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        return_dict = generate_and_run_code(
            agent=agent,
            input_keys=self.input_keys,
            code_prompt_template=self.required_prompt_templates["code_prompt_template"],
            code_filename=self.code_filename,
            code_template_key=self.specification,
            max_retries=self.max_retries,
            fill_template=self.fill_template,
            human_takeover=self.check_trigger_human_takeover(),
            specialized_llm_name=self.specialized_llm_name,
            **self.generate_and_run_kwargs(agent, *args, **kwargs)
        )

        agent.memory.store(content=return_dict["code"], tags=self.output_keys[MemKey.CODE.value])
        returned_code_output = return_dict["code_output"]
        if returned_code_output is not None and len(returned_code_output) > 0:
            code_output_lines = return_dict["code_output"].split("\n")
            code_output = ""
            if len(code_output_lines) > 100:
                code_output += "<truncated output>...\n"
                code_output_lines = code_output_lines[-100:]
            code_output += "\n".join(code_output_lines)
        else:
            code_output = returned_code_output
        agent.memory.store(content=code_output, tags=self.output_keys[MemKey.CODE_OUTPUT.value])
        agent.memory.store(content=return_dict["code_error"], tags=self.output_keys[MemKey.CODE_ERROR.value])
        agent.memory.store(content=return_dict["code_ran"], tags=self.output_keys[MemKey.CODE_RAN.value])
        agent.memory.store(content=return_dict["code_passed"], tags=self.output_keys[MemKey.CODE_PASSED.value])

        # Also store code in RAG_KEY for lookup
        if DB_FAISS.started and DB_FAISS.embedded_field == "code":
            # If DB_FAISS.embedded_field=='code_error' it is only supposed to relate to errors in the unit tests
            # so the RAG_KEY will be set later in the DataPreprocessingTask
            agent.memory.store(content=return_dict["code"], tags=self.output_keys[MemKey.RAG_KEY.value])

        # set RAG_LOCALLY_ACTIVE and N_FAILURES_WITH_RAG_ACTIVE
        turn_off_rag_locally_on_failure = agent.memory.retrieve(
            self.input_keys[MemKey.TURN_OFF_RAG_LOCALLY_ON_FAILURE.value]
        )
        n_failures_with_rag_active = agent.memory.retrieve(self.input_keys[MemKey.N_FAILURES_WITH_RAG_ACTIVE.value])
        if turn_off_rag_locally_on_failure:
            if return_dict["code_passed"]:
                agent.memory.store(content=True, tags=self.output_keys[MemKey.RAG_LOCALLY_ACTIVE.value])
                agent.memory.store(content=0, tags=self.input_keys[MemKey.N_FAILURES_WITH_RAG_ACTIVE.value])
            else:
                if n_failures_with_rag_active >= 3:
                    agent.memory.store(content=False, tags=self.output_keys[MemKey.RAG_LOCALLY_ACTIVE.value])
                agent.memory.store(
                    content=n_failures_with_rag_active + 1,
                    tags=self.input_keys[MemKey.N_FAILURES_WITH_RAG_ACTIVE.value]
                )


class CreateDataMapOrTransform(CreateCodeCommand):
    name: str = "create_data_map_or_tf"
    description: str = "Generate code to create a data map or the transform/inverse transform of the train targets."


class CreateMetric(CreateCodeCommand):
    name: str = "create_metric_command"
    description: str = "Create Metric function"

    specification: str = "metric"
    fill_template: bool = True

    @property
    def code_filename(self):
        return FileMap.METRIC_SCRIPT.value


class CreateColumnTypes(CreateCodeCommand):
    name: str = "create_column_types"
    description: str = "Create Column Types of train maps"

    specification: str = "column_types"
    fill_template: bool = True


class CreateColumnTargetTransform(HumanTakeoverCommand):
    name: str = "select_column_target_for_transform"
    description: str = "Create a json dictionary mapping each column target to their apply transform "

    tf_specification: str
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.SUMMARIZED_SUBMISSION_FORMAT.value: MemKey.SUMMARIZED_SUBMISSION_FORMAT,
        MemKey.SUMMARIZED_METRIC_DESCRIPTION.value: MemKey.SUMMARIZED_METRIC_DESCRIPTION,
        MemKey.SUMMARIZED_DATA_DESCRIPTION.value: MemKey.SUMMARIZED_DATA_DESCRIPTION,
        MemKey.WORKSPACE_PATH.value: MemKey.WORKSPACE_PATH
    }
    max_retries: int = 5
    human_takeover: bool = False

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        workspace_path = agent.memory.retrieve(self.input_keys[MemKey.WORKSPACE_PATH.value])
        train_tab_target_map_path = os.path.join(workspace_path, 'train_tab_target_map.csv')
        train_tab_target_map_df = pd.read_csv(train_tab_target_map_path)
        target_columns_names = train_tab_target_map_df.columns.tolist()
        target_clf_column_names = [colname for colname in target_columns_names if colname.endswith("_classification")]
        train_tab_target_map_info = GetTableView.get_raw_table_info(train_tab_target_map_path)
        metric_summary = agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_METRIC_DESCRIPTION.value])
        data_description = agent.memory.retrieve(self.input_keys[MemKey.SUMMARIZED_DATA_DESCRIPTION.value])

        if len(target_clf_column_names) > 0:
            prompt_args = {k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys}
            prompt_args['target_clf_column_names'] = target_clf_column_names
            prompt_args['train_tab_target_map_info'] = train_tab_target_map_info
            prompt_args['summarized_data_description'] = data_description
            prompt_args['summarized_metric_description'] = metric_summary

            target_columns_transform_json = safe_parsing_chat_completion(
                agent=agent,
                ask_template=self.required_prompt_templates["target_columns_transform_template"],
                prompt_kwargs=prompt_args,
                parse_func=extract_json,
                format_error_message='Your response did not follow the required format'
                                     '\n```json\n...\n```\n. Correct it now.',
                max_retries=self.max_retries,
                human_takeover=self.check_trigger_human_takeover(),
            )

        else:
            target_columns_transform_json = {}

        with open(os.path.join(workspace_path, FileMap.TARGET_COL_CLASSIFICATION_TRANSFORMS.value), 'w') as f:
            json.dump(target_columns_transform_json, f)

        code_tab_transform_template_path = agent.memory.retrieve(MemKey.CODE_TEMPLATES)[self.tf_specification]
        templates_relative_path = agent.memory.retrieve(MemKey.TEMPLATES_RELATIVE_PATH)

        with open(os.path.join(templates_relative_path, code_tab_transform_template_path), 'r') as f:
            code_tab_transform_template = f.read()

        code_path = os.path.join(workspace_path, f'code_{self.tf_specification}.py')
        with open(code_path, 'w') as f:
            f.write(code_tab_transform_template)


class StoreInverseTransformInMemory(HumanTakeoverCommand):
    name: str = " store_metadata_inv_tab_transform"
    description: str = " Store inverse transform metadata in memory"

    input_keys: dict[str, MemKey] = {
        MemKey.WORKSPACE_PATH.value: MemKey.WORKSPACE_PATH,
        MemKey.DATA_PREP_PLAN.value: MemKey.DATA_PREP_PLAN,
    }

    output_keys: dict[str, MemKey] = {
        MemKey.INV_TRANSFORM_HEAD_VIEW_TAB.value: MemKey.INV_TRANSFORM_HEAD_VIEW_TAB,
        MemKey.INV_TRANSFORM_HEAD_VIEW_TXT.value: MemKey.INV_TRANSFORM_HEAD_VIEW_TXT,
        MemKey.INV_TRANSFORM_HEAD_VIEW_IMG.value: MemKey.INV_TRANSFORM_HEAD_VIEW_IMG,
        MemKey.PRED_TABLE_INFO_TAB.value: MemKey.PRED_TABLE_INFO_TAB,
        MemKey.PRED_TABLE_INFO_TXT.value: MemKey.PRED_TABLE_INFO_TXT,
        MemKey.PRED_TABLE_INFO_IMG.value: MemKey.PRED_TABLE_INFO_IMG,
    }
    view_n_rows: int = 2

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        plan = agent.memory.retrieve(self.input_keys[MemKey.DATA_PREP_PLAN.value])
        workspace_path = agent.memory.retrieve(self.input_keys[MemKey.WORKSPACE_PATH.value])
        if plan.has_tab_target:
            df_tab_target_inv_transform_path = os.path.join(workspace_path, FileMap.TAB_INV_TRANSFORM.value)
            df_inv_transform = next(pd.read_csv(df_tab_target_inv_transform_path, chunksize=self.view_n_rows))
            # df_inv_transform_view = df_inv_transform.head(self.view_n_rows)
            df_inv_transform_view = GetTableView.get_table_view(
                table_path=df_tab_target_inv_transform_path, n_rows=self.view_n_rows
            )
            pred_tab_inf = self.get_table_prompt_info(
                df_inv_transform=df_inv_transform, workspace_path=workspace_path
            )
            agent.memory.store(
                content=df_inv_transform_view, tags=self.output_keys[MemKey.INV_TRANSFORM_HEAD_VIEW_TAB.value]
            )
            agent.memory.store(content=pred_tab_inf, tags=self.output_keys[MemKey.PRED_TABLE_INFO_TAB.value])
        if plan.has_txt_target:
            df_txt_target_inv_transform_path = os.path.join(workspace_path, FileMap.TXT_INV_TRANSFORM.value)
            # df_inv_transform = next(pd.read_csv(df_txt_target_inv_transform_path, chunksize=self.view_n_rows))
            # df_inv_transform_view = df_inv_transform.head(self.view_n_rows)
            df_inv_transform_view = GetTableView.get_table_view(
                table_path=df_txt_target_inv_transform_path, n_rows=self.view_n_rows
            )
            # pred_txt_inf = self.get_table_prompt_info(
            #     df_inv_transform=df_inv_transform, workspace_path=workspace_path
            # )
            agent.memory.store(
                content=df_inv_transform_view, tags=self.output_keys[MemKey.INV_TRANSFORM_HEAD_VIEW_TXT.value]
            )
            # agent.memory.store(content=pred_txt_inf, tags=self.output_keys[MemKey.PRED_TABLE_INFO_TXT.value])
        if plan.has_img_target:
            df_img_target_inv_transform_path = os.path.join(workspace_path, FileMap.IMG_INV_TRANSFORM.value)
            # df_inv_transform = next(pd.read_csv(df_img_target_inv_transform_path, chunksize=self.view_n_rows))
            # df_inv_transform_view = df_inv_transform.head(self.view_n_rows)
            df_inv_transform_view = GetTableView.get_table_view(
                table_path=df_img_target_inv_transform_path, n_rows=self.view_n_rows
            )
            # pred_img_inf = self.get_table_prompt_info(
            #     df_inv_transform=df_inv_transform, workspace_path=workspace_path
            # )
            agent.memory.store(
                content=df_inv_transform_view, tags=self.output_keys[MemKey.INV_TRANSFORM_HEAD_VIEW_IMG.value]
            )
            # agent.memory.store(content=pred_img_inf, tags=self.output_keys[MemKey.PRED_TABLE_INFO_IMG.value])

    @staticmethod
    def get_table_prompt_info(df_inv_transform: pd.DataFrame, workspace_path: str) -> str:
        with open(os.path.join(workspace_path, FileMap.TARGET_COL_CLASSIFICATION_TRANSFORMS.value), "r") as f:
            classification_cols = json.load(f)
        predict_table_info = "A prediction dataframe has the following columns:\n"

        for colname, inv_transf_type in classification_cols.items():
            colname_key = colname
            colname = colname[:colname.rfind("_classification")]
            if classification_cols[colname_key] == "proba":
                predict_table_info += (f"\t- For classification target {colname}: `{colname}_<CLASS>` "
                                       f"contain the probabilities for the entry to belong to the class <CLASS>.")
            elif classification_cols[colname_key].startswith("top_k"):
                predict_table_info += f"\t- For classification target {colname}: the column {colname} contains "
                k = int(classification_cols[colname_key].split("(")[1].split(",")[0])
                if k == 1:
                    predict_table_info += "the names of the predicted classes for each entry."
                else:
                    predict_table_info += f"a list with the top-{k} most likely classes for each entry."
            else:
                raise ValueError(classification_cols[colname])
            predict_table_info += "\n"

        for colname in df_inv_transform.columns:
            colname_key = colname + "_classification"
            if colname_key in classification_cols:
                continue
            elif colname != "id":
                predict_table_info += (f"\t-For regression target `{colname}`, column `{colname}`"
                                       f" contains the predicted values.")

            predict_table_info += "\n"
        return predict_table_info


class StoreTabTrainInMemory(HumanTakeoverCommand):
    name: str = "store_tab_train"
    description: str = " Store example rows of tab train map in memory"

    input_keys: dict[str, MemKey] = {MemKey.WORKSPACE_PATH.value: MemKey.WORKSPACE_PATH}

    output_keys: dict[str, MemKey] = {
        MemKey.TAB_TRAIN_COLUMNS.value: MemKey.TAB_TRAIN_COLUMNS,
        MemKey.TAB_TRAIN_INFO.value: MemKey.TAB_TRAIN_INFO
    }
    view_n_rows: int = 5

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        workspace_path = agent.memory.retrieve(self.input_keys[MemKey.WORKSPACE_PATH.value])
        df_tab_train_path = os.path.join(workspace_path, 'train_tab_input_map.csv')
        # df_sample = next(pd.read_csv(df_tab_train_path, index_col=0, chunksize=self.view_n_rows))
        # info = str(df_sample.dtypes)
        # df_sample_view = df_sample.head(self.view_n_rows).__repr__()
        df_sample_view = GetTableView.get_table_view(df_tab_train_path, n_rows=self.view_n_rows)
        df_info = GetTableView.get_raw_table_info(df_tab_train_path)
        agent.memory.store(content=df_sample_view, tags=self.output_keys[MemKey.TAB_TRAIN_COLUMNS.value])
        agent.memory.store(content=df_info, tags=self.output_keys[MemKey.TAB_TRAIN_INFO.value])


class StoreColTypesInMemory(HumanTakeoverCommand):
    name: str = "store_col_types"
    description: str = " Store currently created column types json in memory"

    input_keys: dict[str, MemKey] = {
        MemKey.WORKSPACE_PATH.value: MemKey.WORKSPACE_PATH,
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED
    }

    output_keys: dict[str, MemKey] = {
        MemKey.VALID_COLUMN_TYPES.value: MemKey.VALID_COLUMN_TYPES
    }

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        code_passed = agent.memory.retrieve(self.input_keys[MemKey.CODE_PASSED.value])
        workspace_path = agent.memory.retrieve(self.input_keys[MemKey.WORKSPACE_PATH.value])
        col_types_path = os.path.join(workspace_path, 'metadata/column_types.json')
        if code_passed and os.path.exists(col_types_path):
            with open(col_types_path, "r") as file:
                fileData = file.read()
                jsonData = json.loads(fileData)
            content = str(json.dumps(jsonData, indent=2))
            if len(content.split("\n")) > 100:
                content = content[:50] + '\n...\n' + content[-50:]
            agent.memory.store(content=content, tags=self.output_keys[MemKey.VALID_COLUMN_TYPES.value])


class CreateSubmissionFormat(CreateCodeCommand):
    name: str = "create_submission_format"
    description: str = "Create Submission format function"

    input_keys: dict[str, MemKey] = {
        **CreateCodeCommand.__fields__['input_keys'].default,  # Include superclass input_keys
        MemKey.INV_TRANSFORM_HEAD_VIEW_TAB.value: MemKey.INV_TRANSFORM_HEAD_VIEW_TAB,
        MemKey.INV_TRANSFORM_HEAD_VIEW_TXT.value: MemKey.INV_TRANSFORM_HEAD_VIEW_TXT,
        MemKey.INV_TRANSFORM_HEAD_VIEW_IMG.value: MemKey.INV_TRANSFORM_HEAD_VIEW_IMG,
        MemKey.PRED_TABLE_INFO_TAB.value: MemKey.PRED_TABLE_INFO_TAB,
        MemKey.PRED_TABLE_INFO_TXT.value: MemKey.PRED_TABLE_INFO_TXT,
        MemKey.PRED_TABLE_INFO_IMG.value: MemKey.PRED_TABLE_INFO_IMG,
    }

    specification: str = "submission_format"
    fill_template: bool = False

    def generate_and_run_kwargs(self, agent: LLMAgent, *args: Any, **kwargs: Any) -> dict[str, Any]:
        df_inv_transform_view_tab = agent.memory.retrieve(self.input_keys[MemKey.INV_TRANSFORM_HEAD_VIEW_TAB.value])
        df_inv_transform_view_txt = agent.memory.retrieve(self.input_keys[MemKey.INV_TRANSFORM_HEAD_VIEW_TXT.value])
        df_inv_transform_view_img = agent.memory.retrieve(self.input_keys[MemKey.INV_TRANSFORM_HEAD_VIEW_IMG.value])
        predict_table_info_tab = agent.memory.retrieve(self.input_keys[MemKey.PRED_TABLE_INFO_TAB.value])
        predict_table_info_txt = agent.memory.retrieve(self.input_keys[MemKey.PRED_TABLE_INFO_TXT.value])
        predict_table_info_img = agent.memory.retrieve(self.input_keys[MemKey.PRED_TABLE_INFO_IMG.value])

        return dict(
            prompt_kwargs={
                'df_inv_transform_view_tab': df_inv_transform_view_tab,
                'df_inv_transform_view_txt': df_inv_transform_view_txt,
                'df_inv_transform_view_img': df_inv_transform_view_img,
                'predict_table_info_tab': predict_table_info_tab,
                'predict_table_info_txt': predict_table_info_txt,
                'predict_table_info_img': predict_table_info_img,
            }
        )

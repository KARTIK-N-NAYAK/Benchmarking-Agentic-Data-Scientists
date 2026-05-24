import os
from pathlib import Path
from typing import Any, Tuple, Type

from ds_agent.rag import DB_FAISS
from ds_agent.utils import SubmissionFormatError, save_w_pickle

from agent.memory import MemKey
from agent.tasks.datascience_task.utils import FileMap
from agent.tasks.tasks import Task
from agent.utils import pylogger
from agent.utils.utils import ListableEnum, get_path_to_python
from third_party.data_preprocessing.env import (
    DataPrepAction,
    DataPrepEnv,
    DataPrepPlan,
    DataPrepStageName,
    ModalityIdentificationError,
    DataPrepTestStageName, DataPrepStage,
)
from third_party.data_science.utils import get_agent_root_dir, get_raw_data_root_dir

log = pylogger.get_pylogger(__name__)
WORKSPACE_ROOT = os.path.join(str(Path(__file__).parent.parent.parent), "workspace")


def custom_input(message: str, default: str) -> str:
    """Wrapper around `input` function.

    Replace empty string by `default`
    """
    if os.getenv("AGENT_DEBUG", False):
        val = default
    else:
        val = input(message)
        if val == "":
            val = default
    return val


class CodeTemplateKeys(ListableEnum):
    MAP_TAB_INPUT_TRAIN = "map_tab_input_train"
    MAP_IMG_INPUT_TRAIN = "map_img_input_train"
    MAP_TXT_INPUT_TRAIN = "map_txt_input_train"
    MAP_TAB_TARGET_TRAIN = "map_tab_target_train"
    MAP_IMG_TARGET_TRAIN = "map_img_target_train"
    MAP_TXT_TARGET_TRAIN = "map_txt_target_train"
    TF_TAB_TARGET_TRAIN = "transform_tab_target_train"
    TF_IMG_TARGET_TRAIN = "transform_img_target_train"
    TF_TXT_TARGET_TRAIN = "transform_txt_target_train"
    MAP_TAB_INPUT_TEST = "map_tab_input_test"
    MAP_IMG_INPUT_TEST = "map_img_input_test"
    MAP_TXT_INPUT_TEST = "map_txt_input_test"
    METRIC = "metric"
    RAMP_METRIC = "ramp_metric"
    SUBMISSION_FORMAT = "submission_format"
    SUBMISSION_FORMAT_ALT = "submission_format_alt"
    COLUMN_TYPES = "column_types"


class UnitTestKeys(ListableEnum):
    MAP_TAB_INPUT_TRAIN = "map_tab_input_train"
    MAP_IMG_INPUT_TRAIN = "map_img_input_train"
    MAP_TXT_INPUT_TRAIN = "map_txt_input_train"
    MAP_TAB_TARGET_TRAIN = "map_tab_target_train"
    MAP_IMG_TARGET_TRAIN = "map_img_target_train"
    MAP_TXT_TARGET_TRAIN = "map_txt_target_train"
    TF_TAB_TARGET_TRAIN = "transform_tab_target_train"
    TF_IMG_TARGET_TRAIN = "transform_img_target_train"
    TF_TXT_TARGET_TRAIN = "transform_txt_target_train"
    MAP_TAB_INPUT_TEST = "map_tab_input_test"
    MAP_IMG_INPUT_TEST = "map_img_input_test"
    MAP_TXT_INPUT_TEST = "map_txt_input_test"
    MAP_TAB_TRAIN = "map_tab_train"
    MAP_IMG_TRAIN = "map_img_train"
    MAP_TXT_TRAIN = "map_txt_train"
    MAP_ALL_TRAIN = "map_all_train"
    DATALOADER_TRAIN = "dataloader_train"
    DATALOADER_TEST = "dataloader_test"
    METRIC = "metric"
    SUBMISSION_FORMAT = "submission_format"
    SUBMISSION_FORMAT_ALT = "submission_format_alt"
    COLUMN_TYPES = "column_types"


class FIMTokenKeys(ListableEnum):
    FIM_START_TOKEN = "fim_start_token"
    FIM_HOLE_TOKEN = "fim_hole_token"
    FIM_END_TOKEN = "fim_end_token"


class DataPreProcessing(Task):
    def __init__(
            self,
            task_id: str,
            task_url: str,
            workspace_path: str,
            raw_data_dir_name: str,
            name: str,
            version: str,
            path_to_python: str,
            description: str | None = None,
            subtask: str | None = None,
            code_templates: dict[str, str] | None = None,
            unit_tests: dict[str, list[str]] | None = None,
            modalities: dict[str, list[str]] | None = None,
            sample_submission_file: str | None = None,
            human_takeover_step: int = 5,
            use_final_unit_test: bool = False,
            summarize_cot: bool = False,
            **kwargs,
    ):
        """
        Args:
            task_url: (str) the URL of the competition
            task_id: (str) the competition ID
            workspace_path: (str) the path to the current workspace
            raw_data_dir_name: (str) the path to the raw data directory
            code_templates: (dict[str, str]) a dictionary of one code template per stage
            unit_tests: (dict[str, list[str]]) a dict of one or more unit test(s) per stage
            modalities: a dict containing the list of input modalities and target modalities (if provided then the code
                        will break if the Agent does not identify the correct modalities).
            sample_submission_file: name of the file containing the sample submission
        """
        if description is None:
            description = ""
        if subtask is None:
            subtask = os.path.basename(task_url)
        super().__init__(name=name, subtask=subtask, description=description, version=version, **kwargs)
        self.max_rounds = 20
        self.task_url = task_url
        self.task_id = task_id
        self.workspace_path = os.path.abspath(workspace_path)
        self.raw_data_dir = str(get_raw_data_root_dir() / raw_data_dir_name)
        self.path_to_python = get_path_to_python(path_to_python)
        self.code_templates = code_templates
        self.unit_tests = unit_tests
        self.use_final_unit_test = use_final_unit_test
        self.summarize_cot = summarize_cot
        self.templates_relative_path = kwargs.get("templates_relative_path", None)
        self.fim_tokens = kwargs.get("fim_tokens", None)

        self.done = False
        self.step_num = 0
        self.env = None
        self.obs = None

        self.check_keys(dict_obj=self.code_templates, keys_enum=CodeTemplateKeys)
        self.check_keys(dict_obj=self.unit_tests, keys_enum=UnitTestKeys)
        self.check_keys(dict_obj=self.fim_tokens, keys_enum=FIMTokenKeys)

        # we use this to trigger human takeover if we repetitively return to the same stage
        self.human_takeover_step = human_takeover_step
        self.action_counter: dict[DataPrepStageName, int] = {}
        self.modalities = modalities
        self.sample_submission_file = sample_submission_file
        self.show_sample_submission_in_code = kwargs.get('show_sample_submission_in_code', False)
        self.turn_off_rag_locally_on_failure = kwargs.get('turn_off_rag_locally_on_failure', False)
        self.db_embedding_field = kwargs.get('db_embedding_field', None)
        self.n_unit_test_failed_with_rag_active = 0

        # save repo root dir in env variable for possible later use
        # (in particular in dataloader unit tests to be able to load from third_party)
        os.environ["AGENT_PATH"] = str(get_agent_root_dir())

    @staticmethod
    def check_keys(dict_obj, keys_enum: Type[ListableEnum]):
        if dict_obj is not None:
            for k in dict_obj.keys():
                assert k in keys_enum.list(), f"Key {k} not in {keys_enum}"

    def get_dict_cfg_info(self) -> dict[str, Any]:
        """Output a dictionary containing some info about the run."""
        return {
            "max_round": self.max_rounds,
        }

    def reset(self, next_subtask: str | None = None) -> dict[str, str]:
        """Reset the environment and return the initial observation."""

        # create the new workspace
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(f"{self.workspace_path}/data", exist_ok=True)
        os.makedirs(f"{self.workspace_path}/metadata", exist_ok=True)

        # add text file with path to repo root, useful for some scripts that are ran from the workspace directly
        # but import from modules outside the workspace. We cannot guarantee that adding something like
        # sys.path.insert(0, str(Path(__file__).parent.parent.parent...))
        # will always work, in particular as the workspace can be in different locations and not always in a predictable
        # place.
        with open(os.path.join(self.workspace_path, "root_path_to_agent.txt"), "w") as f:
            f.write(str(get_agent_root_dir()))

        save_w_pickle(obj=self.get_dict_cfg_info(), path=self.workspace_path, filename="info")

        self.done = False
        self.step_num = 0
        self.n_unit_test_failed_with_rag_active = 0
        self.obs = {
            MemKey.TASK_ID: self.task_id,
            MemKey.OBSERVATION: "Pre-episode Flow",
            MemKey.RAW_DATA_DIR: self.raw_data_dir,
            MemKey.PATH_TO_PYTHON: self.path_to_python,
            MemKey.TEMPLATES_RELATIVE_PATH: self.templates_relative_path,
            MemKey.CODE_TEMPLATES: self.code_templates,
            MemKey.UNIT_TESTS: self.unit_tests,
            MemKey.WORKSPACE_PATH: self.workspace_path,
            MemKey.FIM_TOKENS: self.fim_tokens,
            MemKey.SHOW_SAMPLE_SUBMISSION_IN_CODE: self.show_sample_submission_in_code,
            MemKey.TURN_OFF_RAG_LOCALLY_ON_FAILURE: self.turn_off_rag_locally_on_failure,
            MemKey.RAG_LOCALLY_ACTIVE: True,
            MemKey.N_FAILURES_WITH_RAG_ACTIVE: 0,
            MemKey.SUMMARIZE_COT: self.summarize_cot
        }

        # actually we WANT to return self.obs here so that on_episode_start_flow calls self.reset_env()
        # and changes self.obs which is accessible as obs in start.py and then observed.
        # This is a way to change the obs in on_episode_start_flow and add new keys to the agent memory easily.
        return self.obs

    def reset_env(self, plan: DataPrepPlan, stage_max_retries: int = 5) -> None:

        self.env = DataPrepEnv(
            task_id=self.task_id,
            workspace_path=self.workspace_path,
            templates_relative_path=self.templates_relative_path,
            path_to_python=self.path_to_python,
            plan=plan,
            use_final_unit_test=self.use_final_unit_test,
            stage_max_retries=stage_max_retries,
        )
        self.save_plan_to_workspace()

        if not self.check_right_modalities(plan=plan):
            raise ModalityIdentificationError(
                f"Wrong modalities identified, expected {self.modalities}, "
                f"got input_modalities: {plan.get_input_modalities()} "
                f"and target_modalities: {plan.get_target_modalities()}"
            )

        self.env.reset()
        self.obs[MemKey.AVAILABLE_ACTIONS] = self.env.get_available_actions()
        self.obs[MemKey.DATA_PREP_PLAN] = self.env.plan
        self.action_counter = {}

    def check_right_modalities(self, plan: DataPrepPlan) -> bool:
        """
        Check if input and target modalities identified by the agent matches the ground-truth ones
        (if they were provided)

        Args:
            plan: the plan generated by the agent
        """
        right_modalities = True
        if self.modalities["inputs"] is not None:
            right_modalities &= set(self.modalities["inputs"]) == set(plan.get_input_modalities())
        if self.modalities["targets"] is not None:
            right_modalities &= set(self.modalities["targets"]) == set(plan.get_target_modalities())
        return right_modalities

    def is_complete(self) -> bool:
        return self.done

    def step(self, action: DataPrepAction, retrial_chat_completion_time: float)\
            -> Tuple[dict[MemKey, Any], float, bool]:
        """Perform an action and return the next observation, reward, and done flag."""
        self.step_num += 1

        if self.env is not None:
            if action.stage_name.to_str() not in self.env.get_available_actions():
                raise RuntimeError(f"{action.stage_name} not in {[f'{a}' for a in self.env.get_available_actions()]}")

            env_obs, reward, done, env_info = self.env.step(
                action=action, retrial_chat_completion_time=retrial_chat_completion_time
            )
            last_seen_stage = env_info.last_seen_stage
            env_info = env_info.info
            self.save_plan_to_workspace()

            # Env_info features a dict of unit test, so we take the last failing one to return to the agent
            final_test_info = env_info.pop("final", None)
            last_unit_test = list(env_info.keys())[-1]
            for unit_test, unit_test_info in env_info.items():
                if not unit_test_info["passed"]:
                    last_unit_test = unit_test
                    break

            unit_test_info = env_info[last_unit_test]

            self.done = done
            env_obs = env_obs.__dict__
            human_takeover = self.check_trigger_human_takeover(action=action) and not unit_test_info["passed"]

            obs = {
                MemKey.REWARD: env_obs["reward"],
                MemKey.AVAILABLE_ACTIONS: self.env.get_available_actions(),
                MemKey.DATA_PREP_PLAN: env_obs["plan"],
                MemKey.UNIT_TEST_OUTPUT: unit_test_info["output"],
                MemKey.UNIT_TEST_ERROR: unit_test_info["error"],
                MemKey.UNIT_TEST_RAN: unit_test_info["ran"],
                MemKey.UNIT_TEST_PASSED: unit_test_info["passed"],
                MemKey.TRIGGER_HUMAN_TAKEOVER: human_takeover,
            }

            # check if at least one submission format passed normally
            if self.env.plan.submission_format.is_forced and self.env.plan.submission_format_alt.is_forced:
                raise SubmissionFormatError(f"All attempts to create submission formats have failed.")
            elif (self.env.plan.submission_format.is_forced
                    and os.path.exists(os.path.join(self.workspace_path, FileMap.SUBMISSION_FORMAT_ALT_SCRIPT.value))):
                os.rename(
                    os.path.join(self.workspace_path, FileMap.SUBMISSION_FORMAT_ALT_SCRIPT.value),
                    os.path.join(self.workspace_path, FileMap.SUBMISSION_FORMAT_SCRIPT.value)
                )
            elif (self.env.plan.submission_format_alt.is_forced
                    and os.path.exists(os.path.join(self.workspace_path, FileMap.SUBMISSION_FORMAT_ALT_SCRIPT.value))):
                os.remove(os.path.join(self.workspace_path, FileMap.SUBMISSION_FORMAT_ALT_SCRIPT.value))

            if DB_FAISS.started:
                assert self.db_embedding_field == DB_FAISS.embedded_field, \
                    "task.db_embedding_field doesn't match started DB_FAISS"
                if self.db_embedding_field == 'code_error' and reward < len(env_info):
                    # if rag_field == 'code' it is only supposed to relate to errors running the code
                    # so the RAG_KEY is already set in memory in the CreateCodeCommand
                    obs[MemKey.RAG_KEY] = unit_test_info["error"]

                # deactivate rag locally if unit test(s) failed or reactivate if unit test passed
                if self.turn_off_rag_locally_on_failure:
                    if reward == len(env_info):
                        obs[MemKey.RAG_LOCALLY_ACTIVE] = True
                        self.n_unit_test_failed_with_rag_active = 0
                    else:
                        if self.n_unit_test_failed_with_rag_active >= 3:
                            obs[MemKey.RAG_LOCALLY_ACTIVE] = False
                        self.n_unit_test_failed_with_rag_active += 1

            # if stage-group unit test failed, save its error in a different memory key so the error message is not
            # flushed as we attempt again its associated stages and pass their unit tests
            if (last_seen_stage is not None
                    and isinstance(last_seen_stage.name, DataPrepTestStageName)
                    and last_seen_stage.retro_stages is not None
                    and len(last_seen_stage.retro_stages) > 0):
                obs[MemKey.GROUP_UNIT_TEST_OUTPUT] = unit_test_info["output"]
                obs[MemKey.GROUP_UNIT_TEST_ERROR] = unit_test_info["error"]
                obs[MemKey.GROUP_UNIT_TEST_RAN] = unit_test_info["ran"]
                obs[MemKey.GROUP_UNIT_TEST_PASSED] = unit_test_info["passed"]
                obs[MemKey.GROUP_UNIT_TEST_NAME] = last_seen_stage.name.to_str()

            if final_test_info:
                obs[MemKey.FINAL_TEST_PASSED] = final_test_info["passed"]
                if e := final_test_info.get("error"):
                    obs[MemKey.FINAL_TEST_ERROR] = e

            obs = self.free_mem_keys(obs, last_seen_stage)

            return obs, reward, done

        if self.step_num == self.max_rounds:
            self.done = True

        return self.obs, 0., self.done

    def check_trigger_human_takeover(self, action: DataPrepAction) -> bool:
        if action.stage_name in self.action_counter:
            self.action_counter[action.stage_name] += 1
        else:
            self.action_counter[action.stage_name] = 1
        if self.action_counter[action.stage_name] >= self.human_takeover_step:
            return True
        return False

    @staticmethod
    def free_mem_keys(obs: dict[MemKey, Any], last_seen_stage: DataPrepStage | None) -> dict[MemKey, Any]:
        """
        Free up some memory keys that should be erased for the next iteration, in particular:
           - stuff related to CODE are added to memory when running the commands in the flow, but since
                   we reuse the same memory keys, when we start a new stage, we need to free them up
           - stuff related to UNIT_TEST are not ran in commands but rather in the Env but similarly, they
               need to be freed up when a stage is successful so the next stage starting will have fresh memory

        Note that the code and unit tests and their respective summaries are permanently stored in the Stages anyway
        """
        if obs[MemKey.UNIT_TEST_PASSED]:
            obs[MemKey.UNIT_TEST_OUTPUT] = None
            obs[MemKey.UNIT_TEST_ERROR] = None
            obs[MemKey.UNIT_TEST_RAN] = False
            obs[MemKey.UNIT_TEST_PASSED] = False
            obs[MemKey.ERROR_INSTRUCT] = None

            obs[MemKey.RAG_KEY] = None
            obs[MemKey.CODE_RAG_EXAMPLES] = None
            obs[MemKey.UNIT_TEST_RAG_EXAMPLES] = None

            # if unit test passed and last seen stage is a group unit test, it should be the last group unit test
            # for which we recorded the details in the group unit test memory keys - so this group unit test has
            # passed, and we can free its associated memory keys
            if (last_seen_stage is not None
                    and isinstance(last_seen_stage.name, DataPrepTestStageName)
                    and last_seen_stage.retro_stages is not None
                    and len(last_seen_stage.retro_stages) > 0):
                obs[MemKey.GROUP_UNIT_TEST_OUTPUT] = None
                obs[MemKey.GROUP_UNIT_TEST_ERROR] = None
                obs[MemKey.GROUP_UNIT_TEST_RAN] = False
                obs[MemKey.GROUP_UNIT_TEST_PASSED] = False
                obs[MemKey.GROUP_UNIT_TEST_NAME] = None

        return obs

    def save_plan_to_workspace(self) -> None:
        with open(os.path.join(self.workspace_path, FileMap.SETUP_PLAN_JSON.value), 'w') as f:
            f.write(self.env.plan.to_json(indent=4))

    def answer_parser(self, raw_response: str) -> str:
        return raw_response

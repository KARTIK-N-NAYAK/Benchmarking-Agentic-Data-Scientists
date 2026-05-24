import inspect
import os
import pathlib
import re
import shutil
from difflib import SequenceMatcher

from packaging.version import Version

from agent.memory import MemKey
from agent.prompts.builder import BASE_TEMPLATE_PATH
from agent.tasks import ActionSpace
from agent.tasks.datascience_task.utils import replace_in_file
from agent.tasks.tasks import Task
from agent.tools.data_map.map_dataset import MapDataset
from agent.utils import pylogger
from agent.utils.utils import ListableEnum
from ds_agent.utils import save_w_pickle
from third_party.data_science.env import DataScienceEnv, DSAction, DSObsKey
from third_party.data_science.post_processing.summary_generation import generate_cot_summary

log = pylogger.get_pylogger(__name__)

# Mapping env key to memory key
DSObsKEY_TO_MEMKEY = {
    DSObsKey.TASK_DESCRIPTION: MemKey.TASK_DESCRIPTION,
    DSObsKey.DATA_DESCRIPTION: MemKey.DATA_DESCRIPTION,
    DSObsKey.SUMMARIZED_METRIC_DESCRIPTION: MemKey.SUMMARIZED_METRIC_DESCRIPTION,
    DSObsKey.SUBMISSION_LIST: MemKey.SUBMISSION_LIST,
    DSObsKey.CURRENT_SUBMISSION: MemKey.CURRENT_SUBMISSION,
    DSObsKey.CURRENT_SUBMISSION_PERF: MemKey.CURRENT_SUBMISSION_PERF,
    DSObsKey.SENT_SUBMISSION_NAMES: MemKey.SENT_SUBMISSION_NAMES,
    DSObsKey.SUBMISSION_SENT_SUCCESSFULLY: MemKey.SUBMISSION_SENT_SUCCESSFULLY,
}


class DataScienceTaskVersions(ListableEnum):  # Also update the CHANGELOG.md if you add a new version
    V1_1_1 = Version("v1.1.1")
    V1_1_2 = Version("v1.1.2")
    V1_1_3 = Version("v1.1.3")
    V1_1_4 = Version("v1.1.4")
    V1_1_5 = Version("v1.1.5")
    V1_1_6 = Version("v1.1.6")
    V1_1_7 = Version("v1.1.7")
    V1_1_8 = Version("v1.1.8")

    @classmethod
    def list(cls) -> list[Version]:
        versions = super().list()
        versions.sort()
        return versions


MAIN_PIPELINE_VERSIONS = {
    DataScienceTaskVersions.V1_1_1: "Fixed HEBO",
    DataScienceTaskVersions.V1_1_2: "Improved Model Blending",
    DataScienceTaskVersions.V1_1_3: "Fix img_embed unfreeze recommendation",
    DataScienceTaskVersions.V1_1_4: "Add image size statistics and refactor tab_target_transform",
    DataScienceTaskVersions.V1_1_5: "Refactor notebook submission generation and Blend at prediction level",
    DataScienceTaskVersions.V1_1_6: "Make batch size finder use unfrozen image embedder",
    DataScienceTaskVersions.V1_1_7: "Fix typo in solve.py, `get_img_embedder()` returns a tuple",
    DataScienceTaskVersions.V1_1_8: "Fix typo in solve.py to avoid negative remaining times, ramp setup for tasks where target column name is different from the one in sample_submission.csv"
}


class DataScience(Task):
    def __init__(
            self,
            task_id: str,
            prepared_setup_dir: str,
            prepared_version: str,
            exp_id: str,
            workspace_path: str,
            name: str,
            version: str,
            subtask: str | None = None,
            terminate_after_training: bool = False,
            max_exec_time: int | None = None,
            resume_run: bool = False,
            summarize_cot: bool = False,
            is_local_task: bool = False,
            **kwargs,
    ):
        """
        Args:
            task_id: ID of the task kit to be completed.
            prepared_setup_dir: Absolute path to the prepared setup directory.
            prepared_version: ID of the data preparation. Subfolder name of prepared_setup_dir/task_id/
            exp_id: automatically created from task_id, prepared_version and timestamp
            workspace_path: Path to the new workspace to be created.
        """
        if subtask is None:
            subtask = task_id
        self.max_rounds = 20
        self.task_id = task_id
        self.prepared_setup_dir = prepared_setup_dir
        self.prepared_version = prepared_version
        self.exp_id = exp_id
        self.workspace_path = workspace_path
        self.action_space = ActionSpace.DISCRETE
        self.done = False
        self.step_num = 0
        self.resume_run = resume_run
        self.summarize_cot = summarize_cot
        self.is_local_task = is_local_task

        self.env = DataScienceEnv(
            task_id=self.task_id,
            prepared_setup_dir=self.prepared_setup_dir,
            prepared_version=self.prepared_version,
            exp_id=self.exp_id,
            workspace_path=self.workspace_path,
            terminate_after_training=terminate_after_training,
            max_exec_time=max_exec_time,
            is_local_task=is_local_task
        )
        description = self.env.get_task_description()
        super().__init__(name=name, subtask=subtask, description=description, version=version, **kwargs)

    def get_dict_cfg_info(self) -> dict[str, ...]:
        """Output a dictionary containing some info about the run."""
        return {
            "max_round": self.max_rounds,
            "prepared_version": self.prepared_version,
            "ds_task_version": str(DataScienceTaskVersions.list()[-1]),
        }

    def reset(self, next_subtask: str | None = None) -> dict[str, str]:
        """Reset the environment and return the initial observation."""

        self.template_code_reset()
        if self.resume_run:
            self.resume_run = False
            return self.resume()

        # save config in workspace
        save_w_pickle(obj=self.get_dict_cfg_info(), path=self.workspace_path, filename="info")

        self.done = False
        self.step_num = 0

        env_obs, env_info = self.env.reset()
        task_obs = {DSObsKEY_TO_MEMKEY[ds_key]: val for ds_key, val in env_obs.items()}
        task_obs[MemKey.TEMPLATES_RELATIVE_PATH] = self.templates_relative_path
        task_obs[MemKey.AVAILABLE_ACTIONS] = env_info[DSObsKey.AVAILABLE_ACTIONS]
        task_obs[MemKey.SUMMARIZE_COT] = self.summarize_cot
        return task_obs

    def resume(self) -> dict[str, str]:
        """ Resume """

        print(f"Resuming from previous  run {self.workspace_path}")
        log.info(f"Resuming from previous  run {self.workspace_path}")

        env_obs, reward, done, env_info = self.env.resume()

        task_obs: dict[MemKey, str | list[str]] = {
            DSObsKEY_TO_MEMKEY[ds_key]: val for ds_key, val in env_obs.items()
        }
        if not done:
            task_obs[MemKey.AVAILABLE_ACTIONS] = env_info[DSObsKey.AVAILABLE_ACTIONS]

        task_obs[MemKey.TEMPLATES_RELATIVE_PATH] = self.templates_relative_path
        task_obs[MemKey.SUMMARIZE_COT] = self.summarize_cot

        self.step_num = self.env.step_num
        self.done = done
        return task_obs

    def template_code_reset(self) -> None:
        """Create or moved the templates that are task-dependant."""
        os.makedirs(self.templates_path, exist_ok=True)
        os.makedirs(self.workspace_path, exist_ok=True)

        current_dir = str(pathlib.Path(__file__).parent.resolve())

        replacements = {"@ROOT_DS_DATA_PATH@": self.env.get_src_path(),
                        "@WORKSPACE@": self.workspace_path}

        code_templates = [
            "./datascience_task/code_blanks/classical_tab_fe_code_template.py",
            "./datascience_task/code_blanks/tab_fe_code_template.py",
            "./datascience_task/code_blanks/tab_preprocessed_embed_code_template.py",
            "./datascience_task/code_blanks/tab_regression_target_scaler_code_template.py",
            "./datascience_task/code_blanks/tab_head_code_template.py",
            "./datascience_task/code_blanks/img_embed_code_template.py",
            "./datascience_task/code_blanks/train_img_transform_code_template.py",
            "./datascience_task/code_blanks/test_img_transform_code_template.py",
            "./datascience_task/code_blanks/txt_embed_code_template.py",
            "./datascience_task/code_blanks/bag_code_template.py",
            "./datascience_task/code_blanks/class_imbalance_code_template.py",
            "./datascience_task/code_blanks/class_imbalance.py"
        ]

        # Need to move some code with blanks as we modify them depending on the task
        codes = [
            os.path.abspath(inspect.getfile(MapDataset)),
            "./datascience_task/code_blanks/solve.py",
            "./datascience_task/code_blanks/train_utils.py",
            "./datascience_task/code_blanks/solve_common_utils.py",
            "./datascience_task/code_blanks/solve_params.py",
            "./datascience_task/code_blanks/blend_params.py",
            "./datascience_task/code_blanks/tab_fe.py",
            "./datascience_task/code_blanks/tab_embed.py",
            "./datascience_task/code_blanks/tab_regression_target_scaler.py",
            "./datascience_task/code_blanks/tab_head.py",
            "./datascience_task/code_blanks/img_embed.py",
            "./datascience_task/code_blanks/img_transform.py",
            "./datascience_task/code_blanks/txt_embed.py",
            "./datascience_task/code_blanks/tab_fe_code_template.py",
            "./datascience_task/code_blanks/classical_tab_fe_code_template.py",
            "./datascience_task/code_blanks/tab_preprocessed_embed_code_template.py",
            "./datascience_task/code_blanks/tab_regression_target_scaler_code_template.py",
            "./datascience_task/code_blanks/tab_head_code_template.py",
            "./datascience_task/code_blanks/img_embed_code_template.py",
            "./datascience_task/code_blanks/train_img_transform_code_template.py",
            "./datascience_task/code_blanks/test_img_transform_code_template.py",
            "./datascience_task/code_blanks/txt_embed_code_template.py",
            "./datascience_task/code_blanks/bag_code_template.py",
            "./datascience_task/code_blanks/bag_submissions.py",
            "./datascience_task/code_blanks/bag_code.py",
            "./datascience_task/code_blanks/blend.py",
            "./datascience_task/code_blanks/create_blend_dataset.py",
            "./datascience_task/code_blanks/class_imbalance_code_template.py",
            "./datascience_task/code_blanks/class_imbalance.py"
        ]

        def get_template_dst(path: str) -> str:
            """Get destination path of the template."""
            p, ext = os.path.splitext(path)
            if ext != ".jinja":
                path = p + ".jinja"
            return os.path.join(self.templates_path, os.path.basename(path))

        def get_workspace_dst(path: str) -> str:
            """Get destination path of the code blank."""
            p, ext = os.path.splitext(path)
            if ext != ".py":
                path = p + ".py"
            return os.path.join(self.workspace_path, os.path.basename(path))

        for code_template in code_templates:
            code_template = current_dir + "/" + code_template
            dst_path = get_template_dst(path=code_template)
            shutil.copyfile(src=code_template, dst=dst_path)
            replace_in_file(file_path=dst_path, replacements=replacements)

        for code in codes:
            if code[0] != "/":
                code = current_dir + "/" + code
            dst_path = get_workspace_dst(path=code)
            shutil.copyfile(src=code, dst=dst_path)
            replace_in_file(file_path=dst_path, replacements=replacements)

    @property
    def templates_root_path(self) -> str:
        return f"{BASE_TEMPLATE_PATH}/data_science/"

    @property
    def templates_relative_path(self) -> str:
        return f"experiments/{self.exp_id}/"

    @property
    def templates_path(self) -> str:
        return os.path.join(self.templates_root_path, self.templates_relative_path)

    def answer_parser(self, raw_response: str) -> str:
        return raw_response

    def is_complete(self) -> bool:
        return self.done

    def step(self, action: DSAction, retrial_chat_completion_time: float) -> tuple[dict[MemKey, ...], float, bool]:
        if os.getenv("AGENT_DEBUG", False):
            print(
                f"Step {self.step_num} | "
                f"workspace: {os.path.abspath(self.workspace_path)} | "
                f"Dataspace: {os.path.abspath(self.env.get_src_path())}"
            )

        """Perform an action and return the next observation, reward, and done."""
        if action.stage_name not in self.env.get_available_actions():
            raise RuntimeError(f"{action.stage_name} not in {self.env.get_available_actions()}")
        # processed_action = process_action(action=action, choices=self.env.get_available_actions())
        env_obs, reward, done, env_info = self.env.step(
            action=action, retrial_chat_completion_time=retrial_chat_completion_time
        )

        task_obs: dict[MemKey, str | list[str]] = {
            DSObsKEY_TO_MEMKEY[ds_key]: val for ds_key, val in env_obs.items()
        }
        if not done:
            task_obs[MemKey.AVAILABLE_ACTIONS] = self.env.get_available_actions()

        self.step_num += 1
        self.done = done

        if self.done:
            cot_content = generate_cot_summary(competition_workspace=self.workspace_path)
            cot_file_path = os.path.join(self.prepared_setup_dir, self.task_id, self.prepared_version, "summary.txt")
            with open(cot_file_path, "w") as f:
                f.write(cot_content)

        return task_obs, reward, self.done


def process_action(action: str, choices: list[str], logging: bool = False) -> str:
    """Format the action with respect to the choices."""
    if logging:
        log.info("Raw action: %s", action)
    match = re.search("Action:(.*)\n", action)
    if match:
        action = match.group(1)
    else:
        match = re.search("Action:(.*)", action)
        if match:
            action = match.group(1)
    action = action.strip().split("\n")[0]
    if not choices:
        return action
    if action in choices:
        return action
    try:
        max_similarity = 0
        final_ind = 0
        for ind, choice in choices:
            similarity = SequenceMatcher(None, choice, action).ratio()
            if similarity > max_similarity:
                final_ind = ind
        return choices[final_ind]
    except Exception as e:
        log.exception(e)
        log.debug("choices: %s", choices)
        log.debug("action: %s", action)
    return action

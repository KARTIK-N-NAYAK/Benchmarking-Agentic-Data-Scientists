from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ramphy.actions import RAMP_ACTIONS
from ramphy.actions import RampAction


class RampObsKey(Enum):
    AVAILABLE_ACTIONS = "available_actions"
    TASK_DESCRIPTION = "task_description"
    DATA_DESCRIPTION = "data_description"
    SUMMARIZED_METRIC_DESCRIPTION = "metric_description"
    SUBMISSION_LIST = "submission_list"
    CURRENT_SUBMISSION = "current_submission"
    SENT_SUBMISSION_NAMES = "sent_submission_names"
    ACTION_NAME = "action_dict"
    ACTION_PARAMS = "action_params_dict"
    ACTION_EXECUTION_OUTPUT = "action_execution_output"


class RampEnv:
    def __init__(self, ramp_kit_dir: str, ramp_data_dir: Optional[str] = None) -> None:
        self.ramp_kit_dir = ramp_kit_dir
        if ramp_data_dir is None:
            self.ramp_data_dir = ramp_kit_dir
        else:
            self.ramp_data_dir = ramp_data_dir

        self.obs: Dict[RampObsKey, Any] = {}

    def reset(self) -> Tuple[Dict[RampObsKey, Any], Dict[RampObsKey, Any]]:
        """Resets the environment

        Returns:
            Tuple[Dict[RampObsKey, Any], Dict[RampObsKey, Any]]: observation and info dict
        """
        self.obs = {}

        info: Dict[RampObsKey, Any] = {
            RampObsKey.AVAILABLE_ACTIONS: self.get_available_actions(),
            RampObsKey.ACTION_EXECUTION_OUTPUT: [None],
            RampObsKey.ACTION_PARAMS: [None],
            RampObsKey.ACTION_NAME: [None],
        }
        return self.obs, info

    def step(
        self, action: RampAction | List[RampAction]
    ) -> Tuple[Dict[RampObsKey, Any], float, bool, Dict[RampObsKey, Any]]:
        """Performs the step

        Args:
            action (RampAction | List[RampAction]): Action or list of actions to execute

        Returns:
            Tuple[Dict[RampObsKey, Any], float, bool, Dict[RampObsKey, Any]]: obs, reward, done, info
        """
        if not isinstance(action, List):
            action = [action]
        else:
            assert len(action) > 0, "You should pass at least an action"

        info: Dict[RampObsKey, Any] = {}
        info[RampObsKey.AVAILABLE_ACTIONS] = self.get_available_actions()
        info[RampObsKey.ACTION_NAME] = {}

        for atomic_action in action:
            action_execution_output = atomic_action.execute()
            info[RampObsKey.ACTION_NAME][f"{atomic_action.module}.{atomic_action.name}"] = {
                RampObsKey.ACTION_PARAMS: {"args": atomic_action.args, "kwargs": atomic_action.kwargs},
                RampObsKey.ACTION_EXECUTION_OUTPUT: action_execution_output,
            }

        reward = self.get_reward(action=action[-1])
        done = self.get_done(action=action[-1])
        return self.obs, reward, done, info

    def get_available_actions(self) -> List[str]:
        # TODO think if we want to have all the actions available all the time or not
        return list(RAMP_ACTIONS.keys())

    def get_reward(self, action: RampAction) -> float:
        # TODO implement
        return 0

    def get_done(self, action: RampAction) -> bool:
        # TODO implement
        return True  # For now we only use fixed plans, so it returns true

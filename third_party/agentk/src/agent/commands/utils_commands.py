from typing import Any, Callable

from agent.agents import LLMAgent
from agent.commands import LoopFlow
from agent.commands.core import HumanTakeoverCommand, Command
from agent.memory import MemKey


class CodePassedLoopChoiceCmd(HumanTakeoverCommand):
    name: str = "code_pass_loop_choice"
    description: str = "Add choice to terminate or continue depending on the code pass status."

    input_keys: dict[str, MemKey] = {
        MemKey.CODE_PASSED.value: MemKey.CODE_PASSED
    }
    output_keys: dict[str, MemKey] = {MemKey.LOOP_CODE_PASSED_CHOICE.value: MemKey.LOOP_CODE_PASSED_CHOICE}

    check_code_passed: Callable[[LLMAgent], None] | None = None  # func. taking agent as input and modifying CODE_PASSED

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        """ Put 'Continue' or 'Terminate' in choices depending on the status of the code """
        if self.check_code_passed is not None:
            self.check_code_passed(agent=agent)
        code_passed = agent.memory.retrieve(self.input_keys[MemKey.CODE_PASSED.value])
        assert code_passed is not None
        if code_passed:
            choice = LoopFlow.TERMINATE_LOOP_CHOICE
        else:
            choice = LoopFlow.CONTINUE_LOOP_CHOICE
        agent.memory.store(content=[choice], tags=self.output_keys[MemKey.LOOP_CODE_PASSED_CHOICE.value])

    @property
    def choice_key(self) -> MemKey:
        return list(self.output_keys.values())[0]


class CodePassedPostLoopChoiceCmd(Command):
    name: str = "code_passed_before_max_repetitions"
    description: str = "Check whether the Agent generated the code before the max repetition was reached"

    check_loop_passed: Callable[[LLMAgent], bool] = None  # func. taking agent and returning true if the loop was succ.

    success_end_choice: str = ""
    fail_end_choice: str = ""

    output_keys: dict[str, MemKey] = {MemKey.THINK_AND_CODE_END_CHOICE.value: MemKey.THINK_AND_CODE_END_CHOICE}

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        """ Put 'Continue' or 'Terminate' in choices depending on the status of the code """
        if self.check_loop_passed(agent=agent):
            choice = self.success_end_choice
        else:
            choice = self.fail_end_choice

        agent.memory.store(content=[choice], tags=self.output_keys[MemKey.THINK_AND_CODE_END_CHOICE.value])


class MultiTrialPostLoopChoiceCmd(Command):
    name: str = "inner_loop_success_check"
    description: str = "Check whether the inner loop of a multi-trial think and code passed or not"

    success_message: str = ""
    fail_message: str = ""

    input_keys: dict[str, MemKey] = {mkey.value: mkey for mkey in [MemKey.THINK_AND_CODE_END_CHOICE]}
    output_keys: dict[str, MemKey] = {mkey.value: mkey for mkey in [MemKey.MULTI_TRIAL_THINK_AND_CODE_LOOP_CHOICE]}

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        """ Put 'Continue' or 'Terminate' in choices depending on the status of the code """
        inner_loop_end = agent.memory.retrieve(self.input_keys[MemKey.THINK_AND_CODE_END_CHOICE.value])[0]

        if inner_loop_end == self.success_message:
            choice = LoopFlow.TERMINATE_LOOP_CHOICE
        elif inner_loop_end == self.fail_message:
            choice = LoopFlow.CONTINUE_LOOP_CHOICE
        else:
            raise ValueError(
                f"Inner loop end message: {inner_loop_end}\nSuccess message: {self.success_message}"
                f"\nFailure message: {self.fail_message}"
            )

        agent.memory.store(content=[choice], tags=self.output_keys[MemKey.MULTI_TRIAL_THINK_AND_CODE_LOOP_CHOICE.value])


class AlternateCodeCommand(Command):
    name: str = "alternate_code_command"
    description: str = "check for existence of code , if code failed and another version of code available , take that code"

    input_keys: dict[str, MemKey] = {mkey.value: mkey for mkey in [MemKey.CODE_PASSED, MemKey.CODE_BLANK]}
    code_memkey: MemKey | None = None

    code_response: str | None = None

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        """ store code, if it does not exist. If failed to generate code and another already available, use it """

        is_code_passed = agent.memory.retrieve(self.input_keys[MemKey.CODE_PASSED.value])
        available_code = agent.memory.retrieve(self.code_memkey)
        if is_code_passed and available_code is None:
            # if code available save it to use in particular code generation failed case in the current main flow
            available_code = agent.memory.retrieve(MemKey.CODE_BLANK)
            agent.memory.store(content=available_code, tags=self.code_memkey)
        elif not is_code_passed and available_code:
            # failed to generate successful code, use the saved code to  continue the flow
            self.code_response = f"```python \n{available_code} \n```"

    @property
    def available_code(self) -> str | None:
        return self.code_response

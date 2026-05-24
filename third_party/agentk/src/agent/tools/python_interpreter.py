import os
import re
import shutil
from typing import Any

from agent.memory import MemKey
from agent.tools.base_tool import Tool
from agent.utils.utils import get_path_to_python, catch_error_wrap

BLANK_PATTERN = """# <｜fim▁hole｜>"""


class PythonInterpreter(Tool):
    requires_llm_prompt: bool = True
    name: str = "Python interpreter"

    def __init__(
            self,
            path_to_python,
            code_path="testing.py",
            code_output_path="code_output.txt",
            code_error_path="error_file.txt",
            code_warnings_path="warnings_file.txt",
            code_persistent_file_path: str | None = None
    ) -> None:
        """
        Args:
            code_persistent_file_path: path where the code should be saved permanently instead of removing it
        """
        self.path_to_python = get_path_to_python(path_to_python)
        self.code_path = code_path
        self.code_output_path = code_output_path
        self.code_error_path = code_error_path
        self.code_warnings_path = code_warnings_path
        self.aux_code_error_path = os.path.join(os.path.dirname(self.code_error_path), "aux_error_file.txt")
        self.code_persistent_file_path = code_persistent_file_path

    def wrap(self, agent_input: str) -> str:
        """Can modify agent's output before passing it to the interpreter."""
        return agent_input

    @staticmethod
    def wrap_submit(model_code: str) -> str:
        """Can modify agent's output before passing it to the interpreter."""
        return model_code

    @staticmethod
    def extract_python_code_blocks(raw_input: str) -> list[str]:
        """ Extract the code enclosed in python pattern """
        if "```python" in raw_input and len(re.findall("```", raw_input)):
            raw_input += "\n```"  # handle missing final "```"
        pattern = r"```python(.*?)```"
        return re.findall(pattern, raw_input, flags=re.S)

    def __call__(self, agent_input: str) -> dict[str, str]:
        print("============ Python Interpreter ============")
        to_store = {}
        to_store.update(self.pre_hooks(agent_input=agent_input))
        agent_input = self.wrap(agent_input)
        # parse the agent input to get the python code
        code_blocks = self.extract_python_code_blocks(raw_input=agent_input)
        # check if the LLM output does not follow the required format
        self.code_to_memorize = ""
        if len(code_blocks) == 0:
            self.code = ""
            self.code_output = ""
            self.code_error = ("The agent does not generate code in the correct format. Please retry!\n"
                               "Write full code inside ```python\n<code>\n```.")
            print(self.code_error)
        else:
            self.code_to_memorize = code_blocks[-1]
            self.code_to_memorize = self.filter_code_to_memorize(code=self.code_to_memorize)
            # We wrap the actual code in a try / except
            self.code = self.catch_error_wrap(code_blocks[-1])
            with open(self.code_path, "w") as f:
                f.write(self.code)

            # run the code
            print(os.getcwd())
            cmd = f"{self.path_to_python} {self.code_path} 2> {self.aux_code_error_path} > {self.code_output_path}"
            print(cmd)
            os.system(cmd)

            # Catch errors --> find some errors in aux_code_error_path, then any error in code_error_path
            with open(self.aux_code_error_path) as f:
                aux_code_error = f.readlines()
                if len(aux_code_error) > 0:
                    errors_to_catch = ["SyntaxError:", "TabError:", "IndentationError:", "Killed"]
                    caught_error = False
                    for error in errors_to_catch:
                        if error in aux_code_error[-1]:
                            self.code_error = "".join(aux_code_error[-10:])
                            caught_error = True
                            break
                    if not caught_error:
                        if os.path.exists(self.code_error_path):
                            with open(self.code_error_path) as code_error_f:
                                self.code_error = code_error_f.read()
                            if self.code_persistent_file_path is None:
                                os.remove(self.code_error_path)
                        else:
                            self.code_error = ""
                else:
                    self.code_error = ""

            with open(self.code_output_path) as f:
                self.code_output = f.read()

            try:
                if os.path.exists(self.code_warnings_path):
                    with open(self.code_warnings_path, "r") as f:
                        self.code_warnings = f.read()
                    if self.code_persistent_file_path is None:
                        os.remove(self.code_warnings_path)
            except Exception as e:
                print(f"An error occurred: {e}")

            if self.code_error == "" and self.code_output == "":
                with open(self.aux_code_error_path) as f:
                    full_aux_code_error = f.readlines()
                if len(full_aux_code_error) > 0:
                    raise RuntimeError(f"Python command failed with error message:\n{full_aux_code_error}")
                else:
                    raise RuntimeError(f"If you haven't already, please make sure to include a "
                                       f"print statement at the end of your code. :)")

            to_store.update(self.post_hooks())
            if self.code_persistent_file_path is not None:
                shutil.copy(src=self.code_path, dst=self.code_persistent_file_path)
            os.remove(self.code_path)
            os.remove(self.code_output_path)
            os.remove(self.aux_code_error_path)

            print("The code below is run in a python interpreter: ")
            print(self.code)
            if self.code_error == "":
                print("The code runs successfully! Here is the result: ")
                print(self.code_output)
            else:
                print("The code does not run successfully. Here is the error messages: ")
                print(self.code_error)
        print("============================================")
        to_store.update(
            {
                MemKey.CODE: self.code_to_memorize,
                MemKey.CODE_OUTPUT: self.code_output,
                MemKey.CODE_ERROR: self.code_error,
            }
        )
        return to_store

    def pre_hooks(self, agent_input: str) -> dict[MemKey, Any]:
        return {}

    def post_hooks(self) -> dict[MemKey, Any]:
        return {}

    @staticmethod
    def filter_code_to_memorize(code: str) -> str:
        no_memory_start = "# @NO_MEMORY_START@\n"
        no_memory_end = "# @NO_MEMORY_END@\n"
        splitted_code = code.split(no_memory_start)
        codes_to_memorize = [splitted_code[0]]  # part before first NO_MEMORY
        if len(codes_to_memorize) > 2:
            codes_to_memorize += list(map(lambda x: x.split(no_memory_end)[1], splitted_code[1:]))
        return "\n".join(codes_to_memorize)

    def catch_error_wrap(self, code: str) -> str:
        # wrapped_code = "import os\n"
        # wrapped_code += "import traceback\n"
        # wrapped_code += "try:\n"
        # wrapped_code += "\n".join(map(lambda x: "    " + x, code.split("\n")))
        # wrapped_code += "\nexcept Exception as e:\n"
        # wrapped_code += "    error_message = traceback.format_exc()\n"
        # wrapped_code += f"    with open('{self.code_error_path}', 'w') as f:\n"
        # wrapped_code += "        f.write(error_message)\n"
        # wrapped_code += "        raise\n"
        return catch_error_wrap(
            code=code, code_error_path=self.code_error_path, code_warnings_path=self.code_warnings_path
        )


class PythonInterpreterWithBlanks(PythonInterpreter):
    name = "Python interpreter where Agent only fills blanks"

    def __init__(
            self,
            path_to_python: str,
            code_with_blanks: str,
            workspace_path: str,
            code_persistent_file_path: str | None = None
    ):
        """
        Args:
        """

        self.workspace_path = workspace_path
        self.code_with_blanks = code_with_blanks
        super().__init__(
            path_to_python=path_to_python,
            code_path=os.path.join(self.workspace_path, "testing.py"),
            code_output_path=os.path.join(self.workspace_path, "code_output.txt"),
            code_error_path=os.path.join(self.workspace_path, "code_error.txt"),
            code_persistent_file_path=code_persistent_file_path
        )

    def pre_hooks(self, agent_input: str) -> dict[MemKey, Any]:
        """ Save the blanks """
        code_blocks = self.extract_python_code_blocks(raw_input=agent_input)
        if len(code_blocks) == 0:
            return {}
        return {MemKey.CODE_BLANK: code_blocks[-1]}

    def wrap(self, agent_input: str) -> str:
        code_blocks = self.extract_python_code_blocks(raw_input=agent_input)
        # check if the LLM output does not follow the required format
        if len(code_blocks) == 0:
            return agent_input
        code = code_blocks[-1]
        with open(self.code_with_blanks) as f:
            lines = f.readlines()
        new_str = "".join(lines)
        new_str = new_str.replace(BLANK_PATTERN, code)
        return f"```python{new_str}```"

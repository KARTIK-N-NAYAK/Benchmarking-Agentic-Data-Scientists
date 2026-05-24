from pathlib import Path

from agent.commands.core import Command
from agent.loggers import ManyLoggers, FileSystemLogger
from agent.memory import MemKey

from ds_agent.data_preprocessing.log_utils import load_log_as_df
from ds_agent.rag.unit_test_correction import extract_unit_test_code_corrections
from ds_agent.rag.db_faiss import DB_FAISS


class StartRAGDB(Command):
    """This is a command that initialises a particular RAG DB."""

    name: str = "start_rag_db"
    description: str = "Starts the RAG DB."

    db_path: Path
    db_embedded_field: str | None
    db_embedding_model_name: str = "thenlper/gte-small"

    disabled: bool = False
    
    def func(self, agent):
        if self.disabled:
            return
        
        DB_FAISS.start(
            db_path=self.db_path, model_name=self.db_embedding_model_name, embedded_field=self.db_embedded_field
        )


class UpdateRAGDB(Command):
    """This is a command designed to be run on_episode_end to add the logs into the RAG DB."""

    # COMMAND DEFINITION
    name: str = "update_rag_db"
    description: str = ("Store relevant examples from the episode log into the RAG database. "
                        "Requires RAG DB to already be started.")

    input_keys: dict[str, MemKey] = dict(final_test_passed_mem_key=MemKey.FINAL_TEST_PASSED)

    disabled: bool = False

    require_final_test_pass: bool = True
    save_to_disk: bool = True

    def func(self, agent, final_test_passed_mem_key):
        if self.disabled:
            return

        final_test_passed = agent.memory.retrieve(final_test_passed_mem_key)
        if final_test_passed or not self.require_final_test_pass:
            if isinstance(agent.logger, ManyLoggers):
                # IndexError: see below
                fs_logger = agent.logger.fs_loggers[0]
            elif isinstance(agent.logger, FileSystemLogger):
                fs_logger = agent.logger
            else:
                raise ValueError("One of the loggers must be a FileSystemLogger to update DB")

            log_df = load_log_as_df(fs_logger.log_path)
            code_examples = extract_unit_test_code_corrections(log_df)

            if code_examples:
                DB_FAISS.insert_code_examples(code_examples, create_missing=True)
                if self.save_to_disk:
                    DB_FAISS.save_databases()

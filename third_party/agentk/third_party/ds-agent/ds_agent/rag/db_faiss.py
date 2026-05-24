import glob
import heapq
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class DBExample(ABC):
    @abstractmethod
    def db_id(self, embedding_field: Optional = None):
        raise NotImplementedError


@dataclass
class CodeErrorExample(DBExample):
    # These names should match MemKeys that should retrieve them
    template_file: str
    code: str
    code_error: str
    code_fixed: str
    task_id: str

    def db_id(self, embedding_field=None) -> str:
        field_prefix = f"{embedding_field!s}/" if embedding_field is not None else ""
        return field_prefix + self.template_file.replace(".jinja", "")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# DB metadata
class VectorDBInfo(BaseModel):
    embedded_field: str
    model_name: str

    model_config = ConfigDict(extra="allow", protected_namespaces=tuple())


class VectorDBMetadata(BaseModel):
    databases: dict[Path, VectorDBInfo] = Field(default_factory=dict)

    # def join_paths_with_root(self, db_root_path: Path):
    #     joined_db_paths = {}
    #     for path, info in self.databases.items():
    #         joined_db_paths[db_root_path / path] = info
    #     self.databases = joined_db_paths

    model_config = ConfigDict(extra="allow")


class VectorFaissDB(metaclass=Singleton):
    """
    A singleton class for managing FAISS databases.

    Args:
        db_path (str): Root path for the database.
        embedding_function: Function to compute embeddings.
        modality_input (str): Input modality (e.g., "text", "image").
        modality_target (str): Target modality (e.g., "text", "image").

    Attributes:
        root_db_path (Path): Root path for the database.
        embedding_function: Function to compute embeddings.
        input_db (FAISS): FAISS database for input modality.
        target_db (FAISS): FAISS database for target modality.

    Methods:
        retrieve_input_top_k_documents(query: str, k: int = 4) -> list[Document]:
            Retrieves top-k documents from the input database based on similarity to the query.
        retrieve_target_top_k_documents(query: str, k: int = 4) -> list[Document]:
            Retrieves top-k documents from the target database based on similarity to the query.
        create_faiss_db(file_db_path: Path, embedding_function) -> FAISS:
            Creates a FAISS database from documents in the specified directory.

    Note:
        - Requires FAISS library.
        - Assumes that the databases exist at the specified paths.
    """

    CHECK_MODEL_NAME = True
    METADATA_PATH = "metadata.json"

    embedded_field = None

    @property
    def started(self):
        """Reuse the embedded_field variable to indicate starting"""
        return self.embedded_field is not None

    def start(self, db_path: str | Path, model_name: str, embedded_field: str, is_empty: bool = False):
        assert embedded_field is not None
        if self.started:
            params = (Path(db_path), model_name, embedded_field, is_empty)
            started_params = (self._root_db_path, self._model_name, self.embedded_field, False)
            if params == started_params:
                # Allow and return early if initialisation parameters match
                return
            else:
                raise ValueError("VectorFaissDB should only be .start()-ed once!")

        self._model_name = model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self._model_name)

        self._root_db_path = Path(db_path)
        if not self._root_db_path.exists() and not is_empty:
            raise FileNotFoundError(f"db_path doesn't exist: {db_path}")
        elif self._root_db_path.is_file():
            raise FileExistsError(f"Provided path ({db_path}) should be a directory not a file")

        if is_empty:
            if not self._root_db_path.exists() or not list(self._root_db_path.iterdir()):
                self.metadata = VectorDBMetadata()
            else:
                raise FileExistsError(f"is_empty=True but db_path is not an empty directory: {db_path}")
            self.db_info = self.metadata.databases
            self.databases = {}
            # This will create the directory and empty metadata file
            self.embedded_field = embedded_field
            self.save_databases()
        else:
            self.metadata = self._load_metadata(self._root_db_path)
            self.db_info = self.metadata.databases
            self.databases = self._load_databases(self._root_db_path, self.db_info)
            self.embedded_field = embedded_field

    def _load_metadata(self, db_path: Path) -> VectorDBMetadata:
        metadata_path = db_path / self.METADATA_PATH
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{self.METADATA_PATH} file '{metadata_path}' not found, are you using the right db_path?"
            )
        with metadata_path.open("r") as fp:
            metadata = VectorDBMetadata.model_validate_json(fp.read())

        # Check that there are no undocumented .faiss files in the db_path
        faiss_file_paths = {
            Path(p_str).parent.absolute() for p_str in glob.iglob(f"{db_path!s}/**/*.faiss", recursive=True)
        }
        undocumented_dbs = faiss_file_paths - {(db_path / p).absolute() for p in metadata.databases.keys()}
        if undocumented_dbs:
            raise ValueError(f"{metadata_path} is missing some local dbs: {undocumented_dbs}")

        return metadata

    def _load_databases(self, db_root: Path, db_info: dict[Path, VectorDBInfo]) -> dict[Path, FAISS]:
        return {
            p: FAISS.load_local(db_root / p, self.embedding_function, allow_dangerous_deserialization=True)
            for p in db_info.keys()
        }

    def save_databases(self, output_dir: Path = None):
        assert self.started, "VectorFaissDB should be started before saving"

        if output_dir is None:
            output_dir = self._root_db_path

        output_dir.mkdir(exist_ok=True)

        metadata_str = self.metadata.model_dump_json(indent=2)
        with (output_dir / self.METADATA_PATH).open("w") as fp:
            fp.write(metadata_str)

        for path, db in self.databases.items():
            db.save_local(output_dir / path)

    def _make_subset_filter(self, filter_items: dict):
        # Helper function for making a filter a dict subset
        if self.CHECK_MODEL_NAME:
            assert (
                "model_name" in VectorDBInfo.model_fields
            ), "If you have changed the variable 'model_name' in VectorDBInfo you should change it here"
            filter_items["model_name"] = self._model_name

        def subset_filter(info):
            return filter_items.items() <= info.items()

        return subset_filter

    def _retrieve_top_k_documents(
        self, query: str, k: int, db_filter: Callable[[dict], bool], row_filter=None
    ) -> list[Document]:
        assert self.started, "DB should be started before attempting to retrieve documents"
        valid_dbs = {path for path, info in self.db_info.items() if db_filter(info.model_dump())}

        docs_and_scores = []
        for db_path in valid_dbs:
            db = self.databases[db_path]
            docs_and_scores.extend(db.similarity_search_with_score(query, k, row_filter))

        return [doc for doc, _score in heapq.nlargest(k, docs_and_scores, key=lambda x: x[1])]

    def retrieve_top_template_error_example(
        self, query: str, template_file: str, heldout_metadata: Optional[dict] = None
    ) -> Optional[CodeErrorExample]:
        criteria = {"template_file": template_file, "embedded_field": self.embedded_field}

        metadata_filter = None
        if heldout_metadata is not None:

            def metadata_filter(metadata_dict):
                return all(metadata_dict.get(k) != v for k, v in heldout_metadata.items())

        results = self._retrieve_top_k_documents(
            query, k=1, db_filter=self._make_subset_filter(criteria), row_filter=metadata_filter
        )
        return CodeErrorExample(**results[0].metadata) if results else None

    def retrieve_input_top_k_documents(self, query: str, k: int = 4, modality=None) -> list[Document]:
        criteria = {"template_path": "data_preprocessing/data_map/code", "stage": "input"}
        if modality is not None:
            criteria["modality"] = modality
        return self._retrieve_top_k_documents(query, k=k, db_filter=self._make_subset_filter(criteria))

    def retrieve_target_top_k_documents(self, query: str, k: int = 4, modality=None) -> list[Document]:
        criteria = {"template_path": "data_preprocessing/data_map/code", "stage": "target"}
        if modality is not None:
            criteria["modality"] = modality
        return self._retrieve_top_k_documents(query, k=k, db_filter=self._make_subset_filter(criteria))

    def retrieve_metric_top_k_documents(self, query: str, k: int = 4) -> list[Document]:
        criteria = {"template_path": "data_preprocessing/metric"}
        return self._retrieve_top_k_documents(query, k=k, db_filter=self._make_subset_filter(criteria))

    def retrieve_submission_format_top_k_documents(self, query: str, k: int = 4) -> list[Document]:
        criteria = {"template_path": "data_preprocessing/submission_format"}
        return self._retrieve_top_k_documents(query, k=k, db_filter=self._make_subset_filter(criteria))

    @staticmethod
    def format_docs_to_str(docs: list) -> str:
        docs_str = ""
        # Extract code between the specified markers
        pattern = r"# <｜fim▁begin｜>(.*?)# <｜fim▁end｜>"
        for doc in docs:
            code_snippet = doc.page_content
            code_snippet_match = re.search(pattern, code_snippet, re.DOTALL)

            if code_snippet_match:
                code_snippet = code_snippet_match.group(1)

            docs_str += code_snippet + "\n"

        return docs_str

    def _create_database(self, examples: list[DBExample], extra_db_info: dict | None = None):
        assert examples, "Database must be initialised with at least 1 example"

        db_key = examples[0].db_id(embedding_field=self.embedded_field)
        db_key = Path(db_key)
        if db_key in self.databases or db_key in self.db_info:
            raise ValueError(f"A database with key '{db_key}' already exists")

        extra_db_info = {} if extra_db_info is None else extra_db_info
        db_info = VectorDBInfo(embedded_field=self.embedded_field, model_name=self._model_name, **extra_db_info)

        texts = [getattr(e, self.embedded_field) for e in examples]
        metadatas = [asdict(e) for e in examples]
        db = FAISS.from_texts(texts, self.embedding_function, metadatas)

        self.databases[db_key] = db
        self.db_info[db_key] = db_info

        return db

    def insert_code_examples(self, examples: list[CodeErrorExample], create_missing: bool = False):
        assert self.started, "Tried to insert examples in a database that hasn't been started!"

        examples_by_db_id = defaultdict(list)
        for example in examples:
            examples_by_db_id[example.template_file].append(example)

        for template_file, template_examples in examples_by_db_id.items():
            template_filter = self._make_subset_filter(
                {"template_file": template_file, "embedded_field": self.embedded_field}
            )
            valid_db_paths = [path for path, info in self.db_info.items() if template_filter(info.model_dump())]

            if len(valid_db_paths) > 1:
                raise ValueError(f"There should not be multiple valid dbs: {valid_db_paths}")
            elif len(valid_db_paths) == 0:
                if not create_missing:
                    raise ValueError(f"No valid db exists for template {template_file}")

                db = self._create_database(template_examples, {"template_file": template_file})
            else:
                db = self.databases[valid_db_paths[0]]

            texts = [getattr(e, self.embedded_field) for e in template_examples]
            metadatas = [asdict(e) for e in template_examples]
            db.add_texts(texts, metadatas)


DB_FAISS = VectorFaissDB()

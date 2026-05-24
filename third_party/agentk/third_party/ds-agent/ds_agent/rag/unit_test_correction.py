import argparse
import os
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from ds_agent.competition_instances import BenchmarkTaskIds, get_competition_id_list
from ds_agent.data_preprocessing.log_utils import (
    LOG_KEY_COL,
    LOG_VALUE_COL,
    extract_task_id,
    extract_task_success,
    get_experiment_paths,
    make_experiment_filter,
    read_experiment_logs,
)
from ds_agent.rag.db_faiss import CodeErrorExample, VectorDBInfo, VectorDBMetadata, VectorFaissDB
from agent.tools.vector_db.faiss import FAISS_DB


def extract_unit_test_code_corrections(log_df: pd.DataFrame) -> list[CodeErrorExample]:
    """Extracts instances of a unit test being fixed from the log"""
    df = log_df
    task_id = extract_task_id(log_df)

    relevant_keys = ["templates", "memory:store:unit_test_error", "memory:store:code", "reward"]
    df = df[df[LOG_KEY_COL].isin(relevant_keys)]
    # Filter out non-code templates
    df = df[(df[LOG_KEY_COL] != "templates") | df[LOG_VALUE_COL].astype("string").str.contains("code")]

    if set(df[LOG_KEY_COL].unique()) != set(relevant_keys):
        print(f"Not all relevant keys are present: {set(df[LOG_KEY_COL].unique())}")
        return []

    # Reform long table into episode/timestep indexed rows with key columns
    df = pd.pivot_table(df, values=LOG_VALUE_COL, index=["episode", "timestep"], columns=LOG_KEY_COL, aggfunc="last")

    if set(relevant_keys) - set(df.columns):
        return []

    # Augment rows with reward and template change information
    df = df.assign(
        reward_change=df["reward"] - df["reward"].shift(), same_template=df["templates"] == df["templates"].shift()
    )

    # Filter for reward-fixing of the same template (while keeping previous row's code)
    df = df.assign(prev_code=df["memory:store:code"].shift())
    df = df[(df["reward_change"] == 1.0) & df["same_template"]]

    assert tuple(f.name for f in fields(CodeErrorExample)) == (
        "template_file",
        "code",
        "code_error",
        "code_fixed",
        "task_id",
    ), "CodeErrorExample field names have been updated since this script was written!"

    out = list(
        map(
            lambda t: CodeErrorExample(t[0][0], *t[1:], task_id),
            df[["templates", "prev_code", "memory:store:unit_test_error", "memory:store:code"]].itertuples(index=False),
        )
    )

    return out


def create_faiss_db(
    embedding_model: HuggingFaceEmbeddings,
    docs: list[str],
    metadata_dicts: Optional[list[dict]] = None,
) -> FAISS:
    assert docs, "Tried to create a FAISS DB with no documents"
    assert metadata_dicts is None or len(docs) == len(metadata_dicts), "len(metadata_dicts) should match len(docs)"

    embeddings = embedding_model.embed_documents(docs)

    db = FAISS_DB.from_embeddings(zip(docs, embeddings), embedding_model, metadatas=metadata_dicts)
    return db


def create_vector_faiss_db_from_logs(
    db_path: Path,
    model_name: str,
    embedding_fields: list[str],
    benchmark_dir: Path,
    versions: list[str],
    benchmark_id: BenchmarkTaskIds | None,
    include_failures=False,
):
    if db_path.exists():
        raise FileExistsError(f"db_path already exists: {db_path!s}")

    assert (
        embedding_fields
    ), "Some embedding field should be specified with one or more --embedding_field <field> options"
    field_names = {f.name for f in fields(CodeErrorExample)}
    assert all(
        f in field_names for f in embedding_fields
    ), f"embedding fields should be CodeErrorExample fields: {field_names}"

    experiments = get_experiment_paths([benchmark_dir], versions)

    rag_examples: list[CodeErrorExample] = []
    for name, experiment_path in experiments.items():
        if "rag" in name:
            continue
        # experiment_dfs = read_experiment_logs(experiment_path, make_experiment_filter(tasks=["higgs-boson"]))
        if benchmark_id is None:
            experiment_dfs = read_experiment_logs(experiment_path)
        else:
            experiment_dfs = read_experiment_logs(
                experiment_path,
                path_filter=make_experiment_filter(
                    tasks=[c_id.value for c_id in get_competition_id_list(benchmark_id)]
                ),
            )

        print(f"Successfully loaded and converted logs: {list(experiment_dfs.keys())}")
        for run_path, log_df in experiment_dfs.items():
            if extract_task_success(log_df) or include_failures:
                rag_examples.extend(extract_unit_test_code_corrections(log_df))

    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    if not rag_examples:
        raise ValueError("No examples found in the specified log files")

    db_path.mkdir()

    db_info = {}
    for embedded_field_name in embedding_fields:
        # Split into distinct databases
        rag_examples_by_db = defaultdict(list)

        for e in rag_examples:
            rag_examples_by_db[e.db_id(embedded_field_name)].append(e)

        for db_key, db_rag_examples in rag_examples_by_db.items():
            db = create_faiss_db(
                embedding_model,
                [getattr(e, embedded_field_name) for e in db_rag_examples],
                metadata_dicts=[asdict(e) for e in db_rag_examples],
            )
            db.save_local(db_path / db_key)
            db_info[Path(db_key)] = VectorDBInfo(
                model_name=model_name,
                embedded_field=embedded_field_name,
                template_file=db_rag_examples[0].template_file,
            )

    metadata = VectorDBMetadata(databases=db_info)
    metadata_str = metadata.model_dump_json(indent=2)
    with (db_path / VectorFaissDB.METADATA_PATH).open("w") as fp:
        fp.write(metadata_str)

    print(f"Created FAISS db: {db_path}")
    print(metadata_str)


def main():
    parser = argparse.ArgumentParser(
        description="Create a RAG db from a set of previous benchmark runs",
        usage="create-rag-db /path/to/rag/db "
              "-f code -f code_error "
              "--benchmark_dir /path/to/benchmark_results/setup_pipeline/ "
              "--version p.2.0"
    )
    # e.g. python ragflexion.py
    parser.add_argument("db_path", type=str, help="The path to the database where files and query are saved")
    parser.add_argument("--model_name", type=str, default="thenlper/gte-small", help="Embedding model name")
    parser.add_argument("-f", "--embedding_field", action="append")
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="/path/to/benchmark_results/setup_pipeline/",
        help="Benchmark log folder to scan",
    )
    parser.add_argument("-v", "--version", action="append")
    parser.add_argument(
        "--benchmark_id",
        type=str,
        default=None,
        help="Benchmark ID to restrict tasks to",
    )
    args = parser.parse_args()
    # Set umask so output files are shared
    os.umask(0o000)
    create_vector_faiss_db_from_logs(
        Path(args.db_path),
        model_name=args.model_name,
        embedding_fields=args.embedding_field,
        versions=args.version,
        benchmark_dir=Path(args.benchmark_dir),
        benchmark_id=BenchmarkTaskIds(args.benchmark_id) if args.benchmark_id else None,
    )


if __name__ == "__main__":
    main()

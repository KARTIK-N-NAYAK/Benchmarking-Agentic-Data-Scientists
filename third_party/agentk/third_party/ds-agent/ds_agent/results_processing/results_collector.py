from __future__ import annotations

from pathlib import Path
from typing import MutableMapping

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ds_agent.competition_ids import CompetitionID
from ds_agent.competition_struct import Competition
from ds_agent.results_processing.performance_results import VanillaTreeGenNonTabConfig, VanillaTreeGenTabConfig, \
    TreeGenFromRAGNonTabConfig, TreeGenFromRAGTabConfig, NonTabularAgentKConfig, AgentKWarmTreeGenNonTabConfig, \
    TabularAgentKConfig, AgentKWarmTreeGenTabConfig, deduplicate_subs, get_quantiles_from_raw_results, \
    PFNPredictionVersion, TabPFNExpConfig, TabPFNExtensionExpConfig, TabICLExpConfig, \
    TabPFNFromFinetuneClassificationExpConfig, TabPFNFromFinetuneRegressionExpConfig, \
    get_leaderboards_from_competitions, ExpConfig, AgentKTableOfResultsColname, BaseTableOfResultsColname, \
    AgentKScaffoldConfig, AIDETableOfResultsColname, ExpPathHandler
from ds_agent.results_processing.run_tracking_utils import ExpLLM, ProgressElements
from ds_agent.results_processing.setup_perf_utils import get_setup_success_stages, OutcomeName, \
    CORRECT_SETUP_STAGE_ORDER_MAP, SetupSuccessOutcome, SetupFailureOutcome, SetupSkippedOutcome, SetupNotReachedOutcome
from ds_agent.utils import ListableEnum
from third_party.data_preprocessing.env import DataPrepStagesStatusNames

TIME_CONSISTENT_SUFFIX = "time_consistent"


def get_filtered_raw_res(exp_configs: list[ExpConfig], session_state: MutableMapping[str | int, ...]) -> pd.DataFrame:
    """

    Args:
        exp_configs: exp configs to get filtered raw results from
        session_state: state that should contain raw results under exp_config fullname

    Returns:

    """
    raw_results_s = []
    for exp_config in exp_configs:
        raw_results_s.append(exp_config.filter_raw_results(raw_results=session_state[exp_config.fullname].raw_results))

    return pd.concat(raw_results_s, axis=0)


def collect_results(
        competitions: list[Competition], session_state: MutableMapping[str | int, ...],
        results_path_handler: ExpPathHandler
) -> None:
    """
    Collect results from the different experiments and put them in a session state (can be for streamlit).

    Args:
        competitions: list of competitions
        session_state: mutable struct
        results_path_handler: handle results paths
    """
    root_path_to_leaderboard = results_path_handler.path_to_leaderboards
    if SessionStateKeyHelper.get_leaderboards_key() not in session_state:
        leaderboards = get_leaderboards_from_competitions(
            competitions=competitions, root_path_to_leaderboard=root_path_to_leaderboard
        )
        session_state[SessionStateKeyHelper.get_leaderboards_key()] = leaderboards
    else:
        leaderboards = session_state[SessionStateKeyHelper.get_leaderboards_key()]

    qwen = ExpLLM.LLM_PLAYGROUND_QWEN2_5_72B
    ds_r1 = ExpLLM.DEEPSEEK_R1

    # Vanilla tree gen
    for exp_llm in [qwen, ds_r1]:
        config = VanillaTreeGenNonTabConfig(time_limit=4, exp_llm=exp_llm)
        vanilla_tree_gen_non_tab_config = config
        collect_raw_results(
            exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
            session_state=session_state, results_path_handler=results_path_handler
        )

        config = VanillaTreeGenTabConfig(time_limit=2, exp_llm=exp_llm)
        vanilla_tree_gen_tab_config = config
        collect_raw_results(
            exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
            session_state=session_state, results_path_handler=results_path_handler
        )

        clean_results_name = f"react_expl_{exp_llm.value}_clean_res"
        if clean_results_name not in session_state:
            exp_configs = [vanilla_tree_gen_non_tab_config, vanilla_tree_gen_tab_config]
            react_expl_raw_res = get_filtered_raw_res(exp_configs=exp_configs, session_state=session_state)
            _add_clean_results(
                raw_results=react_expl_raw_res, competitions=competitions, leaderboards=leaderboards,
                exp_fullname=clean_results_name, session_state=session_state, exp_config=None
            )

    # Tree gen from RAG
    exp_llm = qwen
    config = TreeGenFromRAGNonTabConfig(time_limit=4, exp_llm=exp_llm)
    react_expl_from_rag_non_tab_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    config = TreeGenFromRAGTabConfig(time_limit=2, exp_llm=exp_llm)
    react_expl_from_rag_tab_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )
    clean_results_name = f"react_expl_{exp_llm.value}_from_rag_clean_res"
    if clean_results_name not in session_state:
        exp_configs = [react_expl_from_rag_non_tab_config, react_expl_from_rag_tab_config]
        react_expl_rag_raw_res = get_filtered_raw_res(exp_configs=exp_configs, session_state=session_state)
        _add_clean_results(
            raw_results=react_expl_rag_raw_res, competitions=competitions, leaderboards=leaderboards,
            exp_fullname=clean_results_name, session_state=session_state, exp_config=None
        )

    # Agent K
    config = NonTabularAgentKConfig(time_limit=2, is_ci=False)
    agent_k_lab_non_tab_non_ci_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    config = NonTabularAgentKConfig(time_limit=2, is_ci=True)
    agent_k_lab_non_tab_ci_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    config = AgentKWarmTreeGenNonTabConfig(from_ci_cot=False, time_limit=2, cot_time_limit=2, exp_llm=qwen)
    agent_k_tree_gen_non_tab_from_non_ci_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    config = AgentKWarmTreeGenNonTabConfig(from_ci_cot=True, time_limit=2, cot_time_limit=2, exp_llm=qwen)
    agent_k_tree_gen_non_tab_from_ci_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    config = TabularAgentKConfig(time_limit=1)
    agent_k_lab_tab_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    config = AgentKWarmTreeGenTabConfig(time_limit=1, cot_time_limit=1, exp_llm=qwen)
    agent_k_tree_gen_tab_config = config
    collect_raw_results(
        exp_config=config, seeds=None, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    clean_results_name = "agent_k_clean_res"
    if clean_results_name not in session_state:
        exp_configs = [agent_k_lab_non_tab_non_ci_config, agent_k_lab_non_tab_ci_config]
        agent_k_lab_non_tab_raw_res = get_filtered_raw_res(exp_configs=exp_configs, session_state=session_state)

        agent_k_lab_raw_res = pd.concat(
            [
                get_filtered_raw_res(exp_configs=[agent_k_lab_tab_config], session_state=session_state),
                agent_k_lab_non_tab_raw_res
            ], axis=0
        )
        time_consistent_filter = agent_k_lab_raw_res[AgentKTableOfResultsColname.IS_TIME_CONSISTENT.value]
        agent_k_lab_raw_res_time_consistent = agent_k_lab_raw_res[time_consistent_filter]

        agent_k_lab_raw_dedup_res_time_consistent = deduplicate_subs(raw_results=agent_k_lab_raw_res_time_consistent)
        agent_k_lab_raw_dedup_res = deduplicate_subs(raw_results=agent_k_lab_raw_res)

        agent_k_tree_gen_non_tab_raw_res = get_filtered_raw_res(
            exp_configs=[agent_k_tree_gen_non_tab_from_non_ci_config, agent_k_tree_gen_non_tab_from_ci_config],
            session_state=session_state
        )
        agent_k_tree_gen_tab_config_raw_res = get_filtered_raw_res(
            exp_configs=[agent_k_tree_gen_tab_config], session_state=session_state
        )
        agent_k_tree_gen_raw_res = pd.concat(
            [agent_k_tree_gen_tab_config_raw_res, agent_k_tree_gen_non_tab_raw_res], axis=0
        )

        agent_k_tree_gen_raw_dedup_res = deduplicate_subs(raw_results=agent_k_tree_gen_raw_res)
        agent_k_tree_gen_clean_results = get_quantiles_from_raw_results(
            raw_results=agent_k_tree_gen_raw_dedup_res, competitions=competitions, leaderboards=leaderboards,
            include_missing=True, pbar_desc=f"**** Get clean results: Agent K - tree gen"
        )
        st_clean_results = StCleanResults(
            exp_fullname="agent_k_tree_gen_clean_res",
            raw_results=agent_k_tree_gen_raw_res,
            clean_results=agent_k_tree_gen_clean_results,
        )
        session_state[st_clean_results.exp_fullname] = st_clean_results

        cot_not_empty_filtr = agent_k_tree_gen_raw_res[AIDETableOfResultsColname.IS_COT_NOT_EMPTY.value]
        agent_k_tree_gen_from_cot_only_raw_res = agent_k_tree_gen_raw_res[cot_not_empty_filtr]
        agent_k_tree_gen_from_cot_only_raw_dedup_res = deduplicate_subs(
            raw_results=agent_k_tree_gen_from_cot_only_raw_res
        )
        agent_k_tree_gen_from_cot_only_clean_results = get_quantiles_from_raw_results(
            raw_results=agent_k_tree_gen_from_cot_only_raw_dedup_res,
            competitions=competitions, leaderboards=leaderboards,
            include_missing=True, pbar_desc=f"**** Get clean results: Agent K - tree gen"
        )
        st_clean_results = StCleanResults(
            exp_fullname="agent_k_tree_gen_from_cot_only_clean_res",
            raw_results=agent_k_tree_gen_from_cot_only_raw_res,
            clean_results=agent_k_tree_gen_from_cot_only_clean_results,
        )
        session_state[st_clean_results.exp_fullname] = st_clean_results

        # Get associate lab and tree gen phases
        agent_k_raw_res = pd.concat([agent_k_lab_raw_res, agent_k_tree_gen_raw_res])
        agent_k_raw_dedup_results = deduplicate_subs(
            raw_results=pd.concat([agent_k_lab_raw_dedup_res, agent_k_tree_gen_raw_dedup_res])
        )

        # Get best from first phase and from total phase to know the improvement
        agent_k_lab_clean_res = get_quantiles_from_raw_results(
            raw_results=agent_k_lab_raw_dedup_res, competitions=competitions, leaderboards=leaderboards,
            include_missing=True, pbar_desc=f"**** Get clean results: Agent K lab"
        )
        st_clean_results = StCleanResults(
            exp_fullname="agent_k_lab_clean_res", raw_results=agent_k_lab_raw_res, clean_results=agent_k_lab_clean_res,
        )
        session_state[st_clean_results.exp_fullname] = st_clean_results

        agent_k_lab_time_consistent_clean_res = get_quantiles_from_raw_results(
            raw_results=agent_k_lab_raw_dedup_res_time_consistent, competitions=competitions, leaderboards=leaderboards,
            include_missing=True, pbar_desc=f"**** Get clean results: Agent K lab - time consistent"
        )
        st_clean_results = StCleanResults(
            exp_fullname=f"agent_k_lab_{TIME_CONSISTENT_SUFFIX}_clean_res",
            raw_results=agent_k_lab_raw_res_time_consistent,
            clean_results=agent_k_lab_time_consistent_clean_res,
        )
        session_state[st_clean_results.exp_fullname] = st_clean_results

        agent_k_clean_res = get_quantiles_from_raw_results(
            raw_results=agent_k_raw_dedup_results, competitions=competitions, leaderboards=leaderboards,
            include_missing=True, pbar_desc=f"**** Get clean results: {clean_results_name}"
        )

        st_clean_results = StCleanResults(
            exp_fullname="agent_k_clean_res", raw_results=agent_k_raw_res, clean_results=agent_k_clean_res
        )
        session_state[st_clean_results.exp_fullname] = st_clean_results

    # Intermediate nodes
    for exp_llm in [qwen, ds_r1]:
        for time_limit, exp_cls in zip([4, 2], [VanillaTreeGenNonTabConfig, VanillaTreeGenTabConfig]):
            collect_raw_results(
                exp_config=exp_cls(time_limit=time_limit, intermediate_best_only=True, exp_llm=exp_llm),
                seeds=None, competitions=competitions, leaderboards=leaderboards,
                session_state=session_state, results_path_handler=results_path_handler
            )

    tree_gen_configs = [
        TreeGenFromRAGTabConfig(time_limit=2, intermediate_best_only=True, exp_llm=qwen),
        TreeGenFromRAGNonTabConfig(time_limit=4, intermediate_best_only=True, exp_llm=qwen),
        AgentKWarmTreeGenNonTabConfig(
            from_ci_cot=False, time_limit=2, cot_time_limit=2, intermediate_best_only=True, exp_llm=qwen
        ),
        AgentKWarmTreeGenNonTabConfig(
            from_ci_cot=True, time_limit=2, cot_time_limit=2, intermediate_best_only=True, exp_llm=qwen
        ),
        AgentKWarmTreeGenTabConfig(time_limit=1, cot_time_limit=1, intermediate_best_only=True, exp_llm=qwen),
    ]

    for tree_gen_config in tree_gen_configs:
        collect_raw_results(
            exp_config=tree_gen_config, seeds=None, competitions=competitions, leaderboards=leaderboards,
            session_state=session_state, results_path_handler=results_path_handler
        )

    pfn_configs = []
    zero_shot = PFNPredictionVersion.ZERO_SHOT
    for pfn_pred_version in [zero_shot, PFNPredictionVersion.FINE_TUNE]:
        pfn_configs.append(TabPFNExpConfig(version=pfn_pred_version))

    pfn_configs.extend(
        [
            TabPFNExtensionExpConfig(version=zero_shot),
            TabICLExpConfig(version=zero_shot),
            TabPFNFromFinetuneClassificationExpConfig(version=zero_shot),
            TabPFNFromFinetuneRegressionExpConfig(version=zero_shot),
        ]
    )

    for pfn_config in pfn_configs:
        add_results_to_session_state(
            config=pfn_config, seeds=None, competitions=competitions, leaderboards=leaderboards, compute_clean_res=True,
            session_state=session_state, results_path_handler=results_path_handler
        )


class SessionStateKeyHelper(BaseModel):

    @staticmethod
    def get_leaderboards_key() -> str:
        return "leaderboards"

    @staticmethod
    def get_ext_lb_key() -> str:
        return "extended_leaderboards"

    @staticmethod
    def get_ext_lb_with_method_key(method: str) -> str:
        return f"extended_leaderboards_{method}"

    @staticmethod
    def get_setup_success_key() -> str:
        return "agent_k_setup_success_per_types"

    @staticmethod
    def get_competition_dates_key() -> str:
        return "competition_dates"

    @staticmethod
    def get_comp_ordering_key() -> str:
        return "competition_ordering"


def collect_setup_results(
        competitions: list[Competition], session_state: MutableMapping[str | int, ...],
        results_path_handler: ExpPathHandler
) -> None:
    if SessionStateKeyHelper.get_setup_success_key() in session_state:
        return

    exp_configs: list[AgentKScaffoldConfig] = [
        NonTabularAgentKConfig(time_limit=2, is_ci=False),
        NonTabularAgentKConfig(time_limit=2, is_ci=True),
        TabularAgentKConfig(time_limit=1)
    ]

    collect_expl_setup_results(
        exp_configs=exp_configs, session_state=session_state, competitions=competitions,
        results_path_handler=results_path_handler
    )


def collect_expl_setup_results(
        exp_configs: list[AgentKScaffoldConfig], competitions: list[Competition],
        session_state: MutableMapping[str | int, ...], results_path_handler: ExpPathHandler
) -> None:
    """Collect setup success results from a list of exp config and seeds

    Args:
        exp_configs: list of exp configs
        competitions: list of competitions
        session_state: session state (can be a dictionary of a streamlit session state)
        results_path_handler: handle results paths
    """
    raw_setup_success_table_rows = []

    comp_name = BaseTableOfResultsColname.COMP_NAME.value
    for exp_config in exp_configs:
        raw_setup_success_table_rows.extend(
            exp_config.get_setup_results(competitions=competitions, results_path_handler=results_path_handler)
        )
    overall_setup_success_df = pd.DataFrame(raw_setup_success_table_rows)
    extra_col_names = [  # columns that are not really used
        AgentKTableOfResultsColname.DESCRIPTION, BaseTableOfResultsColname.SEED,
        AgentKTableOfResultsColname.SETUP_SEED
    ]
    overall_setup_success_df.drop([col_name.value for col_name in extra_col_names], axis=1, inplace=True)

    # Map values to valid outcome names
    overall_setup_success_df[overall_setup_success_df.isin(get_setup_success_stages())] = OutcomeName.SUCCESS.value
    overall_setup_success_df[
        overall_setup_success_df == DataPrepStagesStatusNames.TODO.value] = OutcomeName.NOT_REACHED.value
    overall_setup_success_df[overall_setup_success_df.isna()] = OutcomeName.SKIPPED.value

    # Drop stages that are always skipped
    overall_setup_success_df = overall_setup_success_df.loc[:,
                               ~(overall_setup_success_df == OutcomeName.SKIPPED.value).all()]

    actual_stages = set(overall_setup_success_df.columns).difference({comp_name})
    assert set(CORRECT_SETUP_STAGE_ORDER_MAP.values()) == actual_stages, actual_stages

    stage_cols = list(CORRECT_SETUP_STAGE_ORDER_MAP.values())

    # Group results per competitions
    result = pd.concat(
        [
            overall_setup_success_df.groupby(comp_name)[col].value_counts().unstack(fill_value=0) for col in stage_cols
        ], axis=1, keys=stage_cols
    )

    # Create one map to store each outcome count
    outcome_types = [SetupSuccessOutcome, SetupFailureOutcome, SetupNotReachedOutcome, SetupSkippedOutcome]
    type_to_results: dict[OutcomeName, pd.DataFrame] = {
        tag.name: result[[col for col in result.columns if col[1] == tag.name.value]] for tag in outcome_types
    }

    sum_per_comp = 0  # track overall count
    for tag in type_to_results:
        type_to_results[tag].columns = [col for col, _ in type_to_results[tag].columns]
        type_to_results[tag] = type_to_results[tag].reindex(columns=stage_cols, fill_value=0)
        sum_per_comp += type_to_results[tag]

    for tag in type_to_results:
        type_to_results[tag] = type_to_results[tag].div(sum_per_comp, axis=0).mean(0)

    session_state[SessionStateKeyHelper.get_setup_success_key()] = type_to_results


def collect_raw_results(
        exp_config: ExpConfig, seeds: list[str] | None, competitions: list[Competition],
        leaderboards: dict[CompetitionID, pd.DataFrame], session_state: MutableMapping[str | int, ...],
        results_path_handler: ExpPathHandler
) -> None:
    """Add raw results and progress tracking to the session state """
    exp_fullname = exp_config.fullname
    if seeds is None:
        seeds = exp_config.get_seeds_to_include_in_tracking()
    if exp_fullname not in session_state:
        print(f"Get results for {exp_fullname}")
        raw_res, track_els = exp_config.get_results(
            seeds=seeds, competitions=competitions, leaderboards=leaderboards, results_path_handler=results_path_handler
        )
        st_tracking = StTrackingResults(exp_fullname=exp_fullname, raw_results=raw_res, tracking_elements=track_els)
        session_state[exp_fullname] = st_tracking


def _add_clean_results(
        raw_results: pd.DataFrame, competitions: list[Competition],
        leaderboards: dict[CompetitionID, pd.DataFrame], exp_fullname: str,
        session_state: MutableMapping[str | int, ...], exp_config: ExpConfig | None
) -> None:
    """
    Add results associated with a given config and list of seeds to the session state in a basic way:
        - add raw results
        - add clean results

    Args:
        raw_results: raw results
        competitions: list of competitions
        leaderboards: leaderboards of the competitions
        exp_fullname: name under which to add the results in session_state
        session_state: state to be updated
        exp_config: experiment config
    """
    if exp_config is not None:
        raw_results = exp_config.filter_raw_results(raw_results=raw_results)
    if len(raw_results) > 0:
        deduplicate_raw_results = deduplicate_subs(raw_results=raw_results)
    else:
        deduplicate_raw_results = raw_results
    clean_res = get_quantiles_from_raw_results(
        raw_results=deduplicate_raw_results, competitions=competitions, leaderboards=leaderboards,
        include_missing=True, pbar_desc=f"Get clean results: {exp_fullname}"
    )
    st_clean_results = StCleanResults(exp_fullname=exp_fullname, raw_results=raw_results, clean_results=clean_res)
    session_state[st_clean_results.exp_fullname] = st_clean_results


def add_results_to_session_state(
        config: ExpConfig, seeds: list[str] | None, competitions: list[Competition],
        leaderboards: dict[CompetitionID, pd.DataFrame], compute_clean_res: bool,
        session_state: MutableMapping[str | int, ...], results_path_handler: ExpPathHandler
) -> None:
    """
    Add results associated with a given config and list of seeds to the session state in a basic way:
        - add raw results
        - add clean results

    Args:
          config: the exp config
          seeds: list of seeds for this exp config.
          competitions: list of competitions
          leaderboards: leaderboards of the competitions
          compute_clean_res: whether to compute clean results
          session_state: state to be updated
          results_path_handler: handle results paths
    """
    if seeds is None:
        track_seeds = config.get_seeds_to_include_in_tracking()
    else:
        track_seeds = seeds

    collect_raw_results(
        exp_config=config, seeds=track_seeds, competitions=competitions, leaderboards=leaderboards,
        session_state=session_state, results_path_handler=results_path_handler
    )

    clean_res_name = f"{config.fullname}_clean_res"
    if clean_res_name not in session_state and compute_clean_res:
        _add_clean_results(
            raw_results=session_state[config.fullname].raw_results, competitions=competitions,
            leaderboards=leaderboards, exp_fullname=clean_res_name, session_state=session_state, exp_config=config
        )


class TrackingTabName(ListableEnum):
    PFN = "PFN"
    REACT_EXPL_INTERMEDIATE_NODES = "React + Expl. Intermediate Nodes"
    REACT_EXPL = "React + Expl."
    AGENT_K_TAB = "Agent K (TAB)"
    AGENT_K_NON_TAB = "Agent K (CV/NLP)"


class StResults(BaseModel):
    exp_fullname: str
    raw_results: pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StTrackingResults(StResults):
    tracking_elements: ProgressElements


class StCleanResults(StResults):
    clean_results: pd.DataFrame

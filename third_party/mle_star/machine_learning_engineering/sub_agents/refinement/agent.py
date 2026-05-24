"""Refinement agent for Machine Learning Engineering."""

import os
import json
from typing import Optional
import functools

from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.adk import agents
from google.genai import types

from machine_learning_engineering.sub_agents.refinement import prompt
from machine_learning_engineering.shared_libraries import debug_util
from machine_learning_engineering.shared_libraries import check_leakage_util
from machine_learning_engineering.shared_libraries import common_util
from machine_learning_engineering.shared_libraries import config


def update_inner_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Updates inner loop states."""
    print("Updating inner loop states...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    callback_context.state[f"inner_iter_{task_id}"] += 1
    return None


def update_outer_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Updates outer loop states."""
    print("Updating outer loop states...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    workspace_dir = callback_context.state.get("workspace_dir", "")
    print("Workspace directory:", workspace_dir)
    task_name = callback_context.state.get("task_name", "")
    print("Task name:", task_name)
    lower = callback_context.state.get("lower", True)
    print("Is lower better?", lower)
    inner_loop_round = callback_context.state.get("inner_loop_round", 2)
    print("Inner loop round:", inner_loop_round)
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    print("Run current working directory:", run_cwd)
    prev_solution = callback_context.state.get(
        f"train_code_{step}_{task_id}", ""
    )
    #print("[[update_outer_loop_states123123]]Previous solution:", prev_solution)
    prev_exec_result = callback_context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    print("[[update_outer_loop_states123123]]Previous execution result:", prev_exec_result)
    improvements = []
    worst_possible_score = float('inf') if lower else float('-inf')
    for inner_iter in range(inner_loop_round):
        print("Evaluating improvement for inner iteration:", inner_iter)
        exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        print("[[update_outer_loop_states123123]]Current execution result:", exec_result)
        if lower:
            improvement = prev_exec_result["score"] - exec_result.get("score", worst_possible_score)
        else:
            improvement = exec_result.get("score", worst_possible_score) - prev_exec_result["score"]
        print("Improvement:", improvement)
        improvements.append(improvement)
    best_improvement = max(improvements)
    print("[[update_outer_loop_states123123]]Best improvement:", best_improvement)
    best_idx = improvements.index(best_improvement)
    print("Best index:", best_idx)
    output_filepath = os.path.join(run_cwd, f"train{step+1}.py")
    print("Output file path for updated solution:", output_filepath)
    if best_improvement <= 0.0:
        callback_context.state[
            f"train_code_{step+1}_{task_id}"
        ] = prev_solution
        callback_context.state[
            f"train_code_exec_result_{step+1}_{task_id}"
        ] = prev_exec_result
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(prev_solution)
    else:
        best_solution = callback_context.state.get(
            f"train_code_improve_{best_idx}_{step}_{task_id}", ""
        )
        best_exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{best_idx}_{step}_{task_id}", {}
        )
        callback_context.state[
            f"train_code_{step+1}_{task_id}"
        ] = best_solution
        callback_context.state[
            f"train_code_exec_result_{step+1}_{task_id}"
        ] = best_exec_result
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(best_solution)
    ablation_results = callback_context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    print("[[update_outer_loop_states123123]]Ablation results:", ablation_results)
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    #print("Code block:", code_block)
    callback_context.state[f"prev_ablations_{task_id}"].append(ablation_results)
    callback_context.state[f"prev_code_blocks_{task_id}"].append(code_block)
    callback_context.state[f"refine_step_{task_id}"] += 1
    print("Outer loop step updated to:", callback_context.state[f"refine_step_{task_id}"])
    return None


def init_inner_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Initializes inner loop states."""
    print("Initializing inner loop states...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    callback_context.state[f"inner_iter_{task_id}"] = 0
    return None


def init_outer_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Initializes outer loop states."""
    print("Initializing outer loop states...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    callback_context.state[f"refine_step_{task_id}"] = 0
    callback_context.state[f"prev_ablations_{task_id}"] = []
    callback_context.state[f"prev_code_blocks_{task_id}"] = []
    return None

def get_ablation_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation agent instruction."""
    print("Generating ablation agent instruction...")
    task_id = context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    prev_ablations = context.state.get(f"prev_ablations_{task_id}", [])
    print("Previous ablations:", prev_ablations)
    step = context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    #print("Current code:", code)
    prev_ablations_str = ""
    for i, ablation_result in enumerate(prev_ablations):
        prev_ablations_str += f"## Previous ablation study result {i+1}\n"
        prev_ablations_str += f"{ablation_result}\n\n"
    if prev_ablations_str:
        instruction = prompt.ABLATION_SEQ_INSTR.format(
            code=code,
            prev_ablations=prev_ablations_str,
        )
    else:
        instruction = prompt.ABLATION_INSTR.format(
            code=code,
        )
    return instruction


def get_ablation_summary_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation summary agent instruction."""
    print("Generating ablation summary agent instruction...")
    task_id = context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    code = context.state.get(f"ablation_code_{step}_{task_id}", "")
    #print("Current ablation code:", code)
    result_dict = context.state.get(f"ablation_code_exec_result_{step}_{task_id}", {})
    print("Ablation code execution result:", result_dict)
    return prompt.SUMMARIZE_ABLATION_INSTR.format(
        code=code,
        result=result_dict["ablation_result"],
    )


def get_init_plan_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the initial plan agent instruction."""
    print("Generating initial plan agent instruction...")
    task_id = context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    print("Current code:", code)
    ablation_results = context.state.get(f"ablation_summary_{step}_{task_id}", "")
    print("Ablation results:", ablation_results)
    prev_code_blocks = context.state.get(f"prev_code_blocks_{task_id}", [])
    #print("Previous code blocks:", prev_code_blocks)
    if not prev_code_blocks:
        instruction = prompt.EXTRACT_BLOCK_AND_PLAN_INSTR.format(
            code=code,
            ablation_results=ablation_results,
        )
    else:
        instruction = prompt.EXTRACT_BLOCK_AND_PLAN_SEQ_INSTR.format(
            code=code,
            ablation_results=ablation_results,
            prev_code_blocks=prev_code_blocks,
        )
    return instruction


def get_plan_refinement_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets plan refinement instruction."""
    print("Generating plan refinement instruction...")
    lower = context.state.get("lower", True)
    print("Is lower better?", lower)
    task_id = context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    code_block = context.state.get(f"refine_code_block_{step}_{task_id}", "")
    #print("Code block:", code_block)
    prev_plans = context.state.get(f"refine_plans_{step}_{task_id}", [])
    #print("Previous plans:", prev_plans)
    prev_exec_result = context.state.get(f"train_code_exec_result_{step}_{task_id}", {})
    print("[[get_plan_refinement_instruction123123]] Previous execution result:", prev_exec_result)
    score_plan_time_list = []
    for inner_iter, curr_plan in enumerate(prev_plans):
        exec_result = context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        print(f"[[get_plan_refinement_instruction123123]] Execution result for plan {inner_iter}:", exec_result)
        worst_possible_score = float('inf') if lower else float('-inf')
        if lower:
            improvement = prev_exec_result["score"] - exec_result.get("score", worst_possible_score)
        else:
            improvement = exec_result.get("score", worst_possible_score) - prev_exec_result["score"]
        score_plan_time_list.append((improvement, curr_plan, exec_result.get("execution_time", float('inf'))))
    num_top_plans = context.state.get("num_top_plans", 3)
    print("Number of top plans to consider:", num_top_plans)
    score_plan_time_list.sort(key=lambda x: x[0], reverse=True)
    prev_plan_summary = ""
    selected_score_plan_time_list = score_plan_time_list[:num_top_plans]
    for score, curr_plan, execution_time in selected_score_plan_time_list:
        prev_plan_summary += f"## Plan: {curr_plan}\n"
        prev_plan_summary += f"## Execution time after implement: {execution_time}s\n"
        prev_plan_summary += f"## Score: {score:.5f}\n\n"
    print("[[get_plan_refinement_instruction123123]] Previous plan summary:\n", prev_plan_summary)
    return prompt.PLAN_REFINEMENT_INSTR.format(
        code_block=code_block,
        prev_plan_summary=prev_plan_summary,
    )


def get_plan_implement_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the plan implement agent instruction."""
    print("Generating plan implement agent instruction...")
    task_id = context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    code_block = context.state.get(f"refine_code_block_{step}_{task_id}", "")
    #print("Code block:", code_block)
    plan = context.state.get(f"refine_plans_{step}_{task_id}", [""])[-1]
    print("Plan to implement:", plan)
    return prompt.IMPLEMENT_PLAN_INSTR.format(
        code_block=code_block,
        plan=plan,
    )


def check_ablation_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if the ablation study is finished."""
    print("Checking if ablation study is finished...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    callback_context.state[f"ablation_skip_data_leakage_check_{task_id}"] = True
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    result_dict = callback_context.state.get(f"ablation_code_exec_result_{step}_{task_id}", {})
    print("[[check_ablation_finish123123]]Ablation code execution result:", result_dict)
    if result_dict.get("returncode", 1) == 0:
        return llm_response_module.LlmResponse()
    callback_context.state[f"ablation_skip_data_leakage_check_{task_id}"] = False
    return None


def check_init_plan_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if the initial plan is finished."""
    print("Checking if initial plan is finished...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    code = callback_context.state.get(f"train_code_{step}_{task_id}", "")
    #print("Current code:", code)
    code_block = callback_context.state.get(f"refine_code_block_{step}_{task_id}", "")
    #print("Code block:", code_block)
    status = code and code_block and (code_block in code)
    print("Initial plan finished status:", status)
    if status:
        return llm_response_module.LlmResponse()
    return None


def check_plan_implement_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if the plan implement is finished."""
    print("Checking if plan implementation is finished...")
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
    print("Current inner iteration:", inner_iter)
    suffix = f"{inner_iter}_{step}_{task_id}"
    print("Suffix for state keys:", suffix)
    result_dict = callback_context.state.get(
        f"train_code_improve_exec_result_{suffix}", {}
    )
    print("[[check_plan_implement_finish123123]]Plan implementation execution result:", result_dict)
    callback_context.state[f"plan_implement_skip_data_leakage_check_{suffix}"] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[f"plan_implement_skip_data_leakage_check_{suffix}"] = False
    return None


def get_ablation_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Gets the ablation summary from the response."""
    print("Getting ablation summary from response...")
    response_text = common_util.get_text_from_response(llm_response)
    #print("Ablation summary response text:", response_text)
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    callback_context.state[f"ablation_summary_{step}_{task_id}"] = response_text
    return None


def get_plan_and_code_block(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Gets the plan and code block from the response."""
    print("Getting plan and code block from response...")
    response_text = common_util.get_text_from_response(llm_response)
    #print("Plan and code block response text:", response_text)
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    start_idx = response_text.find("[")
    end_idx = response_text.rfind("]") + 1
    try:
        result = json.loads(response_text[start_idx: end_idx])[0]
        plan = result["plan"]
        code_block = result["code_block"].replace("```python", "").replace("```", "")
    except Exception:
        plan = ""
        code_block = ""
    callback_context.state[f"refine_plans_{step}_{task_id}"] = [plan]
    callback_context.state[f"refine_code_block_{step}_{task_id}"] = code_block
    return None


def get_refined_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Gets the refined plan from the response."""
    print("Getting refined plan from response...")
    response_text = common_util.get_text_from_response(llm_response)
    #print("Refined plan response text:", response_text)
    task_id = callback_context.agent_name.split("_")[-1]
    print("Task ID:", task_id)
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    print("Current refinement step:", step)
    callback_context.state[f"refine_plans_{step}_{task_id}"].append(response_text)
    return None

print("Creating refinement parallel sub-agents...")
use_data_leakage_checker = config.CONFIG.use_data_leakage_checker
print("Use data leakage checker:", use_data_leakage_checker)
refinement_parallel_sub_agents = []
for k in range(config.CONFIG.num_solutions):
    ablation_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"ablation_agent_{k+1}",
        description="Perform ablation studies to improve the solution.",
        instruction=get_ablation_agent_instruction,
        before_model_callback=check_ablation_finish,
        after_model_callback=functools.partial(
            debug_util.get_code_from_response,
            do_eval=not use_data_leakage_checker,
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=1.0,
        ),
        include_contents="none",
    )
    ablation_sequential_sub_agents = [ablation_agent]
    if use_data_leakage_checker:
        data_leakage_checker_agent = check_leakage_util.get_data_leakage_checker_agent(
            prefix="ablation",
            suffix=f"{k+1}",
        )
        ablation_sequential_sub_agents.append(data_leakage_checker_agent)
        additional_agent_description = " and check if there are data leakage issues"
    else:
        additional_agent_description = ""
    ablation_sequential_agent = agents.SequentialAgent(
        name=f"ablation_sequential_agent_{k+1}",
        description=f"Perform ablation studies{additional_agent_description}.",
        sub_agents=ablation_sequential_sub_agents,
    )
    debug_inner_loop_agent = debug_util.get_debug_inner_loop_agent(
        prefix="ablation",
        suffix=f"{k+1}",
    )
    ablation_and_debug_loop_agent = agents.LoopAgent(
        name=f"ablation_and_debug_loop_agent_{k+1}",
        description="Perform ablation studies and debug the code until it succeeds.",
        sub_agents=[
            ablation_sequential_agent,
            debug_inner_loop_agent,
        ],
        max_iterations=config.CONFIG.max_rollback_round,
    )
    ablation_summary_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"ablation_summary_agent_{k+1}",
        description="Summarize the ablation study results.",
        instruction=get_ablation_summary_agent_instruction,
        after_model_callback=get_ablation_summary,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    init_plan_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"init_plan_agent_{k+1}",
        description="Generate an initial plan and a code block.",
        instruction=get_init_plan_agent_instruction,
        before_model_callback=check_init_plan_finish,
        after_model_callback=get_plan_and_code_block,
        generate_content_config=types.GenerateContentConfig(
            temperature=1.0,
        ),
        include_contents="none",
    )
    init_plan_loop_agent = agents.LoopAgent(
        name=f"init_plan_loop_agent_{k+1}",
        description=(
            "Generate an initial plan and a code block until the code block is valid."
        ),
        sub_agents=[init_plan_agent],
        before_agent_callback=init_inner_loop_states,
        max_iterations=config.CONFIG.max_retry,
    )
    init_plan_implement_agent = debug_util.get_run_and_debug_agent(
        prefix="plan_implement_initial",
        suffix=f"{k+1}",
        agent_description="Implement the initial plan to generate a solution.",
        instruction_func=get_plan_implement_agent_instruction,
        before_model_callback=check_plan_implement_finish,
    )
    plan_refine_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"plan_refine_agent_{k+1}",
        description="Refine the plan.",
        instruction=get_plan_refinement_instruction,
        after_model_callback=get_refined_plan,
        generate_content_config=types.GenerateContentConfig(
            temperature=1.0,
        ),
        include_contents="none",
    )
    plan_implement_agent = debug_util.get_run_and_debug_agent(
        prefix="plan_implement",
        suffix=f"{k+1}",
        agent_description="Implement the plan to generate a solution.",
        instruction_func=get_plan_implement_agent_instruction,
        before_model_callback=check_plan_implement_finish,
    )
    plan_refine_and_implement_agent = agents.SequentialAgent(
        name=f"plan_refine_and_implement_agent_{k+1}",
        description="Refine the plan and then implement it.",
        sub_agents=[
            plan_refine_agent,
            plan_implement_agent,
        ],
        after_agent_callback=update_inner_loop_states,
    )
    refine_inner_loop_agent = agents.LoopAgent(
        name=f"refine_inner_loop_agent_{k+1}",
        description="Refine the given solution.",
        sub_agents=[plan_refine_and_implement_agent],
        before_agent_callback=update_inner_loop_states,
        max_iterations=config.CONFIG.inner_loop_round,
    )
    ablation_and_refine_agent = agents.SequentialAgent(
        name=f"ablation_and_refine_agent_{k+1}",
        description="Perform ablation study and refine the code.",
        sub_agents=[
            ablation_and_debug_loop_agent,
            ablation_summary_agent,
            init_plan_loop_agent,
            init_plan_implement_agent,
            refine_inner_loop_agent,
        ],
        after_agent_callback=update_outer_loop_states,
    )
    ablation_and_refine_loop_agent = agents.LoopAgent(
        name=f"ablation_and_refine_loop_agent_{k+1}",
        description="Perform ablation study and refine the code for multiple rounds.",
        sub_agents=[ablation_and_refine_agent],
        before_agent_callback=init_outer_loop_states,
        max_iterations=config.CONFIG.outer_loop_round,
    )
    refinement_parallel_sub_agents.append(ablation_and_refine_loop_agent)
refinement_agent = agents.ParallelAgent(
    name="refinement_agent",
    description="Refine each solution by performing ablation studies.",
    sub_agents=refinement_parallel_sub_agents,
    before_agent_callback=None,
)

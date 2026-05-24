from agent.run_pipelines import parse_args, main, validate_args

if __name__ == "__main__":
    """
    Run Agent K
    Example:
        BLEND_AFTER_N=3 HYDRA_FULL_ERROR=1 NO_HUMAN=1  python ./src/agent/run_pipelines.py --task_id <task> 
        --prep_method <prep method> --prep_task <prep task> --ds_method <ds_method> --llm <llm required> 
        --code_llm <required>  --total_time <total run time> --attempt <attempt> --workspace_name <workspace name/ path> 
        --max_cpu <Max CPU usage> --max_setups<Max number of setups retrials allowed>
        [--tabular_task] [--is_local_task] [--run_setup_only]
    """
    args = parse_args()
    validate_args(args)
    main(
        workspace_name=args.workspace_name,
        task_id=args.task_id,
        llm=args.llm,
        code_llm=args.llm,
        is_local_task=args.is_local_task,
        is_tabular=args.tabular_task,
        total_time=args.total_time,
        max_time_per_submission=args.max_time_per_submission,
        use_ci_handling=args.use_ci_handling,
        blend_after_n=args.blend_after_n,
        alt_raw_data_root=args.alt_raw_data_root,
        max_cpu=args.max_cpu,
        attempt=args.attempt,
        attempt_spec=args.attempt_spec
    )

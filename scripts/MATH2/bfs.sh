python3.11 -B rerun.py \
    --backend o1-mini \
    --task MATH2 \
    --task_start_index 96 \
    --task_end_index 100 \
    --method_select greedy \
    --temperature 0 \
    --decompose_problem \
    --n_generate_sample 3 \
    --n_evaluate_sample 2 \
    --n_select_sample 2 \
    ${@}
python3.11 -B run.py \
    --backend Claude-3.5-Sonnet \
    --task MATH2 \
    --task_start_index 80 \
    --task_end_index 100 \
    --method_select greedy \
    --decompose_problem \
    --n_generate_sample 3 \
    --n_evaluate_sample 2 \
    --n_select_sample 2 \
    ${@}
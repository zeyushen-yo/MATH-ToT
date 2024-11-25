python3.11 -B run.py \
    --backend Qwen2.5-1.5B-Instruct \
    --task MATH2 \
    --task_start_index 0 \
    --task_end_index 100 \
    --method_select greedy \
    --apply_skills \
    --n_generate_sample 3 \
    --n_evaluate_sample 2 \
    --n_select_sample 2 \
    ${@}
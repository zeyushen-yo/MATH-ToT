python3.11 -B run.py \
    --backend Llama3.1-8B-Instruct \
    --task MATH \
    --task_start_index 0 \
    --task_end_index 5000 \
    --method_select greedy \
    --n_generate_sample 5 \
    --n_evaluate_sample 3 \
    --n_select_sample 3 \
    ${@}
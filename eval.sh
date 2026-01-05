#!/bin/bash

# ================= 配置 =================
export http_proxy=http://hk-mmhttpproxy.woa.com:11113/
export https_proxy=$http_proxy
export all_proxy=$http_proxy
export no_proxy="localhost,127.0.0.1,0.0.0.0"

# 定义日志文件
LOG_FILE="eval_main_results_$(date +%Y%m%d).log"

MODEL_LIST=(
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0004095-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0008191-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0012287-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0016383-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0020479-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0024575-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0028671-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0032767-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0036863-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0040959-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0045055-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0049151-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa/iter_0053247-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen3-kda-bsz/iter_0002047-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen3-kda-bsz/iter_0004095-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen3-kda-new-data-4m/iter_0004095-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen3-kda-new-data-4m/iter_0008191-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa-new/iter_0004095-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa-new/iter_0008191-hf"
    # "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa-new/iter_0012287-hf"
    "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-gqa-new/iter_0016383-hf"
)

HOST="0.0.0.0"
PORT=30000
TMUX_SESSION="sglang_eval"
TASKS="gsm8k,hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa,sciq,mbpp,mbpp_plus,humaneval_64,mmlu_stem"

echo "==== 评估任务启动时间: $(date) ====" | tee -a $LOG_FILE

# ================= 循环测试 =================

for MODEL_PATH in "${MODEL_LIST[@]}"; do
    echo "---------------------------------------------------------------" | tee -a $LOG_FILE
    echo "TIME: $(date)" | tee -a $LOG_FILE
    echo "MODEL: $MODEL_PATH" | tee -a $LOG_FILE

    # 1. 清理旧 session
    tmux kill-session -t $TMUX_SESSION 2>/dev/null
    sleep 2

    # 2. 启动 sglang server
    tmux new-session -d -s $TMUX_SESSION "python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host $HOST \
        --port $PORT \
        --log-level warning \
        --dp 8"

    # 3. 检查健康状态
    MAX_RETRIES=60
    COUNT=0
    while true; do
        HTTP_STATUS=$(curl -s -o /dev/null --noproxy "*" -w "%{http_code}" http://127.0.0.1:$PORT/v1/models || echo "000")
        if [ "$HTTP_STATUS" -eq 200 ]; then
            echo "Server Ready!"
            break
        fi
        COUNT=$((COUNT + 1))
        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo "Error: $MODEL_PATH Load Failed" | tee -a $LOG_FILE
            tmux kill-session -t $TMUX_SESSION
            continue 2
        fi
        sleep 10
    done

    # 4. 运行 lm_eval 并直接写入日志
    echo "Running Evaluation..."
    MODEL_ARGS="model=$MODEL_PATH,base_url=http://0.0.0.0:$PORT/v1/completions,num_concurrent=128,max_retries=0,tokenized_requests=False"

    # 使用 tee -a 将结果同步输出到屏幕和文件
    lm_eval --model local-completions \
        --tasks "$TASKS" \
        --model_args "$MODEL_ARGS" 2>&1 | tee -a $LOG_FILE

    # 5. 回收资源
    tmux kill-session -t $TMUX_SESSION
    echo "Model $MODEL_PATH Done." | tee -a $LOG_FILE
    sleep 15
done

echo "==== 所有任务于 $(date) 执行完毕 ====" | tee -a $LOG_FILE

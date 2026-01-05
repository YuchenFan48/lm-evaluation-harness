import subprocess
import time
import requests
import os

# ================= 配置区域 =================
# 填入你的模型路径列表
MODEL_PATHS = [
    "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/Qwen3Kimi-1209-57343",
    "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0131071-hf",
    "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0212991-hf",
    "/apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0389119-hf",
    # "/path/to/another/model",
]

HOST = "0.0.0.0"
PORT = 30000
TMUX_SESSION = "sglang_server"
TASKS = "gsm8k,hellaswag,arc_challenge,arc_easy,piqa,gpqa_main_n_shot,winogrande,openbookqa"

# ===========================================

def run_command(cmd):
    """执行 Shell 命令并打印"""
    print(f"执行命令: {cmd}")
    return subprocess.run(cmd, shell=True)

def is_server_ready(url, timeout=600):
    """等待 sglang server 启动就绪"""
    start_time = time.time()
    print(f"等待 server 启动 (URL: {url})...")
    while time.time() - start_time < timeout:
        try:
            # 尝试访问 v1/models 接口
            response = requests.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("Server 已就绪!")
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def main():
    for model_path in MODEL_PATHS:
        print(f"\n{'='*20} 正在处理模型: {model_path} {'='*20}")
        
        # 1. 确保旧的 tmux session 已关闭
        run_command(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null")
        time.sleep(2)

        # 2. 启动 sglang server
        # 注意: 这里使用了 --dp 8，请根据你的显卡数量调整
        launch_cmd = (
            f"tmux new-session -d -s {TMUX_SESSION} "
            f"\"python3 -m sglang.launch_server "
            f"--model-path {model_path} "
            f"--host {HOST} --port {PORT} "
            f"--log-level warning --dp 8\""
        )
        run_command(launch_cmd)

        # 3. 等待 Server 启动
        base_url = f"http://{HOST}:{PORT}"
        if not is_server_ready(base_url):
            print(f"错误: 模型 {model_path} 启动超时，跳过。")
            run_command(f"tmux kill-session -t {TMUX_SESSION}")
            continue

        # 4. 运行 lm_eval
        # 构造 model_args
        model_args = (
            f"model={model_path},"
            f"base_url={base_url}/v1/completions,"
            f"num_concurrent=256,max_retries=3,tokenized_requests=False"
        )
        
        eval_cmd = (
            f"lm_eval --model local-completions "
            f"--tasks {TASKS} "
            f"--model_args \"{model_args}\""
        )
        
        print("开始运行 lm_eval...")
        run_command(eval_cmd)

        # 5. 测试完成，关闭 server 并释放显存
        print(f"测试完成，关闭 session: {TMUX_SESSION}")
        run_command(f"tmux kill-session -t {TMUX_SESSION}")
        
        # 稍微等待几秒确保显存完全释放
        time.sleep(10)

    print("\n所有模型测试任务已完成！")

if __name__ == "__main__":
    main()

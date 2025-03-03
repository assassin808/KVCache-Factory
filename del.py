
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

from vllm import LLM

from huggingface_hub import login
login()  # 输入Hugging Face账号（需通过Meta的Llama模型访问申请）

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
print(f"当前模型版本：{llm.llm_engine.model_config._get_model_commit_hash()}")
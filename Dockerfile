# 1. 基础镜像（必须 CUDA 12.0，但这里我们不用 GPU 也没关系，仍然用这个基础镜像）
FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu18.04

# 2. 安装 Python（因为基础镜像里不一定有 python）
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# 3. 把当前目录下的所有内容复制到镜像内的 /workspace 目录
COPY . /workspace

# 4. 设定工作目录
WORKDIR /workspace

# 5. 容器启动时执行的命令（调用 python3 运行 run.py，后面两个路径是固定的）
CMD ["python3", "run.py", "/input_path", "/output_path"]

# ASR Dashboard

[English](README.en.md)

桌面语音识别应用（Python + PyQt6 Fluent 界面）。拖入音频/视频文件或整个文件夹，得到带时间戳、带说话人姓名的转录文本，并导出为 TXT / SRT / VTT / JSON / ASS / Markdown。

- **本地识别**：只提供 **Qwen3-ASR-1.7B** 和 **Qwen3-ASR-0.6B**，另配 **Qwen3-ForcedAligner-0.6B** 生成词级时间戳；30 种语言 + 22 种中文方言（自动识别）；所有参数都能在界面上调整。
- **上下文提示（热词）**：全局默认上下文 + 多个具名词库 + 本次任务的临时上下文，自动合并去重。
- **说话人分离**：pyannote Community-1（默认）、DiariZen（隔离环境）、WavLM 兜底；本地和 API 引擎都能用。
- **声纹库**：登记说话人的声音和名字，转录后自动把 `SPEAKER_00` 换成真实姓名（声纹只保存在本机）。
- **批量处理**：严格按顺序逐个处理文件夹，中断后从断点继续，保留子文件夹结构，生成批次报告。
- **API 引擎**：OpenAI 兼容转写、OpenAI 兼容 Chat 音频（远程 Qwen3-ASR）、阿里云百炼、自定义 HTTP。
- **智能后处理**：配置一个大模型 API（保存时验证），文字整理交给大模型；未配置时使用规则化处理。
- **实时转录**：连接自部署 vLLM 或百炼的流式识别服务，边录边出字，停止后自动精修。

---

## 目录
1. [安装](#安装)
2. [下载模型](#下载模型)
3. [使用](#使用)
4. [建立声纹库](#建立声纹库)
5. [配置 API 模型](#配置-api-模型)
6. [批量处理和断点续跑](#批量处理和断点续跑)
7. [配置智能后处理](#配置智能后处理)
8. [部署实时转录服务](#部署实时转录服务)
9. [配置文件与密钥](#配置文件与密钥)
10. [打包](#打包)
11. [常见问题](#常见问题)
12. [开发](#开发)

---

## 安装

### 1. 安装 uv
```powershell
# Windows（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装 FFmpeg（必需）
- Windows：`winget install Gyan.FFmpeg`，或从 <https://www.gyan.dev/ffmpeg/builds/> 下载后把 `bin` 目录加入 PATH。
- Ubuntu / Debian：`sudo apt install ffmpeg`
- 启动时程序会检测 FFmpeg，缺失时弹出安装指引；设置页的“环境诊断”也会显示。

### 3. 安装依赖（含 PyTorch）
Python 要求 3.11–3.13。在项目目录中选择**一种** PyTorch 版本：

```bash
uv sync --extra cu128     # NVIDIA 显卡（CUDA 12.8）
uv sync --extra rocm      # AMD 显卡（ROCm 6.4，仅 Linux；PyTorch 中同样显示为 cuda）
uv sync --extra cpu       # 纯 CPU
uv sync                   # 使用 PyPI 默认的 torch（Linux 为 CUDA 版，Windows 为 CPU 版）
```

可选组件：
```bash
uv sync --extra cu128 --extra flash-attn   # FlashAttention 2（需要编译环境，fp16/bf16 时可选）
uv sync --extra cu128 --extra vllm         # vLLM 推理后端（仅 Linux）
uv sync --extra cu128 --extra dashscope    # 阿里云 DashScope SDK（可选，程序本身直接调 HTTP）
```

> **qwen-asr 与 Transformers 原生后端不能装在同一环境。** 官方 `qwen-asr` 包固定 `transformers==4.57.6`，而 Transformers 原生支持 Qwen3-ASR（`-hf` 模型）从 **5.13** 才开始。因此依赖分成两个互斥的依赖组：
> - 默认组 `qwen`：`qwen-asr` 后端（推荐）。
> - `hf` 组：Transformers 原生后端（支持 `torch.compile`）：`uv sync --extra cu128 --no-group qwen --group hf`
>
> 不可用的后端在设置页置灰，并显示原因和安装命令。

### 4. 启动
```bash
uv run asr-dashboard        # 或 uv run python main.py
```
Windows 可直接双击 `ASR-Dashboard.bat`（首次运行会自动 `uv sync --extra cu128`）。

> `uv.lock` 需要在能访问 `download.pytorch.org` 的环境中执行一次 `uv lock` 生成（三个 PyTorch 源都会被解析）。

---

## 下载模型

### Qwen3-ASR（HuggingFace / ModelScope / 本地目录）
设置 → 本地模型：
- **模型来源**：HuggingFace（默认）、ModelScope（需 `uv pip install modelscope`）、本地目录。
- **模型缓存目录**：留空用默认位置；填写后会设置 `HF_HOME` / `MODELSCOPE_CACHE`。
- 点击 **下载模型** 显示进度；**预热** 把模型加载到内存（日志中会打印最终传给 `from_pretrained` 的参数）；**查看缓存** 显示磁盘占用并可删除。

也可以用命令行预先下载：
```bash
uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B
uv run huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B
# Transformers 原生后端使用 -hf 版本：Qwen/Qwen3-ASR-1.7B-hf、Qwen/Qwen3-ForcedAligner-0.6B-hf
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B
```

### pyannote Community-1（说话人分离默认后端，CC-BY-4.0）
这是受限模型，首次使用：
1. 在 <https://huggingface.co/settings/tokens> 创建 Token，填到 设置 → 说话人分离 → HF Token。
2. 打开 <https://huggingface.co/pyannote/speaker-diarization-community-1>，登录并同意使用条款。
3. 点击 **下载 Community-1**，模型完整下载到本地目录（默认 `~/.asr-dashboard/models/pyannote-community-1`）。
4. 之后一律从本地目录离线加载，不再需要联网和 Token。

### DiariZen（可选，**模型权重仅限非商业用途**）
DiariZen 自带改过的 pyannote-audio 并固定 PyTorch 2.1.1 / Python 3.10，不能与 pyannote.audio 4.x 装在同一环境，所以程序把它安装在独立环境 `~/.asr-dashboard/envs/diarizen/`，通过子进程调用。
1. 设置 → 说话人分离：勾选“确认非商业许可”。
2. 点击 **安装 DiariZen 环境**（需要 `uv` 和 `git`；会执行 `uv venv --python 3.10`、安装 torch 2.1.1、`git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen` 并以可编辑方式安装 DiariZen 和其中的 pyannote-audio）。
3. 默认模型 `BUT-FIT/diarizen-wavlm-large-s80-md-v2`，可在设置中修改。

---

## 使用

1. **单文件**：选择或拖放文件 → 选择模型（“本地”或“API”分组，⚙ 打开该引擎的设置）→ 语言 → 可选词库与临时上下文 → 处理选项 → 开始。进度区显示当前阶段、已用时间与预计剩余时间，可随时取消。
2. **结果**：表格 / 纯文本 / “显示修改”视图；双击编辑文本；双击说话人重命名（所有段落同步）；右键说话人“保存到声纹库”；点击任意一行从对应时间播放原音频；导出 TXT / SRT / VTT / JSON / ASS / Markdown；显示警告与各阶段耗时。
3. **历史**：所有任务记录在 `~/.asr-dashboard/history.json`，可重新打开、重新导出、删除。

时间戳策略：总是先分段再识别（混合策略：目标 60 秒，在切点 ±10 秒内找最长静音切开）。开启强制对齐时按词级时间戳重切成句子（句末标点 / 停顿 > 0.6 秒 / 单句 > 15 秒），分段最长自动限制为 180 秒（qwen-asr 自身对对齐输入的上限）；识别语言不在对齐器支持的 11 种之内时，该块改用分段边界并给出警告。

上下文提示是“软引导”：Qwen3-ASR 没有专门的热词参数，上下文作为系统提示传给模型（qwen-asr 的 `context`、Transformers 的 `prompt`），能提高但不能保证专有名词的准确率。

---

## 建立声纹库

声纹库页面：
- **新增说话人** → 选择 **从文件添加样本**（可指定时间段）或 **麦克风录音**（显示音量和时长），建议干净的单人语音 ≥ 10 秒；时长不足或疑似多人时会提醒。
- 结果页右键某个说话人 → **保存到声纹库**：自动挑选该说话人最长、不与他人重叠的片段（约 30 秒）。
- 转录后自动匹配：每个说话人的平均声纹与库内每个人做余弦相似度，匈牙利算法一对一分配，≥ 阈值（默认 0.6）才填入姓名；结果页显示匹配得分。设置页可“用库内样本校准”阈值。
- 声纹向量必须由同一模型生成：默认 pyannote Community-1 自带声纹模型，可改为 WavLM；切换后在声纹库页面点击“用当前声纹模型重新计算”（原始音频片段一直保留）。
- 可导出 / 导入 zip，迁移到其他电脑。声纹数据只保存在本机，绝不发送给任何 API。

---

## 配置 API 模型

API 模型页面：从预设新建或选择接入类型 → 填写地址、Key、模型名 → **测试连接**（用内置 3 秒测试语音真实调用一次，显示耗时、原始响应和解析结果）→ 保存。保存后会出现在首页模型下拉框的“API”分组。

| 接入类型 | 说明 |
|---|---|
| `openai_transcriptions` | `POST {base_url}/audio/transcriptions`。优先 `verbose_json` 拿时间戳，不支持时自动降级为 `json`。上下文走 `prompt` 字段。Azure 预设使用 `{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version=...` 和 `api-key` 头。 |
| `openai_chat_audio` | `POST {base_url}/chat/completions`，`audio_url` 放 base64 data URL；用于远程 `vllm serve Qwen/Qwen3-ASR-1.7B` / `qwen-asr-serve`；解析 `language X<asr_text>...`。 |
| `dashscope_qwen` | 百炼录音文件识别（异步提交 + 轮询）。本地文件先上传到百炼免费临时存储（`oss://`，48 小时有效）。地域：中国大陆 / 国际站。 |
| `custom_http` | 方法、URL / 请求头 / 请求体模板（`{{api_key}}` `{{model}}` `{{language}}` `{{context}}` `{{audio_base64}}`），multipart / JSON+base64 / 原始二进制；JSONPath 提取全文与分段（秒或毫秒）；可选异步轮询。 |

通用行为：超过文件大小 / 时长上限自动分段（必要时转 16 kHz 单声道 FLAC / MP3 压缩）；429 / 5xx / 超时指数退避重试并遵守 `Retry-After`；可配置并发；取消时立即中止请求；日志和错误信息中的 Key 一律打码。API 没返回时间戳时，先按分段规则切块再逐块发送，因此本地说话人分离和声纹识别同样可用。

内置预设：OpenAI（whisper-1、gpt-4o-transcribe）、Groq（whisper-large-v3-turbo）、Azure OpenAI、阿里云百炼 Qwen3-ASR（qwen3-asr-flash-filetrans）、自部署 vLLM Qwen3-ASR、智谱 GLM-ASR（glm-asr-2512）。预设只预填地址和字段，模型名与限制请以服务商文档为准（见“核实情况”）。

---

## 批量处理和断点续跑

批量处理页面：选择文件夹 → 设置扫描选项（子文件夹、扩展名、隐藏/空文件）和排序（自然排序 `2.mp3` 在 `10.mp3` 前；修改时间；大小）→ 预览清单，可拖动排序、移除、追加 → 开始。
- 严格串行，一个 worker；模型在批次开始时加载一次并常驻。
- 每个文件：等待 / 处理中 / 完成 / 失败 / 跳过 / 已取消。失败（如损坏文件）记录错误后继续；显存不足时批大小减半重试一次。
- 暂停（当前文件完成后停下）/ 继续 / 取消；结束后可“重试失败项”。
- Windows 上处理期间阻止系统睡眠。
- 输出目录保留原来的子文件夹结构，写入 `batch_manifest.json`（每完成一个文件就以“临时文件 + 改名”的方式原子更新）、`batch_report.csv`、`batch_report.json`。
- 程序启动时检测未完成的批次，询问“从第 N 个文件继续”或“放弃”。已完成文件的判断：清单中状态为完成、文件大小和修改时间未变、输出文件存在。
- 再次处理同一个文件夹时，已有输出的文件按设置跳过（默认）/ 覆盖 / 另存为新文件名。

---

## 配置智能后处理

设置 → 智能后处理：填写 OpenAI 兼容的 API 地址、Key、模型名，点击 **保存并验证**（发送一个要求返回 `{"ok": true}` 的小请求，20 秒超时；分别提示网络 / 认证失败 / 模型不存在 / 返回无法解析，并检测是否支持 `response_format=json_object`）。验证失败不会保存。

| 服务 | API 地址 | 模型名示例 | Key |
|---|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | 必填 |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | 必填 |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen3:8b` | 留空 |
| 本地 LM Studio | `http://localhost:1234/v1` | 已加载的模型名 | 留空 |

防止大模型乱改原文：按顺序分块（每块 ≤ 1500 字，附前两段只读上下文），输入输出都是 `[{id, speaker, text}]`，只能改 `text`；返回的 id 集合必须一致、每条改动比例 ≤ 40%、JSON 必须可解析，否则这一块改用规则处理并写入警告。时间戳、说话人、词级数据不交给模型。修改前的文字保存在 `original_text`，结果页“显示修改”可逐条或全部撤销。可选：摘要和要点、推测说话人身份（只作为建议，需点击“采用”）。首次启用非本机地址时会提示转录文字将发送到该地址。

---

## 部署实时转录服务

程序本身不运行流式识别，实时转录通过 WebSocket 连接外部服务。设置 → 实时转录服务 中填写并 **保存并验证**（发送内置测试语音，10 秒内收到非空结果才算通过）后，导航中的“实时转录”才会开启。

### vLLM（Linux / WSL2）
```bash
uv venv && source .venv/bin/activate
uv pip install "vllm[audio]>=0.32"
./server/start_realtime.sh 1.7b 8000 0.8      # 或 0.6b；参数：规格 端口 显存占比
```
`start_realtime.sh` 实际执行：
```bash
vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.8 \
  --hf-overrides '{"architectures": ["Qwen3ASRRealtimeGeneration"]}'
```
vLLM 只有在模型实现了实时接口时才开放 `/v1/realtime`；Qwen3-ASR 对应的实时架构是 `Qwen3ASRRealtimeGeneration`（vLLM 源码 `model_executor/models/qwen3_asr_realtime.py`），而模型文件声明的是 `Qwen3ASRForConditionalGeneration`，因此需要 `--hf-overrides`。设置 `REALTIME=0` 可按普通模式启动（只提供 `/v1/chat/completions` 和 `/v1/audio/transcriptions`，供 `openai_chat_audio` API 引擎使用）。

Docker（需要 NVIDIA Container Toolkit）：
```bash
cd server && MODEL=Qwen/Qwen3-ASR-0.6B PORT=8000 GPU_UTIL=0.5 docker compose up -d
```
也可以使用官方镜像 `qwenllm/qwen3-asr`（包含 qwen-asr 与 `qwen-asr-serve`）。

程序中填写：协议 `vLLM /v1/realtime`，地址 `ws://<服务器IP>:8000/v1/realtime`，模型名与服务端一致（如 `Qwen/Qwen3-ASR-1.7B`）。

**Windows 用户用 WSL2 在本机运行：**
1. 管理员 PowerShell：`wsl --install -d Ubuntu`，重启后进入 Ubuntu。
2. 安装 NVIDIA Windows 驱动即可（WSL2 内无需另装驱动），`nvidia-smi` 能看到显卡。
3. 在 WSL2 中按上面的 vLLM 步骤安装并启动。
4. 程序中填 `ws://localhost:8000/v1/realtime`（WSL2 默认把端口转发到 Windows）。从局域网其他电脑访问时，需要在 Windows 防火墙放行 8000 端口，并用 `netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL IP>` 转发。

**与本机程序共用一块显卡**：调低显存占比（如 `0.4`–`0.5`），流式服务建议使用 0.6B 模型，本机识别模型设置“模型常驻内存”关闭。

**注意**：vLLM 的 `/v1/realtime` 只接受模型名，不接受提示词和语言参数，所以词库在实时阶段不生效（停止后的精修会使用）。流式识别质量低于整段识别（服务端按固定长度切块、无上下文、无时间戳），因此推荐开启“结束后自动精修”：停止后用当前引擎把录音完整处理一遍，正式结果进入结果页和历史，实时草稿保留在 JSON 的 `draft_text`。

### 阿里云百炼实时语音识别
协议选“阿里云百炼实时语音识别”，地址 `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`（国际站 `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime`），模型 `qwen3-asr-flash-realtime`，填写 API Key。词库会作为 `corpus` 发送。

实时页面功能：麦克风 / 系统声音（Windows WASAPI 回环，可直接转录会议软件的声音）/ 混合；音量电平；灰色临时文字 + 带本地时间的确定文字；向上翻看时暂停自动滚动；可选实时说话人标注（本地声纹比对）；一边录一边存 wav + txt + srt；断线自动重连，重连期间缓存最多 60 秒音频并补发，超出部分插入“[连接中断 xx 秒]”。

---

## 配置文件与密钥

- `config.yaml`（程序目录）由 pydantic 校验：文件缺失或字段非法时使用默认值并在日志/设置页提示，不会崩溃。示例见 `config.example.yaml`。
- API Key、LLM Key、实时服务 Key、HF Token **不写入** `config.yaml`：优先存系统钥匙串（`keyring`），不可用时存 `~/.asr-dashboard/secrets.yaml` 并在界面提示明文存储。
- 日志在 `~/.asr-dashboard/logs/`（可在设置中修改），所有密钥打码。

---

## 打包

```bash
uv pip install pyinstaller
uv run python build_exe.py
```
打包版只支持 CPU 或 CUDA（取决于构建环境中安装的 PyTorch），不包含 DiariZen 隔离环境（解包后在设置中安装）、vLLM 和模型权重。

---

## 常见问题

- **显存不足**：调小“最大推理批大小”、改用 0.6B、关闭强制对齐或把对齐器放到 CPU；开启“显存不足时回退 CPU”后会自动回退并提示。批量处理遇到 OOM 会先把批大小减半重试一次。
- **FlashAttention**：需要 `flash-attn` 和 fp16/bf16 精度；未安装时该选项置灰。Windows 上编译困难，可直接用 `sdpa`。
- **Windows 上无法使用 vLLM**：vLLM 只支持 Linux；Windows 请用 qwen-asr 后端，或在 WSL2 中运行 vLLM 服务并用 `openai_chat_audio` / 实时转录连接。
- **DiariZen 的许可限制**：代码 MIT，但预训练权重 CC BY-NC 4.0，仅限非商业用途；商业场景请使用 pyannote Community-1（CC-BY-4.0）。
- **qwen-asr 和 Transformers 原生后端冲突**：见安装第 3 步，两者是互斥依赖组。
- **pyannote 下载 403**：需要先在模型页面同意使用条款，且 Token 有读取权限。

---

## 实现前核实情况（⚠ 项）

| 项目 | 结论 | 依据 |
|---|---|---|
| qwen-asr 用法 | `Qwen3ASRModel.from_pretrained(path, forced_aligner=, forced_aligner_kwargs=, max_inference_batch_size=, max_new_tokens=, **AutoModel kwargs)`；`transcribe(audio, context="", language=None, return_time_stamps=False)`；`Qwen3ASRModel.LLM(model=, ...)`；`Qwen3ForcedAligner.from_pretrained(...).align(audio, text, language)` | qwen-asr 0.0.6 源码 + GitHub README |
| 上下文参数 | qwen-asr / vLLM：`context`（系统消息）；Transformers：`apply_transcription_request(prompt=...)`（系统消息） | 源码 |
| 对齐器上限 | README 写 5 分钟；qwen-asr 代码 `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`，本程序按 180 秒切块 | 源码 |
| Transformers 原生支持 | 从 **5.13.0** 起；类 `Qwen3ASRForConditionalGeneration`（`AutoModelForSpeechSeq2Seq` 映射），对齐器 `Qwen3ASRForTokenClassification`，有 `-hf` 版本（`Qwen/Qwen3-ForcedAligner-0.6B-hf`） | 5.12.1 / 5.13.0 / 5.22.1 wheel 对比 |
| qwen-asr 与 transformers 5 | qwen-asr 0.0.6（GitHub main 也一样）固定 `transformers==4.57.6` → 两个后端分组安装 | PyPI 元数据 + GitHub main |
| pyannote.audio 4.x | `Pipeline.from_pretrained(checkpoint, token=)`，本地目录可离线；输出 `speaker_diarization` / `exclusive_speaker_diarization` / `speaker_embeddings`；要求 torch ≥ 2.8 | 4.0.7 源码 |
| DiariZen | conda/py3.10 + torch 2.1.1 + 自带 pyannote-audio；`DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")`；最新 `-v2` 模型；权重 CC BY-NC 4.0 | GitHub README |
| vLLM realtime | `/v1/realtime`：`session.created`、`session.update{model}`、`input_audio_buffer.append/commit(final)`、`transcription.delta/done`、`error`；不接受 prompt/语言；必须加 `--hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}'` | vLLM 0.32.0 源码 + 官方 supported_models 文档 + 官方示例客户端 |
| DashScope 录音文件识别 | `POST /api/v1/services/audio/asr/transcription` + `X-DashScope-Async: enable`，`GET /api/v1/tasks/{id}`，结果 `transcription_url` | dashscope 2.0.3 SDK 源码 |
| DashScope 临时存储与请求体 | `input.file_url`（单个 URL）、`parameters.channel_id=[0]`、`enable_itn`、`enable_words`；临时存储 `oss://` 48 小时有效，需 `X-DashScope-OssResourceResolve: enable` | 官方 API 参考页（检索摘要；站点在本环境被拦截，未能直接打开） |
| DashScope 实时 | `wss://…/api-ws/v1/realtime?model=qwen3-asr-flash-realtime`，`session.update`（corpus/language/server_vad）；临时结果 `…input_audio_transcription.text`（`text` 为已确定前缀 + `stash` 为待定后缀），最终结果 `…completed`；`session.finish` | SDK 源码 + 官方服务端事件文档（检索摘要） |
| WASAPI 回环库 | 选 PyAudioWPatch（有预编译 Windows wheel、持续发布） | PyPI |
| GLM-ASR | `/api/paas/v4/audio/transcriptions`，模型 `glm-asr-2512`；≤ 30 秒、≤ 25 MB、仅 wav/mp3（预设因此只用 mp3 压缩）；官方未写明 prompt 参数（预设默认关闭上下文） | 官方文档（检索摘要） |
| OpenAI / Groq | `gpt-4o-transcribe` 只支持 `json`；`whisper-1` 支持 `verbose_json` + `timestamp_granularities`；Groq `whisper-large-v3-turbo` | 检索结果 |

---

## 开发

```bash
uv sync --extra cpu
uv run ruff check .
uv run pytest                 # 默认跳过 @pytest.mark.gpu / network
ASR_TEST_DIALOG=对话.wav ASR_TEST_LONG=长音频.mp3 uv run pytest -m gpu -s   # 需要真实模型和显卡
```

目录结构见 `src/asr_dashboard/`：`config/`（pydantic 配置、密钥）、`core/`（音频、分段、VAD、说话人分离、声纹、pipeline、任务与批量）、`engines/`（本地 Qwen3 三种后端、四种 API 接入）、`export/`、`text/`（规则、大模型后处理）、`realtime/`（采集、协议、会话、验证）、`ui/`（页面与控件）。服务端脚本在 `server/`。

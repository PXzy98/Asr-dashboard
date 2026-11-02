# 验收报告（开发环境：云端 Linux 容器，无 GPU）

开发环境的限制：没有 GPU；`huggingface.co`、`download.pytorch.org`、阿里云/智谱文档站点被网络策略拦截；没有麦克风；没有可用的 vLLM / 百炼 / OpenAI 服务。所以凡是需要真实模型权重、真实服务或真实录音的条目都**没有实际跑过**，下面如实标注。

| # | 条目 | 结果 | 说明 |
|---|---|---|---|
| 1 | `uv sync` 后启动；下拉框只有 Qwen3 1.7B / 0.6B + API 模型 | **部分通过** | `main.py` 在 offscreen 模式下正常启动并运行（开发 venv）；下拉框内容由 `test_ui_smoke` 验证。`uv.lock` 无法生成（PyTorch 源不可达）；去掉 PyTorch 源后依赖图可解（464 个包：transformers 4.57.6、torch 2.9.1、pyannote.audio 4.0.7）。需要在你的机器上执行 `uv lock` / `uv sync`。 |
| 2 | 3.4 节设置可改、重启保留、真正传给模型（日志打印加载参数） | **逻辑通过，真实后端未跑** | 设置页保存 → config.yaml → 重新读取（UI 测试）；引擎把参数传给后端并在日志打印 `from_pretrained` 参数（假后端测试）。三种真实后端都没有在本环境加载过。 |
| 3 | 词库提升专有名词准确率 | **未执行** | 需要真实模型和含专有名词的音频。上下文合并/截断与传参（qwen-asr `context`、Transformers `prompt`）已测试。 |
| 4 | 3 分钟双人对话：开/关对齐、pyannote 与 DiariZen | **未用真实模型执行** | 用假引擎/假分离器验证：时间戳非零且单调、说话人交替、词级切分、关闭对齐时用分段边界。DiariZen 子进程协议用假 worker 测试；隔离环境安装流程未实际运行。 |
| 5 | 声纹库自动识别姓名；从结果页保存后能被识别 | **逻辑通过，真实声纹模型未跑** | 存储、匈牙利匹配、阈值、冲突、zip 导入导出均有测试；流水线中用假 embedder 验证姓名替换。 |
| 6 | 20 个文件 + 1 个损坏文件的批量处理、强杀后续跑、目录镜像、报告 | **逻辑通过** | 用假处理函数验证严格顺序、损坏文件失败后继续、模拟崩溃后从第 8 个继续、完成判断、子目录镜像、CSV/JSON 报告；FFmpeg 处理损坏文件抛错单独测试。未用真实模型跑完整批次。 |
| 7 | 30 分钟音频分段推进、取消 2 秒内停止、显存释放 | **取消机制已修复并验证，真实长音频未跑** | 原实现只在批次之间检查取消，GPU 上一个批次可能超过 2 秒。现在通过 `stopping_criteria` 在每个解码步检查取消（qwen-asr 和 Transformers 后端）；用小型随机模型实测：取消后 0.01 秒内停止生成。真实长音频用 `ASR_TEST_LONG=... uv run pytest -m gpu` 验证。vLLM 后端无法中途中断。 |
| 8 | OpenAI 兼容 API“测试连接”+ 转录 + 本地说话人分离 + 声纹 | **用本地假服务通过** | UI 测试：在 API 模型页新建 → 保存 → 测试连接 → 首页用它转录 → 结果页显示。流水线测试：API 引擎 + 假分离器 + 假声纹得到姓名。未连接真实 OpenAI。 |
| 9 | 超过 25MB 的文件经 25MB 限制自动分段并正确拼接 | **通过（缩小比例）** | 120 秒音频配 0.5MB 上限：压缩为 FLAC/MP3 后仍超限 → 在静音处切分 → 结果按偏移拼接。 |
| 10 | 智能后处理：错误 Key 提示“认证失败”、显示“大模型：{模型名}”、可撤销、清除后显示“规则”、Ollama | **用模拟服务通过** | 验证分类（网络/认证/模型不存在/无法解析/成功、JSON 模式探测）、分块校验回退、撤销、模式选择。未连接真实 DeepSeek/Ollama。 |
| 11 | 实时转录：未配置置灰、vLLM 验证、出字、断线重连不丢录音、停止后精修 | **用本地假 vLLM 服务通过（协议层）** | 验证/错误分类、临时与确定文字、断线重连补发（在途音频最多重发 0.5 秒）、服务端错误保留录音；导航置灰由 UI 测试验证。未连接真实 vLLM，未用麦克风。 |
| 12 | config.yaml、日志、错误对话框中没有明文 Key/Token | **通过** | 配置测试、日志过滤测试、UI 测试检查 config.yaml。 |
| 13 | 所有测试通过，ruff 无报错 | **通过** | `pytest`：116 passed（gpu/network 标记默认跳过）；`ruff check .`：All checks passed。 |

## 第二轮修复（解决发现的问题）
- qwen-asr 与 Transformers 5 冲突：在 transformers 5.22.1 上实测 `import qwen_asr` 失败（`check_model_inputs() missing 'func'`），确认冲突真实存在，保留两个互斥依赖组；两条路径用到的 API 签名都已在真实安装的包上核对一致。
- 对齐器 180 秒：qwen-asr GitHub main 仍是 `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`，保持按 180 秒切块。
- vLLM 实时启动参数：vLLM 官方文档写明必须加 `--hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}'`；客户端消息顺序与官方示例一致。
- 百炼：请求体字段已确认，新增 `enable_words`；实时临时结果事件（text + stash）已确认。
- 智谱 GLM-ASR：只支持 wav/mp3 → 预设只用 mp3 压缩；未写明 prompt 参数 → 预设默认关闭上下文。
- 实时断线重连：补发的 0.5 秒音频可能导致重复文字 → 重连后第一句去掉与上一句重复的开头。
- 取消 2 秒内停止：见第 7 条。
- 新增真实模型测试 `tests/test_real_models.py`（`-m gpu`），以及真实 Silero VAD 测试（已在装有 torch 的环境中通过：混合分段的切点都落在静音中）。
- 仍无法在本环境解决：`uv.lock`（需要访问 download.pytorch.org）、真实模型权重（huggingface.co / modelscope.cn 被拦截）、GPU。

## 需要你在本机完成的验证
1. `uv lock && uv sync --extra cu128`，启动后在 设置 → 环境诊断 中确认后端可用。
2. 用真实模型跑：`ASR_TEST_DIALOG=对话.wav ASR_TEST_LONG=长音频.mp3 uv run pytest -m gpu -s`（覆盖条目 2、4、5、7；日志 `~/.asr-dashboard/logs/asr-dashboard.log` 中有 `Qwen3-ASR 加载参数`）。
3. 部署 `server/start_realtime.sh` 后验证条目 11。

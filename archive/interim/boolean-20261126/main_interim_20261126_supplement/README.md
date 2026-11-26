# Boolean 主实验中途诊断快照（补充）

仅供 ABCD 离线分析，不是最终结果。这一份补充 `main_interim_six_methods_20261126` 里只含前 11 题的快照，不替换它。

六种方法都完成 16 个名额的共同题目现在是索引 0–16，共 17 题。uniform 和 round_robin 另外完成了索引 17–19。memory 的第 18 题只有 9 条记录，没有收进本快照；四种方法的第 18–20 题其余部分也还没跑完。具体范围见 snapshot.json。

candidates.jsonl.gz 是 gzip 压缩的 UTF-8 JSONL。解压后每条包含 method、question_index、候选、完整 prompt/response/reasoning_content、attempts、usage、反馈、控制器与结构建议日志。JSON 中的 null 用量表示未知，不应按零收费处理。

protocol.json 和 manifest.json 记录模型参数、题目清单、源码哈希和费用上限。当前主实验不得根据这份中途成绩修改策略、挑选题目或补位；若提出新方法，回开发集验证，再使用独立测试集。

保留中断未知记录。最终应报告所有方法一致排除受中断影响题目的敏感性分析。此次快照不包含密钥、请求授权头、账户账单、checkpoint 或锁文件。费用是预留估算，不是实际账单。

这是本地可见历史排除后的单 seed 实验，测试集暴露范围仍需队友确认。

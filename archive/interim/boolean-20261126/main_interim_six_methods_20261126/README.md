# Boolean 主实验中途诊断快照

仅供 ABCD 离线分析，不是最终结果。运行仍在继续；只收录六种方法均完成16名额的共同题目，具体索引见 snapshot.json。包含 uniform 和 round_robin；只比较六种方法共同完成的题目。

candidates.jsonl.gz 是 gzip 压缩的 UTF-8 JSONL。解压后每条包含 method、question_index、候选、完整prompt/response/reasoning_content、attempts、usage、反馈、控制器与结构建议日志。JSON中的null用量表示未知，不应按零收费处理。

protocol.json 和 manifest.json 记录模型参数、题目清单、源码哈希和费用上限。当前主实验不得根据这份中途成绩修改策略、挑选题目或补位；若提出新方法，回开发集验证，再使用独立测试集。

重点分析：SDE建议服从率；VAR/k=0在公开观测下不可行或已耗尽时的分配；各动作重复率与新有效产出；截断发生在恢复全部答案之前还是之后。无解动作也可能生成偏离目标但有效的表达式，不能把所有这些名额简单算作零收益。

保留中断未知记录。最终应报告所有方法一致排除受中断影响题目的敏感性分析。此次快照不包含密钥、请求授权头、账户账单、checkpoint或锁文件。费用是预留估算，不是实际账单。

这是本地可见历史排除后的单seed实验，测试集暴露范围仍需队友确认。

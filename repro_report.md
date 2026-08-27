# Q1-Q5 复现与一致性报告（coding 负责人：安泰）

日期：2026-08-27  
分支：`coding/repro-baseline`（基线提交 `2cb7633`）  
机器：Windows，Python 3.12.10（`D:\python3.12.10\python.exe`），虚拟环境 `.venv`

## 1. 环境

| 项目 | 本次环境 | 官方基线环境 |
|---|---|---|
| Python | 3.12.10 | 3.9.13 |
| NumPy | 2.5.2 | 2.0.2 |
| SciPy | 1.18.1 | 1.13.1 |
| CVXPY | 1.9.2 | （requirements 约束内） |
| openpyxl / python-docx / matplotlib | 3.1.5 / 1.2.0 / 3.11.1 | — |

依赖按 `requirements.txt` 安装（另加 matplotlib，画图需要）。版本均在 requirements 允许范围内，但与官方复核环境不同，因此 Q2 出现 1e-6 量级的数值差异（见 §3）。

## 2. 运行命令与耗时

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall -q .                                   # 通过
python q1_strict_occlusion.py                               # 1.238 s
python q2_optimize.py                                       # 73.294 s（默认 --g 9.8 --seeds 41,137,809）
python tools\check_excel_consistency.py                     # 12 项检查全部通过
python tools\make_figures.py                                # 生成 5 张图
python q5_optimize.py --multistart-active-blocks `
  --activation-output q5_block_attacked_plan_v2.json `
  --multistart-output q5_multistart_plan_v2.json `
  --seeds 41,137,353                                        # 约 50 分钟
```

## 3. Q1-Q5 复现结果与差异

| 问题 | 正式值 | 本次复现值 | 差异 | 结论 |
|---|---:|---:|---:|---|
| Q1 | 1.391642668 s | 1.391642668 s | 0 | 完全一致 |
| Q2 | 4.588048258 s | 4.588047213 s | -1.045e-6 s | 在证书区间 [4.586009979, 4.588737488] 内 |
| Q3 | 7.650405706 s | result1.xlsx 求和 7.650405706 s | 0 | 完全一致 |
| Q4 | 11.735130825 s | result2.xlsx 求和 11.735130000 s | -8.25e-7 s | Excel 存 6 位小数，舍入误差内 |
| Q5 J_sum | 35.109171990 s | dense 验证 35.10917198998913 s | <1e-9 | 完全一致 |
| Q5 J_min | 7.499774830 s | dense 验证 7.499774830203764 s | <1e-9 | 完全一致 |
| Q5 J_all | 1.439349 s | dense 验证 1.4393491191871863 s | 1.19e-7 | 一致 |

Q2 差异原因：优化器与证书回代使用 SciPy 1.18.1 / NumPy 2.5.2，与官方基线 SciPy 1.13.1 / NumPy 2.0.2 的数值收敛点存在 1e-6 级差异；`C_vs_B_contradictions = 0`，且正式值与本次复现值都落在本次证书区间内。**差异在认证区间内，不需要暂停合并，但版本差异已记录在案。**

## 4. 一致性自动检查

`tools/check_excel_consistency.py` 覆盖：

- result1.xlsx 求和 = 7.650405706 s（Q3 正式值）；
- result2.xlsx 求和 = 11.735130000 s（Q4 正式值 11.735130825 的 6 位舍入）；
- result3.xlsx 11 枚弹与 `q5_bomb_table.json` 逐弹一致（坐标、时长、目标导弹）；
- 每行投放点→起爆点的运动学一致（航向、速度、½gt² 下落）；
- Q5 起爆高度 ≥ 0、覆盖区间不超出起爆后 20 s 有效期；
- Q5 `J_sum / J_min / J_all` 与 dense 验证一致；三位小数回代（rounded_decision_metrics）可行；
- 证书文件数值：Q4 `current_total=11.735130825`、上界 `strict_upper_total=11.839`；Q5 夹逼 `[35.067366178, 35.112385634]`。

结果：12 项检查，0 失败。

## 5. Q5 九次多起点逐次日志

现状缺口已补齐：`q5_multistart_plan_v2.json`（新文件，未覆盖旧记录）。

- 种子：41, 137, 353；3 架无人机（FY3/FY4/FY5）× 3 种子 = 9 次逐次记录；
- 每次记录：`uav / seed / before_total / candidate_total / improvement / optimizer_success / optimizer_evaluations`，胜出起点标注 `selected/accepted`；
- 结果：9 次候选均被拒绝（improvement < 0），最优候选 FY3/seed=41 为 35.057752168（低于基线 35.131241510）；
- 方案未改变，最终指标与源方案一致（total = 35.11636939216913 @ dt=0.01 验收网格）。

配套代码改动：`q5_optimize.py` 的 `multistart_active_blocks` 增加逐起点日志（仅日志，不改数学逻辑与接受判定），`log_version=2`。

## 6. 图表

`figures/` 下 5 张论文可用图（`tools/make_figures.py` 生成，数据全部来自保存的 JSON/Excel）：

- `fig_q3_timeline.png`：Q3 三弹时间轴（累计 7.650405706 s）；
- `fig_q4_timeline.png`：Q4 三机时间轴（累计 11.735130825 s）；
- `fig_q5_timeline.png`：Q5 三导弹覆盖时间轴 + 同时遮蔽阴影（J_sum/J_min/J_all 标注）；
- `fig_q5_joint_additions.png`：Q5 认证联合新增区间（多球联合覆盖增量）；
- `fig_q5_tradeoff.png`：Q5 候选 J_sum vs J_min / J_all 权衡，标出最终 11 弹方案。

## 7. 正式文件完整性

`deliverables/` 三份 Excel 与交接文档未做任何修改；哈希与 `SHA256SUMS.txt` 及基线快照一致。本次新增文件：

- `docs/baseline_snapshot_20260827.md`
- `tools/check_excel_consistency.py`
- `tools/make_figures.py`
- `figures/*.png`（5 张）
- `q5_multistart_plan_v2.json`
- `repro_report.md`（本文件）

## 8. 结论与建议

- Q1-Q5 全部可复现，差异均在认证或舍入区间内；Q2 差异已定位为环境版本差异并记录原因。
- 建议数院队友按建模线审查 Q2 版本敏感性是否需要在论文方法部分注明“复现环境版本”。
- 下一步：将本分支提交并开 PR；Q5 多起点如需更多种子覆盖，可在 PR 合并后以 `--seeds` 参数扩展运行。

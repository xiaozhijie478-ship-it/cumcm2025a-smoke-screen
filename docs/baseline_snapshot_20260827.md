# Coding 基线快照（安泰 / 2026-08-27）

## 仓库状态

- 本地路径：`C:\Users\Conti\Desktop\code\cumcm2025a-smoke-screen`
- 分支：`main`（工作区干净，与 `origin/main` 同步）
- 基线提交：`2cb7633`（docs: add national-first-prize acceptance checklist）
- 远程：`https://github.com/xiaozhijie478-ship-it/cumcm2025a-smoke-screen`（私有）

## 环境

- Python：3.12.10（`D:\python3.12.10\python.exe`），虚拟环境 `.venv`
- numpy 2.5.2 / scipy 1.18.1 / cvxpy 1.9.2 / openpyxl 3.1.5 / python-docx 1.2.0 / matplotlib 3.11.1
- 说明：基线复核环境为 Python 3.9.13 + NumPy 2.0.2 + SciPy 1.13.1；本快照使用 3.12 满足 `requirements.txt` 约束的较新版本，正式结果复核前需记录版本差异。

## 快速检查结果（2026-08-27）

- `python -m compileall -q .`：通过（exit 0）
- 导入检查：numpy / scipy / cvxpy / openpyxl / python-docx / matplotlib 全部通过
- Q1 基线：`c_duration = 1.391642668 s`，`C_vs_B_contradictions = 0`，与正式值一致
- Q2–Q5 `--help`：全部可运行

## 正式输出基线哈希

以下与 `deliverables/SHA256SUMS.txt` 一致：

- result1.xlsx：`FC180E69...65BAF26`
- result2.xlsx：`C83D9F3C...9FF9659`
- result3.xlsx：`9A218800...40FD6`

任何重新生成 Excel 前必须重新比对此基线；差异超出认证区间时暂停合并。

## 准备完成确认

- git 身份已配置：`Gabrielle <wendyodom93322@gmail.com>`（2026-08-27）。

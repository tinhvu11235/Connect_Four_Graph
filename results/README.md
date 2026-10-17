# results/ — bằng chứng hiệu năng

Mọi con số hiệu năng trong README chính đều lấy từ các file ở đây. File được commit lên git; checkpoint
(`.pt`) thì không — chúng nằm trong `runs/` (bị `.gitignore`).

Máy đo: NVIDIA GeForce RTX 5060, Windows 10, torch 2.11.0+cu128, Python 3.12.10.

| Thư mục | Đo gì | Sinh bởi | File |
|---|---|---|---|
| `benchmark/` | Tham số, thời gian inference (batch 256), thời gian 1 bước train (batch 1024), GPU đỉnh — cho từng mạng | `python -m experiments.benchmark_networks` | `benchmark.json` (số + môi trường đo), `benchmark.log` (output console) |
| `trials/<mạng>/` | Train thật 3 vòng × 256 ván, cài đặt của run 90k ván của repo, đánh giá 100 ván vs Fujita mỗi vòng | `python -m experiments.run_trials --suite trials` | `command.txt`, `console.log`, `config.json`, `train_metrics.csv`, `resources.json` |
| `peak_ram/<mạng>/` | RAM tiến trình đỉnh trong 1 vòng (flag mặc định của repo, đánh giá 20 ván) | `python -m experiments.run_trials --suite peak_ram` | như trên |
| `estimate/` | Bảng so sánh và ước tính thời gian train 90k ván | `python -m experiments.estimate_full_run` (đọc các thư mục trên + log 352 vòng của repo) | `full_run_estimate.md` |

`<mạng>` là `baseline_cnn`, `method1`, `method2`.

## Đọc file

* `train_metrics.csv` — một dòng mỗi vòng, do repo AlphaZero ghi. Cột hay dùng:
  `selfplay_seconds`, `training_seconds`, `eval_seconds`, `iteration_seconds`, `optimizer_steps`,
  `policy_loss`, `value_loss`, `eval_score` (điểm vs Fujita, thắng = 1, hoà = 0,5).
* `resources.json` — do `common/repo_bridge.py` ghi sau khi train: `parameters`, `wall_seconds`,
  `peak_ram_mb` (peak working set), `peak_gpu_allocated_mb`, `peak_gpu_reserved_mb`.
* `config.json` — toàn bộ cấu hình train (`TrainingConfig` của repo) + `model_config`.
* `command.txt` — lệnh đã chạy (bỏ `python`). Với các bản đo hiện có, lệnh được dựng lại từ
  `config.json` vì chúng được chạy trước khi script tự ghi file này (dòng đầu ghi rõ).

## Ghi chú về các bản đo hiện có

* `trials/` và `peak_ram/` được chạy ngày 2026-10-16, trước khi đổi tên: `model_config.architecture`
  ghi tên cũ `gnn_dynamic_v1` (= phương pháp 1) và `gnn_grid_v1` (= phương pháp 2); `run_dir` trong
  `config.json` là đường dẫn cũ (`runs/trial_*`, `runs/ram_*`). Số liệu không đổi.
* `benchmark/` được chạy lại ngày 2026-10-17 sau khi đổi tên; khác bản đo 2026-10-16 trong phạm vi nhiễu
  (trừ inference phương pháp 1: 8,9 ms so với 11,8 ms).
* `peak_ram/`: GPU đỉnh ở đây thấp (150–190 MB) vì 1 vòng chưa đủ replay (10.000 thế cờ) để bắt đầu train;
  GPU đỉnh khi train lấy ở `trials/`.

## Kiểm tra lại

```bash
python -m experiments.benchmark_networks
```

```bash
python -m experiments.run_trials --suite trials --iterations 3
```

```bash
python -m experiments.run_trials --suite peak_ram
```

```bash
python -m experiments.estimate_full_run
```

Chạy lại sẽ ghi đè file trong thư mục tương ứng; dùng `git diff results/` để so với bản đo cũ.
Bộ `trials` mất khoảng 40 phút cho cả ba mạng, `peak_ram` khoảng 10 phút, benchmark khoảng 1 phút.

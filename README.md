# Connect4GNN

Biểu diễn bàn cờ Connect 4 bằng đồ thị và train mạng GNN bằng pipeline AlphaZero của
[AlphaZeroConnect4Train](#cài-đặt) (repo đó chỉ được import để dùng, không bị sửa).
So sánh với baseline là mạng ResNet (CNN) của chính repo đó.

| | Phương pháp 1 · Đồ thị động | Phương pháp 2 · Lưới cố định |
|---|---|---|
| Nút | Quân đã đặt + 7 ô sắp rơi vào (frontier); các ô trống khác bị bỏ | Cả 42 ô, kể cả ô trống |
| Cạnh | 8 hướng, nối xa tối đa 3 ô qua ô trống (line-of-sight), cạnh ghi số ô trống ở giữa | 8 hướng, chỉ hàng xóm sát bên |
| Kích thước | 160 kênh × 6 block, 8,17 M tham số | 256 kênh × 6 block, 8,05 M tham số |
| Code | [`method1/`](method1) | [`method2/`](method2) |

Hai mạng dùng chung lớp message passing trong [`common/graph_layers.py`](common/graph_layers.py)
(R-GCN, tính bằng tích chập 3×3 giãn nở; test kiểm tra nó cho đúng số như dạng gather).
Đầu vào/đầu ra giống hệt CNN của repo: bàn `(N, 1, 7, 6)` → policy `(N, 7)` + value `(N,)`.

## Cấu trúc thư mục

| Thư mục / file | Nội dung |
|---|---|
| [`common/`](common) | Dùng chung: luật chơi (`connect4_rules.py`), lớp GNN (`graph_layers.py`), widget bàn cờ (`board_canvas.py`), cầu nối sang repo AlphaZero (`repo_bridge.py`) |
| [`baseline_cnn/`](baseline_cnn) | Baseline: train ResNet của repo qua cùng cầu nối, để đo thời gian cùng một cách |
| [`method1/`](method1) | Phương pháp 1: `board_graph.py` (đồ thị quân cờ dạng tường minh), `visualize_graph.py` (GUI bàn cờ → đồ thị → bàn cờ), `model.py` (`Method1Net`), `train.py` |
| [`method2/`](method2) | Phương pháp 2: `model.py` (`Method2Net`), `train.py` |
| [`demo/`](demo) | GUI so sánh 3 mạng trên cùng thế cờ: đồ thị, embedding, saliency, policy, value |
| [`experiments/`](experiments) | Đo hiệu năng: benchmark, train thử, ước tính 90k ván |
| [`results/`](results) | **Bằng chứng hiệu năng** (log, JSON, CSV), đưa lên git — xem [`results/README.md`](results/README.md) |
| `runs/` | Thư mục train (checkpoint `.pt`), **không** đưa lên git |
| [`slides/`](slides) | Slide giải thích cách biểu diễn (`build_deck.py` sinh ra file `.pptx`) |

Mỗi test nằm cạnh code nó kiểm tra (`*/tests/`). Mỗi hàm có docstring ghi nó làm gì và
ảnh hưởng tới đâu (dòng `Side effects:`).

## Cài đặt

Cần Python 3.12, PyTorch có CUDA (đã chạy với torch 2.11 + CUDA 12.8 trên RTX 5060) và repo
AlphaZeroConnect4Train **đặt cạnh** repo này:

```
<thư mục cha>/
    AlphaZeroConnect4Train/   # repo AlphaZero (chứa train_alphazero/ và checkpoint Fujita)
    Connect4GNN/              # repo này
```

Nếu để chỗ khác thì đặt biến môi trường `ALPHAZERO_REPO` trỏ tới nó.

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q
```

Toàn bộ 238 test chạy trên CPU trong khoảng 10 giây (gồm 2 vòng train thật, rất nhỏ, cho mỗi mạng).

## Chạy

```bash
python -m method1.visualize_graph
```

```bash
python -m demo.app_networks
```

Demo tự tìm checkpoint ở `runs/<mạng>_full/models/latest.pt`, rồi `runs/trials_<mạng>/models/latest.pt`;
không có thì hiện mạng chưa train (tiêu đề ghi rõ). Chỉ định checkpoint: `--baseline-cnn PATH --method1 PATH --method2 PATH`.
`python path/to/demo_networks.py` chạy demo từ thư mục bất kỳ.

Train đầy đủ 90k ván (352 vòng × 256 ván, cùng cài đặt với run CNN 90k ván của repo):

```bash
python -m method1.train --run-dir runs/method1_full --iterations 352 --replay-capacity 50000 --learning-rate 1e-4 --lr-warmup-steps 0 --weight-decay 0.01 --channels 160
```

```bash
python -m method2.train --run-dir runs/method2_full --iterations 352 --replay-capacity 50000 --learning-rate 1e-4 --lr-warmup-steps 0 --weight-decay 0.01 --channels 256
```

```bash
python -m baseline_cnn.train --run-dir runs/baseline_cnn_full --iterations 352 --replay-capacity 50000 --learning-rate 1e-4 --lr-warmup-steps 0 --weight-decay 0.01 --channels 256
```

Mọi flag giống `python -m train_alphazero` của repo. Chạy lại cùng lệnh sẽ resume từ
`<run-dir>/state.pt`. Thêm `--eval-interval 10` để chỉ đánh giá với Fujita mỗi 10 vòng (nhanh hơn
rất nhiều, xem bảng dưới). Mỗi run ghi `train_metrics.csv`, `config.json`, `resources.json`
(tham số, thời gian, RAM/GPU đỉnh) và checkpoint mỗi 10 vòng (~32 MB/file, ~1,2 GB cho cả run,
`state.pt` ~100 MB).

## Hiệu năng

Máy đo: NVIDIA GeForce RTX 5060, Windows 10, torch 2.11.0+cu128, Python 3.12.10.
Mọi con số dưới đây lấy từ file trong [`results/`](results); cách chạy lại nằm ở
[`results/README.md`](results/README.md).

### Benchmark từng mạng

Từ [`results/benchmark/`](results/benchmark) (`python -m experiments.benchmark_networks`).
Inference: batch 256 thế cờ (batch lá của self-play), bf16. Train: 1 bước AdamW, batch 1024.

| Mạng | Tham số | Inference @256 | Thế cờ/giây | Train step @1024 | GPU đỉnh khi train |
|---|---:|---:|---:|---:|---:|
| Baseline CNN 128×6 (mặc định của repo) | 2,14 M | 1,5 ms | ~167k | 17 ms | 483 MB |
| Baseline CNN 256×6 | 7,46 M | 4,5 ms | ~57k | 58 ms | 999 MB |
| Phương pháp 1 160×6, line-of-sight | 8,17 M | 8,9 ms | ~29k | 122 ms | 1511 MB |
| Phương pháp 1 256×6, không line-of-sight (ablation) | 8,05 M | 8,0 ms | ~32k | 99 ms | 1697 MB |
| Phương pháp 2 256×6 | 8,05 M | 6,6 ms | ~39k | 91 ms | 1640 MB |

### Train thử 3 vòng và ước tính cho 90k ván

Từ [`results/trials/`](results/trials), [`results/peak_ram/`](results/peak_ram) và
[`results/estimate/full_run_estimate.md`](results/estimate/full_run_estimate.md).
Mỗi mạng train thật 3 vòng × 256 ván (200 simulation/nước, đánh giá 100 ván với AlphaZero của Fujita mỗi vòng).

| | Baseline CNN 256×6 | Phương pháp 1 160×6 | Phương pháp 2 256×6 |
|---|---:|---:|---:|
| Self-play mỗi vòng (256 ván) | 72 s (×1,00) | 107 s (×1,48) | 90 s (×1,24) |
| 1 bước optimizer khi train | 67 ms (×1,00) | 135 ms (×2,02) | 98 ms (×1,46) |
| Đánh giá vs Fujita mỗi vòng (100 ván) | 133 s (×1,00) | 177 s (×1,33) | 135 s (×1,02) |
| GPU đỉnh (reserved) khi train | 1304 MB | 2298 MB | 2542 MB |
| RAM tiến trình đỉnh | 1645 MB | 1679 MB | 1668 MB |
| **90k ván, ước tính trực tiếp** | **22,6 h** | **33,0 h** | **25,7 h** |
| **90k ván, ước tính theo log thật của repo** | **33,2 h** | **47,6 h** | **37,4 h** |
| ... trong đó self-play / train / đánh giá | 11,8 / 2,3 / 19,1 h | 17,4 / 4,6 / 25,5 h | 14,5 / 3,4 / 19,4 h |
| 90k ván, đánh giá mỗi 10 vòng (trực tiếp / theo log) | 10,9 / 16,0 h | 17,4 / 24,6 h | 13,8 / 19,9 h |
| Điểm vs Fujita 3 vòng đầu | 0,08 · 0,10 · 0,10 | 0,07 · 0,09 · 0,10 | 0,10 · 0,09 · 0,10 |

Hai cách ước tính:

* **Trực tiếp** — một vòng "ổn định" của bản chạy thử (self-play + 391 bước optimizer khi replay
  50k đã đầy + đánh giá) × 352 vòng. Đánh giá thấp vì ván cờ dài ra và chậm dần khi mạng mạnh lên.
* **Theo log thật** — lấy log 352 vòng (90.112 ván, 33,2 h) của CNN 256×6 trong repo
  (`results/train_only_fujita`), nhân từng phần (self-play / train / đánh giá) với tỉ lệ
  chậm hơn đo được ở bản chạy thử. Đây là con số nên dùng để lên kế hoạch.

**Dự kiến:** train đủ 90k ván mất khoảng **37 h cho phương pháp 2** và **48 h cho phương pháp 1**
(baseline CNN: 33 h) trên một RTX 5060; nếu chỉ đánh giá mỗi 10 vòng thì còn khoảng **20 h** và
**25 h**. Hơn một nửa thời gian là đánh giá với Fujita, không phải train. Bộ nhớ không phải giới
hạn: GPU đỉnh ≤ 2,6 GB, RAM ≤ 1,7 GB. Sau 3 vòng, cả ba mạng đều ~0,10 điểm vs Fujita — chỉ chứng
tỏ pipeline học được, chưa nói gì về sức mạnh cuối (CNN của repo đạt 0,49 sau 90k ván).

## Ghi chú kỹ thuật

* Tên kiến trúc lưu trong checkpoint là `method1_v1` / `method2_v1`. Checkpoint cũ ghi
  `gnn_dynamic_v1` / `gnn_grid_v1` (ví dụ các file trong `results/trials/` và `runs/`) vẫn mở được.
* `common/repo_bridge.py` thay hàm `build_policy_value_model` của repo AlphaZero **trong bộ nhớ**
  để repo dựng được mạng của mình; self-play, MCTS, replay, loss, checkpoint, đánh giá đều là code
  của repo, không đổi. Repo không bị ghi `__pycache__`.
* Quy ước bàn cờ: `state[column, row]`, hàng 0 ở dưới; nút `i` là ô `column * 6 + row`.

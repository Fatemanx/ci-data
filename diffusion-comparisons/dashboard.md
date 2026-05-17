# Diffusion Cross-Framework Performance Dashboard

*Generated: May 17 | Commit: `229cade`*

## Cross-Framework Performance Comparison

| Model | Risk | sglang (s) |
|-------|------|---------|
| FLUX.1-dev | ✅ | **6.67** |
| FLUX.2-dev | ✅ | **22.78** |
| Qwen-Image-2512 | ✅ | **12.81** |
| Qwen-Image-Edit-2511 | ✅ | **23.75** |
| Z-Image-Turbo | ✅ | **0.89** |
| Wan2.2-T2V-A14B-Diffusers | ✅ | **216.27** |
| Wan2.2-TI2V-5B-Diffusers | ✅ | **59.19** |
| LTX-2 | ✅ | **8.62** |
| LTX-2.3 | ✅ | **19.07** |
| Wan2.2-I2V-A14B-Diffusers | ✅ | **212.61** |

### Latency Trend: flux1_dev_t2i_1024

![Latency Trend flux1_dev_t2i_1024](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_flux1_dev_t2i_1024.png)


### Latency Trend: flux2_dev_t2i_1024

![Latency Trend flux2_dev_t2i_1024](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_flux2_dev_t2i_1024.png)


### Latency Trend: qwen_image_2512_t2i_1024

![Latency Trend qwen_image_2512_t2i_1024](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_qwen_image_2512_t2i_1024.png)


### Latency Trend: qwen_image_edit_2511

![Latency Trend qwen_image_edit_2511](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_qwen_image_edit_2511.png)


### Latency Trend: zimage_turbo_t2i_1024

![Latency Trend zimage_turbo_t2i_1024](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_zimage_turbo_t2i_1024.png)


### Latency Trend: wan22_t2v_a14b_720p

![Latency Trend wan22_t2v_a14b_720p](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_wan22_t2v_a14b_720p.png)


### Latency Trend: wan22_ti2v_5b_720p

![Latency Trend wan22_ti2v_5b_720p](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_wan22_ti2v_5b_720p.png)


### Latency Trend: ltx2_twostage_t2v

![Latency Trend ltx2_twostage_t2v](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_ltx2_twostage_t2v.png)


### Latency Trend: ltx2.3_twostage_ti2v_2gpus

![Latency Trend ltx2.3_twostage_ti2v_2gpus](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_ltx2.3_twostage_ti2v_2gpus.png)


### Latency Trend: wan22_i2v_a14b_720p

![Latency Trend wan22_i2v_a14b_720p](https://raw.githubusercontent.com/sgl-project/ci-data/main/diffusion-comparisons/charts/latency_wan22_i2v_a14b_720p.png)


## SGLang Performance Trend (Last 15 Runs)

| Date | Commit | flux1_dev_t2i_1024 (s) | flux2_dev_t2i_1024 (s) | qwen_image_2512_t2i_1024 (s) | qwen_image_edit_2511 (s) | zimage_turbo_t2i_1024 (s) | wan22_t2v_a14b_720p (s) | wan22_ti2v_5b_720p (s) | ltx2_twostage_t2v (s) | ltx2.3_twostage_ti2v_2gpus (s) | wan22_i2v_a14b_720p (s) | Trend |
|------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|
| May 17 | `229cade` | 6.67 | 22.78 | 12.81 | 23.75 | 0.89 | 216.27 | 59.19 | 8.62 | 19.07 | 212.61 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| May 16 | `99fc29d` | 6.67 | 22.77 | 12.90 | 24.05 | 0.94 | 215.93 | 59.17 | 8.49 | N/A | 211.62 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:    :left_right_arrow: |
| May 08 | `3c3f0bd` | 6.60 | 22.60 | 12.72 | 23.85 | 0.93 | 211.58 | 59.20 | N/A | N/A | 207.62 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow:    :left_right_arrow: |
| May 06 | `b91b05a` | 6.68 | 22.79 | 12.90 | 24.08 | 0.93 | N/A | 59.21 | 11.00 | 40.14 | 208.67 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow:  :arrow_down:  :left_right_arrow:  :left_right_arrow: |
| May 04 | `c611a3f` | 6.63 | 22.62 | 13.07 | 23.91 | 0.91 | 210.23 | 59.17 | 11.72 | 40.12 | 205.57 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :arrow_down:  :arrow_down: |
| May 03 | `d44eb43` | 6.69 | 22.92 | 13.22 | 24.12 | 0.89 | 210.75 | 60.17 | 12.09 | 53.14 | 226.73 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:   :arrow_up: |
| Apr 30 | `3553fd0` | 6.74 | 22.93 | 13.25 | 24.03 | 0.90 | 212.47 | 62.18 | 11.99 | N/A | 207.63 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 29 | `14b4e6f` | 6.69 | 22.78 | 13.16 | 23.84 | 0.89 | 210.95 | 62.20 | 11.88 | N/A | 207.67 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 28 | `4a04a98` | 6.69 | 22.75 | 13.28 | 23.89 | 0.89 | 211.33 | 61.19 | 11.84 | N/A | 208.57 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :arrow_down:  :left_right_arrow:   :left_right_arrow: |
| Apr 27 | `977830e` | 6.72 | 22.91 | 13.24 | 23.91 | 0.89 | 212.10 | 63.20 | 11.80 | N/A | 208.53 | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:   :left_right_arrow: |
| Apr 26 | `7141735` | 6.71 | 22.84 | 13.24 | 23.89 | 0.89 | 212.74 | 62.22 | 11.86 | N/A | 208.55 |          |
| Apr 25 | `a4facdf` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |          |
| Apr 24 | `c016635` | 6.71 | 22.81 | 13.77 | 23.83 | 0.89 | 211.38 | 62.22 | 16.09 | 69.23 | 209.67 | :left_right_arrow:  :left_right_arrow:  :arrow_up:  :left_right_arrow:  :left_right_arrow:    :arrow_down:  :arrow_up:  |
| Apr 23 | `c689f77` | 6.60 | 22.62 | 13.03 | 23.68 | 0.88 | N/A | N/A | 16.75 | 58.19 | N/A | :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:  :left_right_arrow:    :arrow_up:  :arrow_down:  |
| Apr 18 | `9c47bba` | 6.68 | 22.82 | 13.13 | 23.78 | 0.89 | 210.45 | 62.17 | 15.32 | 60.19 | 206.66 | -- |

---
*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*

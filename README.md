## Best so far:
```bash
uv run -m src.main --method hough --transforms abs clip normalize resize nlm median clip normalize --output temp.jpg --start-time '09:00:17' --end-time '09:03:32' --group-lines
```

With config.yaml:
```yaml
paths:
  data_dir: 'data/'

metadata:
  dx: 5.106500953873407
  dt: 0.0016
  file_duration: 10
  samples_per_file: 6250
  date_start: '2024-05-07'

transforms:
  zscore:
    window: 11
  median: 
    kernel_size: 3
  tv:
    weight: 0.05
  resize:
    width: 500
    height: 500
  clip:
    first_percentile: 1
    last_percentile: 99
  nlm:
    h: 20
    template_window_size: 7
    search_window_size: 35

methods:
  hough:
    threshold1: 50
    threshold2: 200
    rho: 1
    theta: 0.01
    hough_threshold: 120

group_lines:
  min_cluster_size: 3
```
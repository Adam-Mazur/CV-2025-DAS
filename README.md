## To run:
```bash
uv run -m src.main --methods hough hough_prob --transforms abs clip normalize resize nlm median clip normalize --output output.jpg --start-time '09:22:22' --end-time '09:24:12' --group-lines hough_prob=true,hough=true,all=true --cluster --polynomial
```

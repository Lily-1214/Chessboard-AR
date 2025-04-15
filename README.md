# Chessboard-AR



## 원기둥 생성 함수

```python
# 원기둥 AR 생성 함수
def create_cylinder(center, radius, height, segments=36):
    angle_step = 2 * np.pi / segments
    top_circle = [
        [center[0] + radius * np.cos(i * angle_step),
         center[1] + radius * np.sin(i * angle_step),
         center[2] - height] for i in range(segments)
    ]
    bottom_circle = [
        [center[0] + radius * np.cos(i * angle_step),
         center[1] + radius * np.sin(i * angle_step),
         center[2]] for i in range(segments)
    ]
    return np.array(top_circle + bottom_circle, dtype=np.float32).reshape(-1, 1, 3)
```



## 기둥 생성 영상(gif)

![Image](https://github.com/user-attachments/assets/69e3ecfc-ced5-4f0a-a9ea-bc965adf4df6)

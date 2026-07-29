# 纯几何碎片拼矩形求解器

`algorithm.py` 用于将观测到的 2～4 个多边形碎片重新拼成矩形。算法通过
边长匹配、刚体变换、重叠排除和矩形面积覆盖率进行求解，不使用扑克牌纹理
或其他图案信息。

调用求解器之前，必须先将摄像头像素坐标转换为经过透视矫正的毫米坐标。
每块碎片使用以下格式表示：

```python
{"pts": [Pt(x0, y0), Pt(x1, y1), Pt(x2, y2)]}
```

轮廓顶点必须按照边界顺序排列，顺时针和逆时针均可。每块碎片可以有
3～5 条边。

## 调用示例

```python
from algorithm import Pt, find_rectangle_solution

solution = find_rectangle_solution(
    pieces,
    target_center=Pt(105.0, 220.0),
)

if solution is not None:
    for placement in solution["placements"]:
        print(
            placement["piece_index"],
            placement["source_center"],
            placement["target_center"],
            placement["angle"],
        )
```

`angle` 的单位为弧度，表示碎片从当前观测方向旋转到目标位姿所需的角度。
如果返回 `None`，表示在当前几何容差和搜索预算内没有找到有效矩形，此时
机械臂不应执行抓取或放置动作。

## 默认约束

- 矩形短边范围：50～90 mm
- 矩形长边范围：90～120 mm
- 边匹配容差：3 mm 或边长的 8%，取较大值
- 矩形未覆盖面积容差：6%
- 搜索预算：最多检查 5000 个局部布局，防止异常图像导致程序长时间卡住

当前测试包含 30 个精确合成拼图，以及 10 个顶点分别带有 ±1 mm 随机误差
的拼图。实际摄像头处理流程应为：

`透视矫正 → 像素转毫米 → 轮廓提取 → 多边形角点简化 → 调用矩形求解器`

## 运行测试

执行以下命令运行合成拼图测试：

```powershell
python -m unittest -v test_algorithm.py
```

## 生成可视化结果

执行以下命令生成“拼接前/拼接后”中文对照图：

```powershell
python visual_demo.py --seed 29 --noise 1.0 --output rectangle_demo.png
```

## 树莓派摄像头程序

`camera_pipeline.py` 是树莓派实际运行入口，默认采集分辨率固定为
`1600×1200`，支持以下图像源：

- 树莓派 CSI 摄像头：Picamera2；
- USB 摄像头：OpenCV `VideoCapture`；
- 已保存的静态照片：用于离线调试。

完整处理流程为：

`1600×1200 图像 → 四点透视矫正 → 二值分割 → 轮廓提取 → 多边形拟合 → 毫米坐标 → 矩形求解`

### 1. 安装树莓派依赖

建议使用 Raspberry Pi OS Bookworm 的系统 Python：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-pil python3-picamera2 fonts-noto-cjk
```

使用 CSI 摄像头前，可先确认摄像头能够正常采集：

```bash
rpicam-hello
```

### 2. 准备工作区

默认配置文件 [vision_config.json](vision_config.json) 将工作区设置为横向 A4：

- 宽度：297 mm；
- 高度：210 mm；
- 拼接目标中心：`(148.5, 105.0) mm`；
- 推荐背景：哑光黑色或深蓝色；
- 推荐碎片：比背景明显更亮。

如果使用的工作区不是 A4，需要先修改配置中的 `width_mm`、`height_mm`
和 `target_center_mm`。

### 3. 完成四点透视标定

固定好摄像头后运行：

```bash
cd ~/2026_E/Vision
python3 camera_pipeline.py --source picamera2 --calibrate
```

在窗口中依次点击：

1. 工作区左上角；
2. 工作区右上角；
3. 工作区右下角；
4. 工作区左下角。

按 `R` 可重新选择，四点完成后按回车保存。只要摄像头位置发生变化，就必须
重新标定。

### 4. 处理单帧图像

CSI 摄像头：

```bash
python3 camera_pipeline.py --source picamera2 --display
```

USB 摄像头：

```bash
python3 camera_pipeline.py --source usb --device 0 --display
```

离线照片：

```bash
python3 camera_pipeline.py --image test.jpg --display
```

### 5. 连续运行

带显示窗口运行：

```bash
python3 camera_pipeline.py --source picamera2 --continuous --display
```

无显示器运行：

```bash
python3 camera_pipeline.py --source picamera2 --continuous
```

按 `Q` 或 `Esc` 退出显示窗口。无显示器连续运行时可使用 `Ctrl+C` 停止。

### 6. 输出文件

程序持续更新 `Vision/output` 中的以下文件：

| 文件 | 内容 |
| --- | --- |
| `latest_corrected.jpg` | 透视矫正后的工作区图像 |
| `latest_mask.png` | 碎片二值分割结果 |
| `latest_result.jpg` | 检测前和目标布局对照图 |
| `latest_result.json` | 碎片轮廓、目标中心和旋转角 |

只有当 JSON 中的 `status` 为 `ok` 时，机械臂控制程序才允许读取并执行
`solution.placements`。其他状态都必须保持机械臂停止。

### 7. 常用分割参数

这些参数位于 `vision_config.json`：

| 参数 | 含义 |
| --- | --- |
| `mode` | `light_on_dark` 表示亮碎片、暗背景 |
| `threshold` | `0` 表示使用 Otsu 自动阈值，也可以填写固定灰度阈值 |
| `morphology_mm` | 开闭运算尺度，用于清理小噪点和小孔洞 |
| `polygon_epsilon_mm` | 多边形角点拟合容差 |
| `min_piece_area_mm2` | 最小碎片面积，小于该值的轮廓会被忽略 |
| `max_piece_area_mm2` | 最大碎片面积，大于该值的轮廓会被忽略 |

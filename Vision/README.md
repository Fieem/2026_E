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

- 矩形短边范围：40～100 mm
- 矩形长边范围：80～130 mm
- 边匹配容差：3 mm 或边长的 8%，取较大值
- 矩形未覆盖面积容差：6%
- 边长相对容差：5%
- 局部接缝最小接触比例：60%
- 相邻碎片重叠容差：4 mm
- 搜索预算：最多检查 20000 个局部布局，防止异常图像导致程序长时间卡住

求解器会先用多边形包围盒排除明显分离的候选，再执行精确边相交判断；
碎片边长也会在预处理阶段缓存，减少重复计算。

当前测试包含 30 个精确合成拼图，以及 10 个顶点分别带有 ±1 mm 随机误差
的拼图。实际摄像头处理流程应为：

`透视矫正 → 像素转毫米 → 轮廓提取 → 多边形角点简化 → 调用矩形求解器`

## 运行测试

执行以下命令运行合成拼图测试：

```powershell
python -m unittest -v test_algorithm.py
```

## 生成可视化结果

执行以下命令生成“拼接前/拼接后”对照图。为避免树莓派缺少中文字体，
图片内的标题和状态使用英文，代码注释、终端信息和 JSON 仍使用中文：

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

当前支持两种碎片模式，通过 [vision_config.json](vision_config.json) 里的
`piece_mode` 切换：

- `plain`：纯色碎片模式，只走几何拼接流程；
- `playing_cards`：扑克牌碎片模式。使用深蓝背景颜色距离分割，
  将白底、红蓝色块和黑色牌面统一视为碎片，再进行几何求解、纹理评分，
  并可选地启用 `jqk_template` 做 J/Q/K 模板重排。

如果只是做当前纯色拼图，建议保持：

```json
"piece_mode": "plain"
```

只有在你要开始调试扑克牌纹理拼接时，再切到：

```json
"piece_mode": "playing_cards"
```

并同时把 `texture.enabled` 设为 `true`。

如果你正在专门调试人物牌 J/Q/K，可以再额外开启：

```json
"jqk_template": {
  "enabled": true
}
```

模板文件默认放在 [templates/jqk](templates/jqk/README.md) 中，命名格式为：

- `spade_J.png`
- `heart_Q.png`
- `club_K.png`

这条链路当前只用于“在多个几何候选里选出更像 J/Q/K 的拼法”，不会额外输出
具体是哪一张牌。

## 树莓派—Gimbal1 串口服务

`vision_serial_service.py` 等待 Gimbal1 通过 USART1 发送 `VISION_START`，
收到后只采集并处理一帧图像。识别成功后，每块碎片发送一个结果帧，结果包含：

```text
pick_j1_rad, place_j1_rad,
pick_j2_rad, place_j2_rad,
pick_wrist_rad, place_wrist_rad
```

其中腕部角度用于后续舵机/电磁铁方向控制。当前阶段 Gimbal1 只缓存该角度，
不会驱动尚未接入的 Gimbal2。

树莓派部署依赖：

```bash
sudo apt install -y python3-serial
```

启动串口服务：

```bash
python3 vision_serial_service.py --serial /dev/serial0 --source picamera2
```

当前 `vision_config.json` 中的 SCARA 连杆长度默认为 `0`，这是安全占位值。
必须填写 `link1_mm`、`link2_mm`、相机到基座变换、零位和关节限位后，串口服务
才会发送角度结果；参数未配置时只返回错误码，不会发送运动角度。

### 1. 安装树莓派依赖

建议使用 Raspberry Pi OS Bookworm 的系统 Python：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-pil python3-picamera2
```

使用 CSI 摄像头前，可先确认摄像头能够正常采集：

```bash
rpicam-hello
```

### 2. 准备工作区

默认配置文件 [vision_config.json](vision_config.json) 将工作区设置为竖向 A4：

- 宽度：210 mm；
- 高度：297 mm；
- 拼接目标中心：`(105.0, 148.5) mm`；
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

### 5. 按键触发运行（推荐）

平时只显示摄像头预览，不执行轮廓提取和矩形求解。按下空格键后，程序会
抓取当前画面并检测一次，完成后重新回到待机状态：

```bash
python3 camera_pipeline.py --source picamera2 --triggered --display
```

- 空格键：检测当前画面一次；
- `Q` 或 `Esc`：退出程序；
- 每次检测结果都会覆盖更新到 `Vision/output`。

如果需要调节深蓝背景的分割效果，可以打开调试窗口：

```bash
python3 camera_pipeline.py --source picamera2 --debug --display
```

纯色模式调试窗口包含四个区域：

- `CORRECTED`：透视矫正后的工作区；
- `GRAY`：灰度图；
- `BINARY MASK`：实际送入轮廓提取的二值图；
- `CONTOURS`：当前二值图检测出的外轮廓。

扑克牌模式下，第二个区域会替换为 `BLUE BACKGROUND MODEL`，
第四个区域左侧会显示 `CARD FOREGROUND MASK`：

- `BLUE BACKGROUND MODEL`：白色表示被颜色模型判定为深蓝背景；
- `CARD FOREGROUND MASK`：白色表示最终保留的整块扑克牌。

按 `D` 可将当前四宫格保存到 `output/latest_color_model_debug.png`。

调试时拖动两个滑条：

- `Threshold (0=Otsu)`：阈值，0 表示自动阈值；
- `Morphology x0.1mm`：形态学开闭运算尺度。

确认二值图中碎片为白色、深蓝背景为黑色后，按空格使用当前滑条参数执行
一次矩形求解。按 `S` 可将当前参数直接保存到 `vision_config.json`。

如果树莓派没有连接显示器，可以使用终端触发模式：

```bash
python3 camera_pipeline.py --source picamera2 --triggered
```

在终端中按回车检测一次，输入 `Q` 后按回车退出。

### 6. 连续运行（仅用于调试）

带显示窗口运行：

```bash
python3 camera_pipeline.py --source picamera2 --continuous --display
```

无显示器运行：

```bash
python3 camera_pipeline.py --source picamera2 --continuous
```

按 `Q` 或 `Esc` 退出显示窗口。无显示器连续运行时可使用 `Ctrl+C` 停止。

### 7. 输出文件

程序持续更新 `Vision/output` 中的以下文件：

| 文件 | 内容 |
| --- | --- |
| `latest_corrected.jpg` | 透视矫正后的工作区图像 |
| `latest_mask.png` | 碎片二值分割结果 |
| `latest_result.jpg` | 检测前和目标布局对照图 |
| `latest_result.json` | 碎片轮廓、目标中心和旋转角 |
| `latest_jqk_card_preview.png` | 模板重排使用的标准化整牌图 |
| `latest_jqk_best_template.png` | 当前最佳 J/Q/K 模板 |
| `latest_jqk_mask_compare.png` | 拼接结果与模板的红/黑花纹对比图 |

只有当 JSON 中的 `status` 为 `ok` 时，机械臂控制程序才允许读取并执行
`solution.placements`。其他状态都必须保持机械臂停止。

JSON 中的坐标仍然属于相机工作区坐标，不能未经转换就直接作为机械臂坐标。
联动机械臂之前，还需要标定“相机工作区坐标 → SCARA 基坐标”的二维刚体或
仿射变换。

### 失败原因诊断

当矩形求解失败时，终端会输出：

- 首层接缝候选数量；
- 完整布局数量；
- 整边和局部边候选数量；
- 重叠、尺寸范围和面积误差排除数量；
- 每块碎片的边数、面积和边长。

常见判断方法：

- `首层候选 0`：优先检查四点标定、像素到毫米比例和轮廓角点；
- `完整布局 0`：轮廓形状、顶点顺序或接缝方向可能错误；
- `矩形尺寸` 排除较多：工作区比例或碎片尺寸识别可能不正确；
- `面积` 排除较多：碎片之间有缝隙、重叠，或轮廓包含了背景；
- `节点达到上限`：边长容差过大，或者拟合出了过多错误角点。

### 8. 常用分割参数

这些参数位于 `vision_config.json`：

| 参数 | 含义 |
| --- | --- |
| `mode` | `light_on_dark` 表示亮碎片、暗背景 |
| `threshold` | `0` 表示使用 Otsu 自动阈值，也可以填写固定灰度阈值 |
| `background_distance_threshold` | 扑克牌模式中，像素与深蓝背景估计值的颜色距离阈值，默认 `28` |
| `blue_hue_tolerance_deg` | 深蓝背景色相容差，默认 `28°` |
| `blue_background_min_saturation` | 判定为蓝色背景的最小饱和度 |
| `blue_background_value_margin` | 允许蓝色背景亮度相对参考值增加的范围 |
| `foreground_component_min_area_mm2` | 扑克牌模式中删除的小前景噪声面积阈值 |
| `morphology_mm` | 开闭运算尺度，用于清理小噪点和小孔洞 |
| `polygon_epsilon_mm` | 多边形角点拟合容差 |
| `playing_card_max_polygon_epsilon_mm` | 扑克牌模式的最大拟合容差，默认 `12`，用于消除印刷边缘造成的伪角点 |
| `rounded_corner_enabled` | 是否在扑克牌模式合并圆角产生的连续小边 |
| `rounded_corner_min_turn_deg` | 连续小转角的最小角度，默认 `6°` |
| `short_edge_mm` | 小于该长度的边视为伪短边，并用两侧边的延长线求角点 |
| `merge_angle_deg` | 相邻两条边的夹角大于该值时，视为同一条直边并合并，推荐先试 `165°～175°` |
| `max_corner_extension_mm` | 短边修复时允许角点延长的最大距离 |
| `min_piece_area_mm2` | 最小碎片面积，小于该值的轮廓会被忽略 |
| `max_piece_area_mm2` | 最大碎片面积，大于该值的轮廓会被忽略 |

扑克牌模式还会对明显超过 5 个角点的接触轮廓执行距离变换和分水岭拆分，
用于处理两块碎片边角相碰的情况。若调试图中已经得到 4 块碎片但仍然显示
“矩形求解失败”，问题就进入接缝匹配阶段，应继续查看 `latest_result.json`
中的 `diagnostics`，而不是继续调灰度阈值。

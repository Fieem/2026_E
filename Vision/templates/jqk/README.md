# J/Q/K 模板目录说明

将 12 张标准整牌模板按以下文件名放入本目录：

- `spade_J.jpg`
- `spade_Q.jpg`
- `spade_K.jpg`
- `heart_J.jpg`
- `heart_Q.jpg`
- `heart_K.jpg`
- `club_J.jpg`
- `club_Q.jpg`
- `club_K.jpg`
- `diamond_J.jpg`
- `diamond_Q.jpg`
- `diamond_K.jpg`

当前代码默认按 `.jpg` 查找，也兼容同名的 `.jpeg` 和 `.png`。

模板要求：

- 完整整牌正视图；
- 白底保留，背景不要带深蓝工作台；
- 长边方向固定；
- 建议直接裁成竖向整牌图；
- 图片越干净、越接近现场牌面印刷，模板匹配越稳定。

第一阶段代码只把模板用于：

- 对几何候选做二次重排；
- 判断哪种拼法更像 J/Q/K；

当前不会额外输出“这是哪一张牌”的识别结果。

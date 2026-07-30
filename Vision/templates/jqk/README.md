# J/Q/K 模板目录说明

将 12 张标准整牌模板按以下文件名放入本目录：

- `spade_J.png`
- `spade_Q.png`
- `spade_K.png`
- `heart_J.png`
- `heart_Q.png`
- `heart_K.png`
- `club_J.png`
- `club_Q.png`
- `club_K.png`
- `diamond_J.png`
- `diamond_Q.png`
- `diamond_K.png`

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

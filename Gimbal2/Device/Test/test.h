//
// VOFA+ FireWater 在线调参接口
//

#ifndef TEST_H
#define TEST_H

#include <stdint.h>


/* ---- VOFA 使用 USART3 ---- */
#define TEST_VOFA_HUART            (&huart3)
#define TEST_VOFA_FRAME_MAX_LEN    (64U)

/* ---- 环形接收缓冲区（ISR 写入, main loop 读取） ---- */
#define TEST_VOFA_RX_RING_SIZE     (128U)

/* ---- printsf 多行缓存 ---- */
#define PRINTSF_MAX_LINES      (8U)
#define PRINTSF_MAX_LINE_LEN   (256U)
#define PRINTSF_TEXT_BUF_LEN   (PRINTSF_MAX_LINES * (PRINTSF_MAX_LINE_LEN + 2U) + 1U)
// 初始化 VOFA+ 接收状态
void test_vofa_init(void);

// 轮询串口字节并解析 FireWater 命令
// 建议在高频循环中调用（例如 main 的 while(1)）
void test_vofa_poll(void);

// 向解析器输入一个字节（适用于中断/回调场景）
void test_vofa_feed_byte(uint8_t byte);

// 直接解析一整帧 FireWater 文本
// 示例: "P1=0.40,I1=0.02,D1=0.00!"
void test_vofa_parse_frame(const char *frame);

// Nextion 文本控件打印:
// 例: prints(0, "helloworld") -> t0.txt="helloworld" + 0xFF 0xFF 0xFF
void prints(uint8_t index, const char *content);

// 在 RTOS 创建任务前初始化串口屏输出互斥锁
void screen_output_init(void);

// 带格式化参数的 Nextion 文本控件打印:
// 例: printsf(0, "speed=%.2f", speed);
void printsf(uint8_t index, const char *fmt, ...);

void printsf_clear(uint8_t index);


void rx_ring_push(uint8_t byte);

extern uint8_t s_rx_byte;
#endif // TEST_H

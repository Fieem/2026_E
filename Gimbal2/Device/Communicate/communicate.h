//
// Created by Administrator on 2026/7/13.
// 树莓派通信 - USART1, 921600, PA9/PA10
// 文本协议: 逗号分隔，单行一条消息，\n 结尾
//

#ifndef INC_2024E_COMMUNICATE_H
#define INC_2024E_COMMUNICATE_H

#include "main.h"
#include "usart.h"
#include <stdint.h>
#include <stdbool.h>

/* ---- 缓冲区配置 ---- */
#define COMM_RX_RING_SIZE   256U
#define COMM_FRAME_MAX_LEN  128U

extern uint8_t comm_rx_byte;   /* HAL_UART_Receive_IT 单字节接收缓冲 */

/* ---- 初始化与轮询 ---- */
void comm_pi_init(void);
void comm_pi_poll(void);

/* ---- ISR 安全写入环形缓冲区 ---- */
void comm_pi_ring_push(uint8_t byte);

/* ---- 发送请求给树莓派 ---- */
void comm_send_place(char color, uint8_t num, uint8_t row, uint8_t col);
void comm_send_battle_start(char color);
void comm_send_ready(void);
void comm_send_new(void);
#endif //INC_2024E_COMMUNICATE_H

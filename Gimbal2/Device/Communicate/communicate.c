//
// Created by Administrator on 2026/7/13.
// 树莓派通信 - 文本协议解析与收发
// 协议: 逗号分隔，单行一条消息，\n 结尾
//

#include "communicate.h"
#include "Public/public.h"
#include "Test/test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================
 *  环形缓冲区（ISR 安全，单生产者单消费者）
 * ============================================================ */
static uint8_t           s_comm_rx_ring[COMM_RX_RING_SIZE];
static volatile uint16_t s_comm_rx_head = 0U;   /* ISR 写入 */
static uint16_t          s_comm_rx_tail = 0U;   /* 任务读取 */

/* ---- ISR 安全 push ---- */
void comm_pi_ring_push(uint8_t byte)
{
    uint16_t next = (uint16_t)(s_comm_rx_head + 1U) % COMM_RX_RING_SIZE;
    if (next != s_comm_rx_tail) {
        s_comm_rx_ring[s_comm_rx_head] = byte;
        s_comm_rx_head = next;
    }
    /* 缓冲区满 → 丢弃该字节 */
}

/* ---- 任务侧 pop，返回 0=空，1=取出 ---- */
static int comm_pi_ring_pop(uint8_t *out)
{
    if (s_comm_rx_tail == s_comm_rx_head) {
        return 0;
    }
    *out = s_comm_rx_ring[s_comm_rx_tail];
    s_comm_rx_tail = (uint16_t)(s_comm_rx_tail + 1U) % COMM_RX_RING_SIZE;
    return 1;
}

/* ============================================================
 *  行累积与解析
 * ============================================================ */
static char    s_comm_frame_buf[COMM_FRAME_MAX_LEN];
static uint8_t s_comm_frame_len = 0U;

/* ---- 忽略大小写字符串比较（仅全大写 ASCII 命令） ---- */
static int strcmp_upper(const char *a, const char *b_upper)
{
    while (*a && *b_upper) {
        char ca = *a;
        if (ca >= 'a' && ca <= 'z') ca -= 32;   /* 转大写 */
        if (ca != *b_upper) return 1;
        a++; b_upper++;
    }
    return (*a == '\0' && *b_upper == '\0') ? 0 : 1;
}

/* ---- 解析一条完整的 \n 终止行 ---- */
static void comm_pi_parse_line(const char *line)
{
    char work[COMM_FRAME_MAX_LEN];
    char *saveptr = NULL;
    strncpy(work, line, sizeof(work) - 1);
    work[sizeof(work) - 1] = '\0';

    char *cmd = strtok_r(work, ",", &saveptr);
    if (cmd == NULL) return;

    if (strcmp_upper(cmd, "PULSES") == 0) {
        /* PULSES,<pick_p1>,<pick_p2>,<place_p1>,<place_p2> */
        char *a1 = strtok_r(NULL, ",", &saveptr);
        char *a2 = strtok_r(NULL, ",", &saveptr);
        char *a3 = strtok_r(NULL, ",", &saveptr);
        char *a4 = strtok_r(NULL, ",", &saveptr);
        if (a1 && a2 && a3 && a4) {
            comm_pick_p1  = atoi(a1);
            comm_pick_p2  = atoi(a2);
            comm_place_p1 = atoi(a3);
            comm_place_p2 = atoi(a4);
            comm_response_ready = true;
            printsf(0, "PULSES OK %d %d %d %d", comm_pick_p1, comm_pick_p2, comm_place_p1, comm_place_p2);
        }
        else{printsf(0,"Pulse");}
    }
    else if (strcmp_upper(cmd, "ERROR") == 0) {
        /* ERROR,<code>,<message> */
        char *code = strtok_r(NULL, ",", &saveptr);
        char *msg  = strtok_r(NULL, ",", &saveptr);
        printsf(0, "ERR %s: %s", code ? code : "?", msg ? msg : "");
        comm_response_ready = false;
    }
    else if (strcmp_upper(cmd, "BUSY") == 0) {
        /* BUSY,<message> */
        char *msg = strtok_r(NULL, ",", &saveptr);
        printsf(0, "BUSY: %s", msg ? msg : "");
        comm_response_ready = false;
    }
    /* 未知命令 → 静默忽略 */
}

/* ---- 逐字节喂入，遇 \n 自动解析 ---- */
static void comm_pi_feed_byte(uint8_t byte)
{
    if (byte == '\r') {
        return;   /* 忽略 CR，只等 LF */
    }
    if (byte == '\n') {
        /* 行结束 → 解析 */
        s_comm_frame_buf[s_comm_frame_len] = '\0';
        if (s_comm_frame_len > 0) {
            comm_pi_parse_line(s_comm_frame_buf);
        }
        s_comm_frame_len = 0;
        return;
    }
    /* 普通字符 → 追加到行缓冲 */
    if (s_comm_frame_len < (COMM_FRAME_MAX_LEN - 1)) {
        s_comm_frame_buf[s_comm_frame_len++] = (char)byte;
    } else {
        /* 行溢出 → 丢弃整行 */
        s_comm_frame_len = 0;
    }
}

/* ============================================================
 *  对外接口
 * ============================================================ */

uint8_t comm_rx_byte;   /* HAL_UART_Receive_IT 单字节缓冲 */

void comm_pi_init(void)
{
    s_comm_frame_len = 0;
    memset(s_comm_frame_buf, 0, sizeof(s_comm_frame_buf));
    s_comm_rx_head = 0;
    s_comm_rx_tail = 0;
    HAL_UART_Receive_IT(&huart1, &comm_rx_byte, 1);
}

void comm_pi_poll(void)
{
    uint8_t ch;
    while (comm_pi_ring_pop(&ch)) {
        comm_pi_feed_byte(ch);
    }
}

/* -----------------------------------------------------------
 *  发送请求给树莓派
 * ----------------------------------------------------------- */
void comm_send_place(char color, uint8_t num, uint8_t row, uint8_t col)
{
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "PLACE,%c,%u,%u,%u\n", color, num, row, col);
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)len, 100);
}

void comm_send_battle_start(char color)
{
    char buf[24];
    int len = snprintf(buf, sizeof(buf), "BATTLE_START,%c\n", color);
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)len, 100);
}

void comm_send_ready(void)
{
    const char *msg = "READY\n";
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, 6, 100);
}
void comm_send_new(void)
{
    const char *msg = "NEW\n";
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, 3, 100);
}

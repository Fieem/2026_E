//
// Created by Administrator on 2026/7/13.
// 上位机通信 - 文本协议解析与收发
// 协议: 逗号分隔，单行一条消息，\n 结尾
//

#include "communicate.h"
#include "Public/public.h"
#include "Test/test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "Emm_V5/Emm_V5.h"

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

/* ---- 忽略大小写字符串比较 ---- */
static int strcmp_upper(const char *a, const char *b_upper)
{
    while (*a && *b_upper) {
        char ca = *a;
        if (ca >= 'a' && ca <= 'z') ca -= 32;
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
    //printsf(0, "RAW: %s", work);
    char *cmd = strtok_r(work, ",", &saveptr);
    if (cmd == NULL) return;

    if (strcmp_upper(cmd, "TASK") == 0) {
        /* TASK,<angle1>,<angle2> */
        char *a1 = strtok_r(NULL, ",", &saveptr);
        char *a2 = strtok_r(NULL, ",", &saveptr);
        if (a1 && a2) {
            comm_angle1 = (float)atof(a1);
            comm_angle2 = (float)atof(a2);
            comm_task_ready = true;
            printsf(0, "TASK %.3f %.3f", comm_angle1, comm_angle2);
        } else {
            printsf(0, "TASK PARAM");
        }
    }
    else if (strcmp_upper(cmd, "ERROR") == 0) {
        char *code = strtok_r(NULL, ",", &saveptr);
        char *msg  = strtok_r(NULL, ",", &saveptr);
        printsf(0, "ERR %s: %s", code ? code : "?", msg ? msg : "");
    }
    else if (strcmp_upper(cmd, "BUSY") == 0) {
        char *msg = strtok_r(NULL, ",", &saveptr);
        printsf(0, "BUSY: %s", msg ? msg : "");
    }
    else if (strcmp_upper(cmd, "WIN") == 0) {
        comm_win_flag = true;
        //Emm_V5_Origin_Trigger_Return(1,0,false);
        Emm_V5_Origin_Trigger_Return(2,0,false);
        printsf(0, "WIN");
    }
    else if (strcmp_upper(cmd, "START") == 0) {
        printsf(0, "START");
    }
    /* 未知命令 → 静默忽略 */
}

/* ---- 逐字节喂入，遇 \n 自动解析 ---- */
static void comm_pi_feed_byte(uint8_t byte)
{
    if (byte == '\r') {
        return;
    }
    if (byte == '\n') {
        s_comm_frame_buf[s_comm_frame_len] = '\0';
        if (s_comm_frame_len > 0) {
            comm_pi_parse_line(s_comm_frame_buf);
        }
        s_comm_frame_len = 0;
        return;
    }
    if (s_comm_frame_len < (COMM_FRAME_MAX_LEN - 1)) {
        s_comm_frame_buf[s_comm_frame_len++] = (char)byte;
    } else {
        s_comm_frame_len = 0;
    }
}

/* ============================================================
 *  对外接口
 * ============================================================ */

uint8_t comm_rx_byte;

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
        //printsf(0, "RX: 0x%02X '%c'", ch, (ch >= 32 && ch <= 126) ? ch : '?');
        comm_pi_feed_byte(ch);
    }
}

/* ---- 发送 ---- */
void comm_send_ready(void)
{
    const char *msg = "READY\n";
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, strlen(msg), 100);
}

void comm_send_ok(void)
{
    const char *msg = "OK\n";
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, strlen(msg), 100);
}

void comm_send_pick(void)
{
    const char *msg = "PICK\n";
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, strlen(msg), 100);
}

void comm_send_finish(void)
{
    const char *msg = "FINISH\n";
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, strlen(msg), 100);
}

/*******************************************************
 * 串口命令说明（VOFA/串口助手）
 *
 * 通用格式:
 *   KEY=VALUE!
 *   也支持一帧多命令: KEY1=V1,KEY2=V2,KEY3=V3!
 *
 * 帧结束符:
 *   !  或  \r  或  \n   （任意一个即可触发解析）
 *   注意: 必须是英文半角 ! ，不是中文全角 ！
 *
 * 命令列表:
 *
 * 示例:
 *   P1=0.80!
 *   TY=90!
 *   SL=120,SR=120!
 *   CAL=1!
 *******************************************************/

#include "test.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmsis_os2.h"
#include "usart.h"
#include "main.h"
#include "Emm_V5/Emm_V5.h"
#include "SG90/sg90.h"
#include "Communicate/communicate.h"
#include <string.h>

#define USART3_TX_BUF_LEN  512U
static char g_screen_tx_buf[USART3_TX_BUF_LEN];
static void prints_u(uint8_t index, unsigned int val);



/* ================================================================
 *  静态变量
 * ================================================================ */

/* VOFA 帧组装缓冲 */
static char     s_vofa_frame_buf[TEST_VOFA_FRAME_MAX_LEN];
static uint8_t  s_vofa_frame_len = 0U;

/* UART 环形缓冲区：ISR 写 head，main loop 读 tail */
static uint8_t           s_rx_ring[TEST_VOFA_RX_RING_SIZE];
static volatile uint16_t s_rx_head = 0U;
static uint16_t          s_rx_tail = 0U;

/* HAL_UART_Receive_IT 单字节接收缓冲 */
uint8_t s_rx_byte = 0U;

/* printsf 多行环形缓存 */
static char    s_log_lines[PRINTSF_MAX_LINES][PRINTSF_MAX_LINE_LEN];
static uint8_t s_log_head  = 0U;
static uint8_t s_log_count = 0U;
static osMutexId_t s_screen_output_mutex = NULL;

/* ================================================================
 *  环形缓冲区操作
 * ================================================================ */

static int rx_ring_is_empty(void)
{
    return (s_rx_tail == s_rx_head);
}

/* 从环形缓冲区取一个字节，返回 0 表示空 */
static int rx_ring_pop(uint8_t *out)
{
    if (rx_ring_is_empty())
    {
        return 0;
    }
    *out = s_rx_ring[s_rx_tail];
    s_rx_tail = (s_rx_tail + 1U) % TEST_VOFA_RX_RING_SIZE;
    return 1;
}

/* 向环形缓冲区写入一个字节（仅 ISR / 回调上下文中调用） */
void rx_ring_push(uint8_t byte)
{
    uint16_t next = (s_rx_head + 1U) % TEST_VOFA_RX_RING_SIZE;
    if (next != s_rx_tail)   /* 未满则写入 */
    {
        s_rx_ring[s_rx_head] = byte;
        s_rx_head = next;
    }
    /* 满了则丢弃该字节 */
}



/* ================================================================
 *  VOFA 命令解析
 * ================================================================ */

static int test_key_equal(const char *a, const char *b)
{
    while ((*a != '\0') && (*b != '\0'))
    {
        if (toupper((unsigned char)*a) != toupper((unsigned char)*b))
        {
            return 0;
        }
        a++;
        b++;
    }
    return ((*a == '\0') && (*b == '\0'));
}

static int test_vofa_apply_kv(const char *key, float value) {
    if (test_key_equal(key, "ZERO")) {

        Emm_V5_MMCL_En_Control(1, true, false);
        Emm_V5_MMCL_En_Control(2, true, false);
        Emm_V5_Multi_Motor_Cmd(0);   // 广播触发，两个电机同时开始
        Emm_V5_Origin_Trigger_Return(1, 0, false);
        Emm_V5_Origin_Trigger_Return(2, 0, false);
        Emm_V5_Multi_Motor_Cmd(0);   // 广播触发，两个电机同时开始
        last_pos_pitch = 0;
        last_pos_yaw   = 0;
        printsf(0,"ZERO");
        return 1;
    }
    if (test_key_equal(key, "ON")) {
        comm_send_new();
        printsf(0,"NEW");
        // Emm_V5_Modify_PID_Params(1,true,18000,0,1024000);
        // Emm_V5_Modify_PID_Params(2,true,18000,0,102400);
        return 1;
    }
    if (test_key_equal(key, "OFF")) {
        Magnet_OFF();
        printsf(0,"OFF");
        return 1;
    }
    if (test_key_equal(key, "LOW")) {
        SG90_SetAngle(Low_Angle);
        printsf(0,"LOW");
        return 1;
    }
    if (test_key_equal(key, "HIGH")) {
        SG90_SetAngle(High_Angle);
        printsf(0,"HIGH");
        return 1;
    }
    if (test_key_equal(key, "DISABLE")) {
        Emm_V5_MMCL_En_Control(1, false, false);
        Emm_V5_MMCL_En_Control(2, false, false);
        Emm_V5_Multi_Motor_Cmd(0);   // 广播触发，两个电机同时开始
        printsf(0,"DISABLE");
        return 1;
    }
    if (test_key_equal(key, "TEST")) {
        int32_t pulse1 = Emm_V5_Get_Pulse(1);   // yaw 轴当前脉冲
        int32_t pulse2 = Emm_V5_Get_Pulse(2);   // pitch 轴当前脉冲
        printsf(0, "P1=%ld P2=%ld", pulse1, pulse2);
        return 1;
    }

    /* ---- 树莓派协议调试：变量预设 + 触发发送 ---- */
    /* 静态变量，保存屏幕设定的参数 */
    static char    s_cmd_color = 'W';
    static uint8_t s_cmd_num   = 1;
    static uint8_t s_cmd_row   = 1;
    static uint8_t s_cmd_col   = 1;

    if (test_key_equal(key, "COLOR")) {
        s_cmd_color = ((int)value == 0) ? 'B' : 'W';
        printsf(0, "COLOR=%c", s_cmd_color);
        return 1;
    }
    if (test_key_equal(key, "NUM")) {
        s_cmd_num = (uint8_t)value;
        printsf(0, "NUM=%u", (unsigned int)s_cmd_num);
        return 1;
    }
    if (test_key_equal(key, "POSR")) {
        s_cmd_row = (uint8_t)value;
        printsf(0, "POSR=%u", (unsigned int)s_cmd_row);
        return 1;
    }
    if (test_key_equal(key, "POSC")) {
        s_cmd_col = (uint8_t)value;
        printsf(0, "POSC=%u", (unsigned int)s_cmd_col);
        return 1;
    }
    if (test_key_equal(key, "PLACE")) {
        comm_send_place(s_cmd_color,s_cmd_num, s_cmd_row, s_cmd_col);
        printsf(0, "PLACE %c,%u,%u,%u", s_cmd_color,s_cmd_num,
                (unsigned int)s_cmd_row, (unsigned int)s_cmd_col);
        return 1;
    }
    if (test_key_equal(key, "BATTLE")) {
        comm_send_battle_start(s_cmd_color);
        printsf(0, "BATTLE %c", s_cmd_color);
        return 1;
    }
    if (test_key_equal(key, "READY")) {
        comm_send_ready();
        printsf(0,"READY sent");
        return 1;
    }
    /*
     * 在此处添加你的 PID 参数赋值逻辑，例如:
     *
     *   if (test_key_equal(key, "P1")) { pid_angle.Kp = value;
     *       printsf(0, "[VOFA] P1=%.4f", value);
     *       return 1;
     *   }
     *   if (test_key_equal(key, "I1")) { pid_angle.Ki = value; return 1; }
     *   if (test_key_equal(key, "D1")) { pid_angle.Kd = value; return 1; }
     */

    (void)key;
    (void)value;
    return 0;
}

static void test_vofa_parse_segment(char *segment)
{
    char *eq        = NULL;
    char *key       = NULL;
    char *value_str = NULL;
    char *endptr    = NULL;
    float value     = 0.0f;

    /* 跳过前导空格 */
    while (isspace((unsigned char)*segment)) { segment++; }
    if (*segment == '\0')
    {
        return;
    }

    eq = strchr(segment, '=');
    if (eq == NULL)
    {
        return;
    }

    *eq = '\0';
    key       = segment;
    value_str = eq + 1;

    /* 跳过值前导空格 */
    while (isspace((unsigned char)*value_str)) { value_str++; }
    if (*value_str == '\0')
    {
        return;
    }

    value = strtof(value_str, &endptr);
    if (endptr == value_str)
    {
        return;
    }

    /* 值后只允许空白字符 */
    while ((endptr != NULL) && isspace((unsigned char)*endptr)) { endptr++; }
    if ((endptr != NULL) && (*endptr != '\0'))
    {
        return;
    }

    if (test_vofa_apply_kv(key, value))
    {
        // printsf(0, "[VOFA] %s=%.4f", key, value);
    }
}

void test_vofa_parse_frame(const char *frame)
{
    char   work_buf[TEST_VOFA_FRAME_MAX_LEN];
    char  *token  = NULL;
    size_t len    = 0U;

    if (frame == NULL)
    {
        return;
    }

    len = strlen(frame);
    if (len >= TEST_VOFA_FRAME_MAX_LEN)
    {
        len = TEST_VOFA_FRAME_MAX_LEN - 1U;
    }

    memcpy(work_buf, frame, len);
    work_buf[len] = '\0';

    token = strtok(work_buf, ",;");
    while (token != NULL)
    {
        test_vofa_parse_segment(token);
        token = strtok(NULL, ",;");
    }
}

void test_vofa_feed_byte(uint8_t byte)
{
    /* 碰到帧结束符则触发解析 */
    if ((byte == '!') || (byte == '\r') || (byte == '\n'))
    {
        if (s_vofa_frame_len > 0U)
        {
            s_vofa_frame_buf[s_vofa_frame_len] = '\0';
            test_vofa_parse_frame(s_vofa_frame_buf);
            s_vofa_frame_len = 0U;
            memset(s_vofa_frame_buf, 0, sizeof(s_vofa_frame_buf));
        }
        return;
    }

    /* 累积字节 */
    if (s_vofa_frame_len < (TEST_VOFA_FRAME_MAX_LEN - 1U))
    {
        s_vofa_frame_buf[s_vofa_frame_len++] = (char)byte;
    }
    else
    {
        /* 缓冲区溢出，丢弃当前帧等待下一个分隔符 */
        s_vofa_frame_len = 0U;
        memset(s_vofa_frame_buf, 0, sizeof(s_vofa_frame_buf));
    }
}

/* ================================================================
 *  公开 API
 * ================================================================ */

/**
  * @brief  初始化 VOFA+ 接收
  * @note   配置 USART2 NVIC 中断，并启动 HAL_UART_Receive_IT 单字节循环接收
  */
void test_vofa_init(void)
{
    s_vofa_frame_len = 0U;
    memset(s_vofa_frame_buf, 0, sizeof(s_vofa_frame_buf));

    s_rx_head = 0U;
    s_rx_tail = 0U;


    /* ---- 启动单字节中断接收（每收到一个字节触发 HAL_UART_RxCpltCallback） ---- */
    HAL_UART_Receive_IT(TEST_VOFA_HUART, &s_rx_byte, 1);
}

/**
  * @brief  轮询环形缓冲区并解析收到的字节
  * @note   建议在 main 的 while(1) 中高频调用
  */
void test_vofa_poll(void)
{
    uint8_t ch;
    while (rx_ring_pop(&ch))
    {
        test_vofa_feed_byte(ch);
    }
}

/* ================================================================
 *  Nextion 显示函数
 * ================================================================ */

void screen_output_init(void)
{
    s_screen_output_mutex = osMutexNew(NULL);
}

static int screen_output_lock(void)
{
    if (s_screen_output_mutex == NULL)
    {
        return 1;
    }
    return osMutexAcquire(s_screen_output_mutex, 100U) == osOK;
}

static void screen_output_unlock(void)
{
    if (s_screen_output_mutex != NULL)
    {
        (void)osMutexRelease(s_screen_output_mutex);
    }
}

static int screen_uart_write(const uint8_t *data, uint16_t length)
{
    uint32_t start = HAL_GetTick();
    uint16_t i;

    for (i = 0U; i < length; i++)
    {
        while ((USART3->SR & USART_SR_TXE) == 0U)
        {
            if ((HAL_GetTick() - start) >= 100U)
            {
                return 0;
            }
        }
        USART3->DR = data[i];
    }

    while ((USART3->SR & USART_SR_TC) == 0U)
    {
        if ((HAL_GetTick() - start) >= 100U)
        {
            return 0;
        }
    }
    return 1;
}

static void prints_unlocked(uint8_t index, const char *content)
{
    if (content == NULL)
    {
        content = "";
    }

    int len = snprintf(g_screen_tx_buf, sizeof(g_screen_tx_buf),
                       "page0.t%u.txt=\"%s\"\xFF\xFF\xFF",
                       (unsigned int)index, content);
    if (len > 0)
    {
        if (len >= (int)sizeof(g_screen_tx_buf))
        {
            len = (int)sizeof(g_screen_tx_buf) - 1;
        }
        (void)screen_uart_write((const uint8_t *)g_screen_tx_buf,
                                (uint16_t)len);
    }
}
/**
  * @brief  Nextion 文本控件打印
  * @param  index   控件索引（t0.txt, t1.txt ...）
  * @param  content 文本内容
  */
void prints(uint8_t index, const char *content)
{
    if (!screen_output_lock())
    {
        return;
    }
    prints_unlocked(index, content);
    screen_output_unlock();
}


static void prints_u(uint8_t index, unsigned int val)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%u", val);
    prints(index, buf);
}

/**
  * @brief  将一行文本存入环形日志缓存
  */
static void printsf_store_line(const char *line)
{
    uint8_t pos;

    if (s_log_count < PRINTSF_MAX_LINES)
    {
        pos = (uint8_t)((s_log_head + s_log_count) % PRINTSF_MAX_LINES);
        s_log_count++;
    }
    else
    {
        /* 覆盖最旧行 */
        pos        = s_log_head;
        s_log_head = (uint8_t)((s_log_head + 1U) % PRINTSF_MAX_LINES);
    }

    strncpy(s_log_lines[pos], line, PRINTSF_MAX_LINE_LEN - 1U);
    s_log_lines[pos][PRINTSF_MAX_LINE_LEN - 1U] = '\0';
}

/**
  * @brief  格式化 Nextion 文本控件打印（支持多行滚动）
  * @param  index 控件索引
  * @param  fmt   格式化字符串
  * @param  ...   可变参数
  */
void printsf(uint8_t index, const char *fmt, ...)
{
    static char    line[PRINTSF_MAX_LINE_LEN];
    static char    merged[PRINTSF_TEXT_BUF_LEN];
    va_list args;
    size_t  off = 0U;
    uint8_t i;

    if (fmt == NULL)
    {
        return;
    }

    if (!screen_output_lock())
    {
        return;
    }

    va_start(args, fmt);
    (void)vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);

    /* 将双引号替换为单引号，避免破坏 Nextion 命令字符串 */
    for (i = 0U; line[i] != '\0'; i++)
    {
        if (line[i] == '"') { line[i] = '\''; }
    }

    printsf_store_line(line);

    merged[0] = '\0';
    for (i = 0U; i < s_log_count; i++)
    {
        uint8_t pos = (uint8_t)((s_log_head + i) % PRINTSF_MAX_LINES);
        int n = snprintf(merged + off, sizeof(merged) - off, "%s%s",
                         s_log_lines[pos],
                         (i + 1U < s_log_count) ? "\r\n" : "");
        if (n < 0) { break; }
        if ((size_t)n >= (sizeof(merged) - off))
        {
            off = sizeof(merged) - 1U;
            merged[off] = '\0';
            break;
        }
        off += (size_t)n;
    }

    prints_unlocked(index, merged);
    screen_output_unlock();
}

/**
  * @brief  清空指定 Nextion 文本控件
  */
void printsf_clear(uint8_t index)
{
    uint8_t i;

    if (!screen_output_lock())
    {
        return;
    }

    s_log_head  = 0U;
    s_log_count = 0U;

    for (i = 0U; i < PRINTSF_MAX_LINES; i++)
    {
        s_log_lines[i][0] = '\0';
    }

    prints_unlocked(index, "");
    screen_output_unlock();
}

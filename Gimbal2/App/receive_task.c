//
// Created by Administrator on 2026/7/8.
// 上位机通信接收任务 - USART1 文本协议
//

#include "main.h"
#include "usart.h"
#include <stm32f4xx_hal_uart.h>
#include "Communicate/communicate.h"
#include "Test/test.h"
#include "cmsis_os2.h"
#include "Public/public.h"

void StartReceiveTask(void *argument)
{
    comm_pi_init();   /* 启动 USART1 单字节中断接收（文本协议） */

    for (;;) {
        osDelay(5);   /* 5ms 周期 */

        /* 排空环形缓冲区，解析文本行 */
        comm_pi_poll();

        /* 收到完整 TASK 响应 → 写入 arm_cmd 触发状态机 */
        if (comm_task_ready) {
            comm_task_ready = false;
            if (arm_state == ARM_IDLE && arm_cmd.type == CMD_NONE) {
                arm_cmd.pick_x  = comm_arm_start;
                arm_cmd.pick_y  = comm_rot_start;
                arm_cmd.place_x = comm_arm_end;
                arm_cmd.place_y = comm_rot_place;
                arm_cmd.type    = CMD_EXEC;
            } else {
                printsf(0, "ARM BUSY");
            }
        }
    }
}

/* ================================================================
 *  HAL UART 接收回调（覆盖 __weak 默认实现）
 *  每收到一个字节就推入对应环形缓冲区，然后重新启动单字节接收
 * ================================================================ */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        /* 上位机通信 → 环形缓冲区 */
        comm_pi_ring_push(comm_rx_byte);
        HAL_UART_Receive_IT(&huart1, &comm_rx_byte, 1);
    }
    else if (huart == TEST_VOFA_HUART)
    {
        /* VOFA+ → 环形缓冲区 */
        rx_ring_push(s_rx_byte);
        HAL_UART_Receive_IT(TEST_VOFA_HUART, &s_rx_byte, 1);
    }
}

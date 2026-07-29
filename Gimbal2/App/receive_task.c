//
// Created by Administrator on 2026/7/8.
// 树莓派通信接收任务 - USART1 文本协议
//

#include "main.h"
#include "usart.h"
#include <stm32f4xx_hal_uart.h>
#include "Communicate/communicate.h"
#include "Test/test.h"
#include "cmsis_os2.h"
#include "Public/public.h"
#include "Emm_V5/Emm_V5.h"

void StartReceiveTask(void *argument)
{
    comm_pi_init();   /* 启动 USART1 单字节中断接收（文本协议） */

    for (;;) {
        osDelay(5);   /* 5ms 周期，快于 MotorTask 的 10ms */

        /* 排空环形缓冲区，解析文本行 */
        comm_pi_poll();

        /* 收到完整 PULSES 响应 → 触发取子→放子序列 */
        if (comm_response_ready) {
            comm_response_ready = false;
            if (arm_state == ARM_IDLE && arm_cmd.type == CMD_NONE) {
                Arm_Execute_Pick_Place(comm_pick_p1,  comm_pick_p2,
                                       comm_place_p1, comm_place_p2);
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
        /* 树莓派文本协议 → 环形缓冲区 */
        comm_pi_ring_push(comm_rx_byte);
        HAL_UART_Receive_IT(&huart1, &comm_rx_byte, 1);
    }
    else if (huart == TEST_VOFA_HUART)
    {
        /* VOFA+ → 环形缓冲区（不变） */
        rx_ring_push(s_rx_byte);
        HAL_UART_Receive_IT(TEST_VOFA_HUART, &s_rx_byte, 1);
    }
}

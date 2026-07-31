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
    comm_pi_init();

    for (;;) {
        osDelay(5);
        comm_pi_poll();

        if (comm_task_ready) {
            comm_task_ready = false;

            /* 先尝试作为第一段 TASK（抓取点），不在 IDLE 则尝试第二段（放置点）；
             * 状态校验与写入在同一临界区内完成，两边都不受理说明时序错位，丢弃并提示 */
            if (!Arm_TryPostPick(comm_angle1, comm_angle2) &&
                !Arm_TryPostPlace(comm_angle1, comm_angle2)) {
                printsf(0, "TASK DROP");
            }
        }

        if (comm_win_flag) {
            comm_win_flag = false;
            for (int i = 0; i < 20; i++) {
                HAL_GPIO_TogglePin(GPIOF, GPIO_PIN_9);
                osDelay(100);
            }
        }
    }
}

/* ================================================================
 *  HAL UART 接收回调
 * ================================================================ */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        HAL_GPIO_TogglePin(GPIOF, GPIO_PIN_9);  // 加这行
        comm_pi_ring_push(comm_rx_byte);
        HAL_UART_Receive_IT(&huart1, &comm_rx_byte, 1);
    }
    else if (huart == TEST_VOFA_HUART)
    {
        rx_ring_push(s_rx_byte);
        HAL_UART_Receive_IT(TEST_VOFA_HUART, &s_rx_byte, 1);
    }
}
/* 错误回调：噪声/过载导致 USART 停摆时自动恢复 */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        HAL_UART_Receive_IT(&huart1, &comm_rx_byte, 1);
    } else if (huart == TEST_VOFA_HUART) {
        HAL_UART_Receive_IT(TEST_VOFA_HUART, &s_rx_byte, 1);
    }
}

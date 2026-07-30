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

            if (arm_state == ARM_IDLE && arm_cmd.type == CMD_NONE) {
                /* 第一段 TASK → 初始位置 */
                arm_cmd.pick_x = comm_angle1;
                arm_cmd.pick_y = comm_angle2;
                arm_cmd.type   = CMD_TASK;
            }
            else if (arm_state == ARM_WAIT_PLACE && arm_cmd.type == CMD_NONE) {
                /* 第二段 TASK → 放置位置 */
                arm_cmd.place_x = comm_angle1;
                arm_cmd.place_y = comm_angle2;
                arm_cmd.type    = CMD_PLACE;
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

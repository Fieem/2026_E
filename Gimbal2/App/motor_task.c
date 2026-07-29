//
// Created by Administrator on 2026/7/29.
// 机械臂状态机 - 取物→放物→回零
//
#include "main.h"
#include "can.h"
#include "cmsis_os2.h"
#include "Emm_V5/Emm_V5.h"
#include "SG90/sg90.h"
#include "Public/public.h"
#include "Communicate/communicate.h"

void StartMotorTask(void *argument)
{
    /* ---- 初始化 CAN 滤波器并启动回零 ---- */
    USER_CAN1_Filter_Init();
    Emm_V5_Origin_Trigger_Return(1, 0, false);
    Emm_V5_Origin_Trigger_Return(2, 0, false);
    osDelay(MOTOR_DELAY_MS);   /* 等待上电回零完成 */

    HAL_GPIO_TogglePin(GPIOF, GPIO_PIN_9);

    for (;;) {
        switch (arm_state) {

        case ARM_IDLE:
            if (arm_cmd.type == CMD_EXEC) {
                arm_cmd.type = CMD_NONE;
                arm_state = ARM_MOVING_TO_START;

                /* 两电机同时启动 → 初始位置 */
                Move(1, arm_cmd.pick_x);                             // p1: 机械臂
                Rotate(2, PULSES_TO_DEG(arm_cmd.pick_y));            // p2: 旋转
            }
            break;

        case ARM_MOVING_TO_START:
            osDelay(MOTOR_DELAY_MS);
            arm_state = ARM_SERVO_PICK_DOWN;
            SG90_SetAngle(Low_Angle);    /* 舵机下降 */
            break;

        case ARM_SERVO_PICK_DOWN:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_MAGNET_PICK_ON;
            Magnet_ON();                 /* 电磁铁吸合 */
            break;

        case ARM_MAGNET_PICK_ON:
            osDelay(MAGNET_DELAY_MS);
            arm_state = ARM_SERVO_PICK_UP;
            SG90_SetAngle(High_Angle);   /* 舵机上升 */
            break;

        case ARM_SERVO_PICK_UP:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_MOVING_TO_PLACE;

            /* 两电机同时启动 → 放置位置 */
            Move(1, arm_cmd.place_x);                                // p3: 机械臂
            Rotate(2, PULSES_TO_DEG(arm_cmd.place_y));               // p4: 旋转
            break;

        case ARM_MOVING_TO_PLACE:
            osDelay(MOTOR_DELAY_MS);
            arm_state = ARM_SERVO_PLACE_DOWN;
            SG90_SetAngle(Low_Angle);    /* 舵机下降 */
            break;

        case ARM_SERVO_PLACE_DOWN:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_MAGNET_PLACE_OFF;
            Magnet_OFF();                /* 电磁铁释放 */
            break;

        case ARM_MAGNET_PLACE_OFF:
            osDelay(MAGNET_DELAY_MS);
            arm_state = ARM_SERVO_PLACE_UP;
            SG90_SetAngle(High_Angle);   /* 舵机上升 */
            break;

        case ARM_SERVO_PLACE_UP:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_MOVING_TO_ZERO;

            /* 回零 */
            Move(1, 0);
            Rotate(2, 0.0f);
            break;

        case ARM_MOVING_TO_ZERO:
            osDelay(MOTOR_DELAY_MS);
            arm_state = ARM_DONE;
            break;

        case ARM_DONE:
            comm_send_new();             /* 通知上位机完成 */
            arm_state = ARM_IDLE;
            break;
        }
        osDelay(5);   /* 5ms 状态机节拍 */
    }
}

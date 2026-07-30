//
// Created by Administrator on 2026/7/29.
// 机械臂状态机 - 两段 TASK 交互
//
// 流程:
//   TASK(dist,a1,a2) → 移动到初始 → 舵机下降(CalcPickAngle) → 电磁铁吸合 → 舵机上升 → 发 PICK
//   TASK(dist,a1,a2) → 移动到放置 → 舵机下降(CalcPickAngle) → 电磁铁释放 → 舵机上升 → FINISH
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
    osDelay(1000);

    HAL_GPIO_TogglePin(GPIOF, GPIO_PIN_9);

    for (;;) {
        switch (arm_state) {

        /* ---- 等上位机发第一段 TASK ---- */
        case ARM_IDLE:
            if (arm_cmd.type == CMD_TASK) {
                arm_cmd.type = CMD_NONE;
                arm_state = ARM_MOVING_TO_START;

                Rotate(1, (float)arm_cmd.pick_x);
                Rotate(2, (float)arm_cmd.pick_y);
            }
            break;

        case ARM_MOVING_TO_START:
            osDelay(MOTOR_DELAY_MS);
            arm_state = ARM_SERVO_PICK_DOWN;
            SG90_SetAngle(CalcPickAngle(arm_cmd.pick_dist));
            break;

        case ARM_SERVO_PICK_DOWN:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_MAGNET_PICK_ON;
            Magnet_ON();
            break;

        case ARM_MAGNET_PICK_ON:
            osDelay(MAGNET_DELAY_MS);
            SG90_SetAngle(High_Angle);          /* 先抬起舵机 */
            osDelay(SERVO_DELAY_MS);            /* 等待舵机到位 */
            arm_state = ARM_WAIT_PLACE;
            comm_send_pick();                   /* 舵机已抬起，通知上位机抓取完成 */
            break;

        /* ---- 等上位机发第二段 TASK ---- */
        case ARM_WAIT_PLACE:
            if (arm_cmd.type == CMD_PLACE) {
                arm_cmd.type = CMD_NONE;
                arm_state = ARM_MOVING_TO_PLACE;
                Rotate(1, (float)arm_cmd.place_x);
                Rotate(2, (float)arm_cmd.place_y);
            }
            break;

        case ARM_SERVO_PICK_UP:                 /* 不再使用，保留以防枚举索引错位 */
            arm_state = ARM_MOVING_TO_PLACE;
            break;

        case ARM_MOVING_TO_PLACE:
            osDelay(MOTOR_DELAY_MS);
            arm_state = ARM_SERVO_PLACE_DOWN;
            SG90_SetAngle(CalcPickAngle(arm_cmd.place_dist));
            break;

        case ARM_SERVO_PLACE_DOWN:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_MAGNET_PLACE_OFF;
            Magnet_OFF();
            break;

        case ARM_MAGNET_PLACE_OFF:
            osDelay(MAGNET_DELAY_MS);
            arm_state = ARM_SERVO_PLACE_UP;
            SG90_SetAngle(High_Angle);
            break;

        case ARM_SERVO_PLACE_UP:
            osDelay(SERVO_DELAY_MS);
            arm_state = ARM_DONE;
            break;

        case ARM_DONE:
            comm_send_finish();
            arm_state = ARM_IDLE;
            break;
        }
        osDelay(5);
    }
}

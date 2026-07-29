#include "cmsis_os2.h"
#include "SG90/sg90.h"
//
// Created by Administrator on 2026/7/9.
//
void StartSg90Task(void *argument) {
    SG90_Init();
    Magnet_OFF();
    // 初始化到停止位（360°舵机 1.5ms = 停止）
    //SG90_SetAngle(90);
    //HAL_GPIO_TogglePin(GPIOF, GPIO_PIN_9);
    for (;;) {
        // 正转（75° → CW 方向慢速旋转）
        //osDelay(1000);
        //
        // // 反转（105° → CCW 方向慢速旋转）
        //Magnet_ON();

        osDelay(1000);
    }
}
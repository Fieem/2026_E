//
// Created by Administrator on 2026/7/9.
//

#ifndef INC_2024E_SG90_H
#define INC_2024E_SG90_H

#include "main.h"
#include "tim.h"

/**********************************************************
***	SG90 舵机驱动
***	引脚：PA3 (TIM2_CH4)
***	PWM 频率：50Hz (20ms)
***	0° = 0.5ms 脉宽  → CCR = 500
***	90° = 1.5ms 脉宽 → CCR = 1500
***	180° = 2.5ms 脉宽 → CCR = 2500
**********************************************************/

#define Low_Angle   180

#define High_Angle   10
/**
  * @brief  SG90 初始化，开启 PWM 输出
  * @param  无
  * @retval 无
  */
void SG90_Init(void);

/**
  * @brief  设置舵机角度
  * @param  angle ：目标角度，范围 0 ~ 180
  * @retval 无
  */
void SG90_SetAngle(uint8_t angle);


/**********************************************************
  ***   电磁铁驱动
  ***   引脚：PA5 (Magnet)
  ***   低电平吸合，高电平释放
  **********************************************************/

  /**
    * @brief  电磁铁吸合
    * @param  无
    * @retval 无
    */
  void Magnet_ON(void);

  /**
    * @brief  电磁铁释放
    * @param  无
    * @retval 无
    */
  void Magnet_OFF(void);
#endif //INC_2024E_SG90_H

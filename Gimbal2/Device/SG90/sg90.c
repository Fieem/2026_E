//
// Created by Administrator on 2026/7/9.
//

#include "sg90.h"

/**********************************************************
***	SG90 舵机驱动
***	引脚：PA3 (TIM2_CH4)
***	PWM 频率：50Hz (20ms)
***	0° = 0.5ms 脉宽  → CCR = 500
***	90° = 1.5ms 脉宽 → CCR = 1500
***	180° = 2.5ms 脉宽 → CCR = 2500
**********************************************************/

/**
  * @brief  SG90 初始化，开启 PWM 输出
  * @param  无
  * @retval 无
  */
void SG90_Init(void)
{
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);
    SG90_SetAngle(High_Angle);
}

/**
  * @brief  设置舵机角度
  * @param  angle ：目标角度，范围 0 ~ 180
  * @retval 无
  */
void SG90_SetAngle(uint8_t angle)
{
    // 限制角度范围
    if (angle > 180) { angle = 180; }

    // 角度转脉宽：500 + angle * 2000 / 180
    // 500 = 0.5ms（0°），2500 = 2.5ms（180°）
    uint16_t pulse = 500 + (uint16_t)((uint32_t)angle * 2000 / 180);

    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, pulse);
}

/**********************************************************
  ***   电磁铁驱动
  ***   引脚：PA5 (Magnet)
  ***   低电平吸合，高电平释放
  **********************************************************/

  /**
    * @brief  电磁铁释放（PA5 输出低电平）
    * @param  无
    * @retval 无
    */
  void Magnet_OFF(void)
  {
      HAL_GPIO_WritePin(Magnet_GPIO_Port, Magnet_Pin, GPIO_PIN_RESET);
  }

  /**
    * @brief  电磁铁吸合（PA5 输出高电平）
    * @param  无
    * @retval 无
    */
void Magnet_ON(void)
{
    HAL_GPIO_WritePin(Magnet_GPIO_Port, Magnet_Pin, GPIO_PIN_SET);
}
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

#define Low_Angle   160

#define High_Angle   12

/* ---- 舵机下垂补偿（电机角度绝对值越大→下垂越小→下降角度越大）---- */
#define COMP_ANGLE_MIN   40     /* 电机角度=0°(下垂最严重) */
#define COMP_ANGLE_MAX   120    /* 电机角度=±60°(下垂最轻微，之后饱和) */
#define COMP_LOW_MIN     70   /* 下垂最严重时的舵机角度 */
#define COMP_LOW_MAX     175   /* 下垂最轻微时的舵机角度 */
#define COMP_CURVE_EXP   0.5f  /* 曲线指数: >1=开头缓结尾陡, <1=开头陡结尾缓, 1=线性 */
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

uint8_t CalcPickAngle(float angle);   /* 根据第一个电机旋转度数计算舵机下降角度 */


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

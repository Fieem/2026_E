//
// Created by Administrator on 2026/7/9.
//

#include "sg90.h"
#include <math.h>

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
/**
  * @brief  根据电机角度计算舵机下降角度（取绝对值）
  * @param  angle ：电机角度(度)，范围 ±180
  * @retval 舵机角度 0~180
  * @note   角度绝对值越大→离底座越近→下垂越小→下降角度越大
  */
uint8_t CalcPickAngle(float angle)
{
    float abs_angle = (angle >= 0.0f) ? angle : -angle;
    if (abs_angle <= COMP_ANGLE_MIN) return COMP_LOW_MIN;
    if (abs_angle >= COMP_ANGLE_MAX) return COMP_LOW_MAX;
    float ratio = (abs_angle - COMP_ANGLE_MIN) / (float)(COMP_ANGLE_MAX - COMP_ANGLE_MIN);
    float curved = powf(ratio, COMP_CURVE_EXP);
    return (uint8_t)(COMP_LOW_MIN + curved * (COMP_LOW_MAX - COMP_LOW_MIN) + 0.5f);
}
/* ---- 舵机下垂补偿查表（每10°一个区间）---- */
typedef struct {
    uint8_t angle_min;    /* 区间下限(包含) */
    uint8_t servo;        /* 对应舵机角度   */
} PickLutEntry_t;

static const PickLutEntry_t s_pick_lut[] = {
    { 40,  70 },   /*  40°- 49° */
    { 50,  85 },   /*  50°- 59° */
    { 60, 110 },   /*  60°- 69° */
    { 70, 112 },   /*  70°- 79° */
    { 80, 129 },   /*  80°- 89° */
    { 90, 140 },   /*  90°- 99° */
    { 100, 151 },  /* 100°-109° */
    { 110, 161 },  /* 110°-119° */
};
#define PICK_LUT_SIZE  (sizeof(s_pick_lut) / sizeof(s_pick_lut[0]))

/**
  * @brief  查表计算舵机下降角度（每10°一个区间）
  * @param  angle ：电机角度(度)，范围 ±180
  * @retval 舵机角度 0~180
  */
uint8_t CalcPickAngleLut(float angle)
{
    float abs_angle = (angle >= 0.0f) ? angle : -angle;

    /* 范围外钳位 */
    if (abs_angle <= (float)s_pick_lut[0].angle_min)
        return s_pick_lut[0].servo;
    if (abs_angle >= (float)(s_pick_lut[PICK_LUT_SIZE - 1].angle_min + 10U))
        return COMP_LOW_MAX;

    /* 从后往前找第一个 angle >= 区间下限 */
    int i;
    for (i = (int)PICK_LUT_SIZE - 1; i >= 0; i--) {
        if (abs_angle >= (float)s_pick_lut[i].angle_min) {
            return s_pick_lut[i].servo;
        }
    }
    return s_pick_lut[0].servo;  /* fallback */
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
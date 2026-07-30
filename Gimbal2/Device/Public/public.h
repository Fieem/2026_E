//
// Created by Administrator on 2026/7/11.
//

#ifndef INC_2024E_PUBLIC_H
#define INC_2024E_PUBLIC_H
#include <stdint.h>
#include <stdbool.h>

/* ---- 延时宏定义（用户自行调整）---- */
#define MOTOR_DELAY_MS      6000U   /* 电机移动到位延时(ms) */
#define SERVO_DELAY_MS      500U    /* 舵机动作延时(ms)      */
#define MAGNET_DELAY_MS     300U    /* 电磁铁稳定延时(ms)    */

/* ---- 脉冲 ↔ 角度换算 ---- */
#define PULSES_PER_REV      3200U
#define PULSES_TO_DEG(p)    ((float)(p) * 360.0f / (float)PULSES_PER_REV)

/* ---- 全局变量 ---- */
extern int32_t last_pos_yaw;
extern int32_t last_pos_pitch;

/* ---- 机械臂命令类型 ---- */
typedef enum {
    CMD_NONE = 0,
    CMD_TASK,
    CMD_PLACE,
} ArmCmdType_t;

/* ---- 机械臂状态机 ---- */
typedef enum {
    ARM_IDLE = 0,
    ARM_MOVING_TO_START,        /* 两电机 → 初始位置              */
    ARM_SERVO_PICK_DOWN,        /* 舵机下降                       */
    ARM_MAGNET_PICK_ON,         /* 电磁铁吸合                     */
    ARM_WAIT_PLACE,             /* 发 PICK，等上位机发下一组角度  */
    ARM_SERVO_PICK_UP,          /* 舵机上升                       */
    ARM_MOVING_TO_PLACE,        /* 两电机 → 放置位置              */
    ARM_SERVO_PLACE_DOWN,       /* 舵机下降                       */
    ARM_MAGNET_PLACE_OFF,       /* 电磁铁释放                     */
    ARM_SERVO_PLACE_UP,         /* 舵机上升                       */
    ARM_DONE,                   /* 发 FINISH → IDLE              */
} ArmState_t;

/* ---- 命令结构体 ---- */
typedef struct {
    ArmCmdType_t type;
    int32_t      pick_x;        /* 初始位置 电机1 (度) */
    int32_t      pick_y;        /* 初始位置 电机2 (度) */
    int32_t      place_y;       /* 放置位置 电机2 (度) */
    int32_t      place_x;       /* 放置位置 电机1 (度) */
} ArmCmd_t;

extern ArmCmd_t   arm_cmd;
extern ArmState_t arm_state;

/* ---- TASK 协议解析结果（每次 2 个角度值）---- */
extern bool    comm_task_ready;
extern int32_t comm_angle1;
extern int32_t comm_angle2;

extern bool    comm_win_flag;

#endif //INC_2024E_PUBLIC_H

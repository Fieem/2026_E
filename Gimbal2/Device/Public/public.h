//
// Created by Administrator on 2026/7/11.
//

#ifndef INC_2024E_PUBLIC_H
#define INC_2024E_PUBLIC_H
#include <stdint.h>
#include <stdbool.h>

/* ---- 延时宏定义（用户自行调整）---- */
#define MOTOR_DELAY_MS      3000U   /* 电机移动到位延时(ms) */
#define SERVO_DELAY_MS      1000U    /* 舵机动作延时(ms)      */
#define MAGNET_DELAY_MS     1000U    /* 电磁铁稳定延时(ms)    */

/* ---- 脉冲 ↔ 角度换算 ---- */
#define PULSES_PER_REV      6400U
#define PULSES_TO_DEG(p)    ((float)(p) * 360.0f / (float)PULSES_PER_REV)

/* ---- 全局变量（motor_task / test_task 共享，ISR 外读写，加 volatile 防缓存）---- */
extern volatile int32_t last_pos_yaw;
extern volatile int32_t last_pos_pitch;

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

/* ---- 机械臂状态/命令（receive_task ↔ motor_task 跨任务共享）----
 * 禁止直接访问底层变量，一律通过下列 API：
 * 内部用临界区保证 "检查状态 → 写命令" 与 "取命令 → 切状态" 均为原子操作，
 * 避免编译器缓存和两任务交错执行导致的半更新/重复投递。 */
ArmState_t Arm_GetState(void);
void       Arm_SetState(ArmState_t new_state);

/* receive_task 发布命令：仅在对应状态且邮箱为空时生效，返回 true=已受理 */
bool Arm_TryPostPick(float x, float y);    /* ARM_IDLE       + 邮箱空 → CMD_TASK  */
bool Arm_TryPostPlace(float x, float y);   /* ARM_WAIT_PLACE + 邮箱空 → CMD_PLACE */

/* motor_task 取走命令：匹配 expect 时取出坐标并原子切换到 new_state */
bool Arm_TakeCmd(ArmCmdType_t expect, ArmState_t new_state, float *x, float *y);

/* ---- TASK 协议解析结果（每次 2 个角度值）---- */
extern bool    comm_task_ready;
extern float   comm_angle1;
extern float   comm_angle2;
extern bool    comm_win_flag;
#endif //INC_2024E_PUBLIC_H

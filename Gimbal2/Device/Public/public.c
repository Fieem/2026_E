#include <stdbool.h>
#include <stdint.h>
#include "main.h"
#include "FreeRTOS.h"
#include "task.h"
#include "public.h"
//
// Created by Administrator on 2026/7/11.
//

volatile int32_t last_pos_yaw   = 0;
volatile int32_t last_pos_pitch = 0;

/* ================================================================
 *  机械臂命令邮箱（receive_task 写 / motor_task 读）
 *  type 充当发布标志：先写坐标、最后写 type；取件时先拷出、再清 type，
 *  且取件与状态切换在同一临界区内完成，杜绝窗口期重复投递。
 * ================================================================ */
typedef struct {
    ArmCmdType_t type;
    float        pick_x;        /* 初始位置 电机1 (度，小数) */
    float        pick_y;        /* 初始位置 电机2 (度，小数) */
    float        place_x;       /* 放置位置 电机1 (度，小数) */
    float        place_y;       /* 放置位置 电机2 (度，小数) */
} ArmMailbox_t;

static ArmMailbox_t s_arm_mailbox = { .type = CMD_NONE };
static ArmState_t   s_arm_state   = ARM_IDLE;

ArmState_t Arm_GetState(void)
{
    /* 单字枚举读在 M4 上原子；函数调用边界也阻止了编译器缓存 */
    return s_arm_state;
}

void Arm_SetState(ArmState_t new_state)
{
    taskENTER_CRITICAL();
    s_arm_state = new_state;
    taskEXIT_CRITICAL();
}

bool Arm_TryPostPick(float x, float y)
{
    bool posted = false;

    taskENTER_CRITICAL();
    if (s_arm_state == ARM_IDLE && s_arm_mailbox.type == CMD_NONE) {
        s_arm_mailbox.pick_x = x;
        s_arm_mailbox.pick_y = y;
        s_arm_mailbox.type   = CMD_TASK;   /* 最后写 type，充当发布标志 */
        posted = true;
    }
    taskEXIT_CRITICAL();
    return posted;
}

bool Arm_TryPostPlace(float x, float y)
{
    bool posted = false;

    taskENTER_CRITICAL();
    if (s_arm_state == ARM_WAIT_PLACE && s_arm_mailbox.type == CMD_NONE) {
        s_arm_mailbox.place_x = x;
        s_arm_mailbox.place_y = y;
        s_arm_mailbox.type    = CMD_PLACE;
        posted = true;
    }
    taskEXIT_CRITICAL();
    return posted;
}

bool Arm_TakeCmd(ArmCmdType_t expect, ArmState_t new_state, float *x, float *y)
{
    bool taken = false;

    taskENTER_CRITICAL();
    if (s_arm_mailbox.type == expect) {
        if (expect == CMD_TASK) {
            *x = s_arm_mailbox.pick_x;
            *y = s_arm_mailbox.pick_y;
        } else {
            *x = s_arm_mailbox.place_x;
            *y = s_arm_mailbox.place_y;
        }
        s_arm_mailbox.type = CMD_NONE;
        s_arm_state        = new_state;  /* 取件与状态切换原子完成 */
        taken = true;
    }
    taskEXIT_CRITICAL();
    return taken;
}

/* ---- TASK 协议解析结果（仅 receive_task 内读写，无需同步）---- */
bool    comm_task_ready = false;
float comm_angle1     = 0;
float comm_angle2     = 0;

bool    comm_win_flag   = false;

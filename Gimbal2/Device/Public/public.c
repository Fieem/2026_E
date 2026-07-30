#include <stdbool.h>
#include <stdint.h>
#include "main.h"
#include "public.h"
//
// Created by Administrator on 2026/7/11.
//
int32_t last_pos_yaw   = 0;
int32_t last_pos_pitch = 0;

ArmCmd_t   arm_cmd   = { .type = CMD_NONE };
ArmState_t arm_state = ARM_IDLE;

/* ---- TASK 协议解析结果 ---- */
bool    comm_task_ready = false;
int32_t comm_angle1     = 0;
int32_t comm_angle2     = 0;

bool    comm_win_flag   = false;

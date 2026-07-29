#include <stdbool.h>
#include <stdint.h>
#include "main.h"
#include "public.h"
//
// Created by Administrator on 2026/7/11.
//
int32_t last_pos_yaw   = 0;
int32_t last_pos_pitch = 0;

bool arrive_flag = false;

ArmCmd_t   arm_cmd   = { .type = CMD_NONE, .pick_x = 0, .pick_y = 0, .place_x = 0, .place_y = 0 };
ArmState_t arm_state = ARM_IDLE;

/* ---- 树莓派协议解析结果 ---- */
bool    comm_response_ready = false;
int32_t comm_pick_p1  = 0;
int32_t comm_pick_p2  = 0;
int32_t comm_place_p1 = 0;
int32_t comm_place_p2 = 0;

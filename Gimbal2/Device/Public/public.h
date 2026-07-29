//
// Created by Administrator on 2026/7/11.
//

#ifndef INC_2024E_PUBLIC_H
#define INC_2024E_PUBLIC_H
#include <stdint.h>
#include <stdbool.h>

/* ---- 全局变量 ---- */
extern int32_t last_pos_yaw;
extern int32_t last_pos_pitch;
extern bool    arrive_flag;

/* ---- 机械臂命令类型 ---- */
typedef enum {
    CMD_NONE = 0,
    CMD_EXEC,           /* 执行取子→放子完整序列       */
} ArmCmdType_t;

/* ---- 机械臂状态机 ---- */
typedef enum {
    ARM_IDLE = 0,       /* 空闲，等待新命令               */
    ARM_STARTUP_ZERO,   /* 上电回零等待                   */
    ARM_MOVING_TO_PICK, /* 移动到取子位置                  */
    ARM_SERVO_LOWERING, /* 舵机下降                       */
    ARM_MAGNET_SETTLING,/* 电磁铁吸合稳定等待             */
    ARM_SERVO_RAISING_TO_PLACE, /* 吸棋后升起               */
    ARM_PICK_ARRIVED,   /* 到位，吸合电磁铁取子            */
    ARM_MOVING_TO_PLACE,/* 移动到放子位置                  */
    ARM_PLACE_ARRIVED,  /* 到位，释放电磁铁放子            */
    ARM_SERVO_LOWERING_TO_PLACE, /* 落子前下降             */
    ARM_MAGNET_RELEASING,/* 电磁铁释放等待                 */
    ARM_SERVO_RAISING,  /* 舵机上升                       */
    ARM_MOVING_TO_ZERO, /* 移动回零点                     */
} ArmState_t;

/* ---- 命令结构体 ---- */
typedef struct {
    ArmCmdType_t type;
    int32_t      pick_x;       /* 取子位置（yaw 脉冲） */
    int32_t      pick_y;       /* 取子位置（pitch 脉冲）*/
    int32_t      place_x;      /* 放子位置（yaw 脉冲） */
    int32_t      place_y;      /* 放子位置（pitch 脉冲）*/
} ArmCmd_t;

extern ArmCmd_t   arm_cmd;
extern ArmState_t arm_state;

/* ---- 树莓派协议解析结果 ---- */
extern bool    comm_response_ready;
extern int32_t comm_pick_p1;
extern int32_t comm_pick_p2;
extern int32_t comm_place_p1;
extern int32_t comm_place_p2;

#endif //INC_2024E_PUBLIC_H

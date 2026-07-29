#ifndef SCARA_PROTOCOL_H
#define SCARA_PROTOCOL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    SCARA_FRAME_HEADER = 0xA5,
    SCARA_MAX_FLOATS = 8,
};

typedef enum {
    SCARA_CMD_HEARTBEAT = 0x0001,
    SCARA_CMD_J1_ENABLE = 0x0101,
    SCARA_CMD_J1_DISABLE = 0x0102,
    SCARA_CMD_J1_SET_ANGLE = 0x0103, // data[0]: joint angle, rad
    SCARA_CMD_J1_SET_SPEED = 0x0104, // data[0]: manual speed, rpm
    SCARA_CMD_J1_STOP = 0x0105,
    SCARA_CMD_J1_CLEAR_FAULT = 0x0106,
} ScaraCommandId;

typedef enum {
    SCARA_J1_COMMAND_ENABLE = 0,
    SCARA_J1_COMMAND_DISABLE,
    SCARA_J1_COMMAND_SET_ANGLE,
    SCARA_J1_COMMAND_SET_SPEED,
    SCARA_J1_COMMAND_STOP,
    SCARA_J1_COMMAND_CLEAR_FAULT,
    SCARA_J1_COMMAND_HEARTBEAT,
} ScaraJ1CommandType;

typedef struct {
    ScaraJ1CommandType type;
    float value;
} ScaraJ1Command;

typedef enum {
    SCARA_J1_STATE_DISABLED = 0,
    SCARA_J1_STATE_POSITION,
    SCARA_J1_STATE_SPEED,
    SCARA_J1_STATE_FAULT,
} ScaraJ1State;

typedef struct {
    ScaraJ1State state;
    bool motor_enabled;
    bool feedback_online;
    float target_angle_rad;
    float measured_angle_rad;
    float measured_speed_rpm;
    float measured_current_a;
    uint32_t last_feedback_tick_ms;
} ScaraJ1Status;

bool ScaraJ1_SubmitCommand(ScaraJ1Command command);
bool ScaraJ1_GetStatus(ScaraJ1Status *status);

#ifdef __cplusplus
}
#endif

#endif

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
    SCARA_MAX_VISION_PIECES = 4,
    SCARA_VISION_POSE_FLOATS = 6,
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
    SCARA_J1_STATE_HOMING,
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

typedef enum {
    SCARA_CMD_VISION_START = 0x0201,
    SCARA_CMD_VISION_RESULT = 0x0202,
    SCARA_CMD_VISION_ERROR = 0x0203,
    SCARA_CMD_VISION_NEXT = 0x0204,
} ScaraVisionCommandId;

typedef enum {
    SCARA_VISION_STATE_IDLE = 0,
    SCARA_VISION_STATE_WAITING,
    SCARA_VISION_STATE_READY,
    SCARA_VISION_STATE_ERROR,
} ScaraVisionState;

typedef enum {
    SCARA_VISION_ERROR_NONE = 0,
    SCARA_VISION_ERROR_FRAGMENT_COUNT = 1,
    SCARA_VISION_ERROR_NO_SOLUTION = 2,
    SCARA_VISION_ERROR_IK_UNREACHABLE = 3,
    SCARA_VISION_ERROR_INVALID_CONFIG = 4,
    SCARA_VISION_ERROR_BUSY = 5,
    SCARA_VISION_ERROR_INVALID_FRAME = 6,
    SCARA_VISION_ERROR_TIMEOUT = 7,
    SCARA_VISION_ERROR_UART = 8,
} ScaraVisionError;

typedef struct {
    float pick_j1_rad;
    float place_j1_rad;
    float pick_j2_rad;
    float place_j2_rad;
    float pick_wrist_rad;
    float place_wrist_rad;
} ScaraVisionPose;

typedef struct {
    ScaraVisionState state;
    uint16_t sequence;
    uint8_t expected_piece_count;
    uint8_t received_mask;
    ScaraVisionError error;
    ScaraVisionPose poses[SCARA_MAX_VISION_PIECES];
} ScaraVisionResult;

bool ScaraJ1_SubmitCommand(ScaraJ1Command command);
bool ScaraJ1_GetStatus(ScaraJ1Status *status);

// These functions are called from a FreeRTOS task, not an interrupt callback.
bool Vision_RequestStart(void);
bool Vision_RequestNext(void);
void Vision_Poll(void);
void Vision_OnResultFrame(uint16_t flags, const float *data, uint8_t float_num);
void Vision_OnErrorFrame(uint16_t flags, uint16_t error_code);
ScaraVisionState Vision_GetState(void);
ScaraVisionError Vision_GetError(void);
bool Vision_GetResult(ScaraVisionResult *result);

#ifdef __cplusplus
}
#endif

#endif

#ifndef GIMBAL2_LINK_H
#define GIMBAL2_LINK_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GIMBAL2_LINK_RX_RING_SIZE = 256U,
    GIMBAL2_LINK_FRAME_MAX_LEN = 128U,
};

typedef enum {
    GIMBAL2_LINK_STATE_IDLE = 0,
    GIMBAL2_LINK_STATE_READY,
    GIMBAL2_LINK_STATE_PICK,
    GIMBAL2_LINK_STATE_BUSY,
    GIMBAL2_LINK_STATE_FINISH,
    GIMBAL2_LINK_STATE_ERROR,
} Gimbal2LinkState;

typedef struct {
    Gimbal2LinkState state;
    uint32_t last_rx_tick_ms;
    bool new_flag;
    bool ready_flag;
    bool pick_flag;
    bool finish_flag;
    bool busy_flag;
    bool error_flag;
    char last_line[GIMBAL2_LINK_FRAME_MAX_LEN];
} Gimbal2LinkStatus;

extern uint8_t gimbal2_link_rx_byte;

void Gimbal2Link_Init(void);
void Gimbal2Link_Poll(void);
void Gimbal2Link_RingPush(uint8_t byte);
bool Gimbal2Link_GetStatus(Gimbal2LinkStatus *status);
bool Gimbal2Link_ClearFlags(void);

bool Gimbal2Link_SendTask(float angle1, float angle2);
bool Gimbal2Link_SendLine(const char *line);

#ifdef __cplusplus
}
#endif

#endif

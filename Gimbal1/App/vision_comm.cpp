#include <cmath>
#include <cstring>

#include "FreeRTOS.h"
#include "task.h"

#include "main.h"
#include "scara_j1_config.h"
#include "scara_protocol.h"
#include "usart.h"

namespace {

using namespace scara::j1_config;

constexpr uint16_t kSequenceMask = 0x0FFFU;
constexpr uint16_t kCountMask = 0x3U;
constexpr uint16_t kIndexMask = 0x3U;
constexpr uint8_t kFrameLengthWithoutData = 4U;

ScaraVisionState vision_state = SCARA_VISION_STATE_IDLE;
ScaraVisionError vision_error = SCARA_VISION_ERROR_NONE;
ScaraVisionResult vision_result{};
uint16_t next_sequence = 0U;
uint32_t vision_started_tick = 0U;

uint16_t sequenceFromFlags(uint16_t flags) {
    return flags & kSequenceMask;
}

uint8_t countFromFlags(uint16_t flags) {
    const uint8_t code = static_cast<uint8_t>((flags >> 12U) & kCountMask);
    return code == 0U ? 0U : static_cast<uint8_t>(code + 1U);
}

uint8_t indexFromFlags(uint16_t flags) {
    return static_cast<uint8_t>((flags >> 14U) & kIndexMask);
}

uint8_t contiguousReceivedCount(uint8_t mask) {
    uint8_t count = 0U;
    while ((mask & (1U << count)) != 0U && count < SCARA_MAX_VISION_PIECES) {
        ++count;
    }
    return count;
}

bool sendVisionCommand(const uint16_t cmd_id, const uint16_t flags) {
    const uint8_t frame[2U + kFrameLengthWithoutData] = {
        SCARA_FRAME_HEADER,
        kFrameLengthWithoutData,
        static_cast<uint8_t>(cmd_id & 0xFFU),
        static_cast<uint8_t>((cmd_id >> 8U) & 0xFFU),
        static_cast<uint8_t>(flags & 0xFFU),
        static_cast<uint8_t>((flags >> 8U) & 0xFFU),
    };
    return HAL_UART_Transmit(&huart1, const_cast<uint8_t *>(frame), sizeof(frame), 100U) == HAL_OK;
}

bool validJointAngle(float value, float minimum, float maximum) {
    return std::isfinite(value) && value >= minimum && value <= maximum;
}

bool validPose(const float *data) {
    // 数据顺序：拾取 J1、放置 J1、拾取 J2、放置 J2、拾取腕部角、放置腕部角。
    return validJointAngle(data[0], kMinJointAngleRad, kMaxJointAngleRad) &&
           validJointAngle(data[1], kMinJointAngleRad, kMaxJointAngleRad) &&
           validJointAngle(data[2], kMinJ2JointAngleRad, kMaxJ2JointAngleRad) &&
           validJointAngle(data[3], kMinJ2JointAngleRad, kMaxJ2JointAngleRad) &&
           std::isfinite(data[4]) &&
           std::isfinite(data[5]);
}

void setError(ScaraVisionError error) {
    vision_error = error;
    vision_state = SCARA_VISION_STATE_ERROR;
    vision_result.state = SCARA_VISION_STATE_ERROR;
    vision_result.error = error;
}

} // namespace

extern "C" bool Vision_RequestStart(void) {
    if (vision_state == SCARA_VISION_STATE_WAITING) return false;

    uint16_t sequence = static_cast<uint16_t>((next_sequence + 1U) & kSequenceMask);
    if (sequence == 0U) sequence = 1U;
    next_sequence = sequence;

    vision_state = SCARA_VISION_STATE_WAITING;
    vision_error = SCARA_VISION_ERROR_NONE;
    vision_result = ScaraVisionResult{};
    vision_result.state = SCARA_VISION_STATE_WAITING;
    vision_result.sequence = sequence;
    vision_started_tick = HAL_GetTick();

    if (!sendVisionCommand(SCARA_CMD_VISION_START, sequence)) {
        setError(SCARA_VISION_ERROR_UART);
        return false;
    }
    return true;
}

extern "C" bool Vision_RequestNext(void) {
    if (vision_result.sequence == 0U || vision_state == SCARA_VISION_STATE_WAITING) {
        return false;
    }
    if (vision_result.expected_piece_count == 0U) {
        return false;
    }

    const uint8_t next_piece_index = contiguousReceivedCount(vision_result.received_mask);
    if (next_piece_index >= vision_result.expected_piece_count) {
        return false;
    }

    const uint16_t flags = static_cast<uint16_t>(
        vision_result.sequence | (static_cast<uint16_t>(next_piece_index) << 14U));

    vision_state = SCARA_VISION_STATE_WAITING;
    vision_result.state = SCARA_VISION_STATE_WAITING;
    vision_started_tick = HAL_GetTick();
    if (!sendVisionCommand(SCARA_CMD_VISION_NEXT, flags)) {
        setError(SCARA_VISION_ERROR_UART);
        return false;
    }
    return true;
}

extern "C" void Vision_Poll(void) {
    if (vision_state == SCARA_VISION_STATE_WAITING &&
        (HAL_GetTick() - vision_started_tick) > kVisionTimeoutMs) {
        setError(SCARA_VISION_ERROR_TIMEOUT);
    }
}

extern "C" void Vision_OnResultFrame(
    const uint16_t flags, const float *data, const uint8_t float_num) {
    if (data == nullptr || float_num != SCARA_VISION_POSE_FLOATS) return;
    if (vision_state != SCARA_VISION_STATE_WAITING) return;

    const uint16_t sequence = sequenceFromFlags(flags);
    const uint8_t piece_count = countFromFlags(flags);
    const uint8_t piece_index = indexFromFlags(flags);
    const uint8_t expected_piece_index = contiguousReceivedCount(vision_result.received_mask);
    if (sequence != vision_result.sequence ||
        piece_count < 2U || piece_count > SCARA_MAX_VISION_PIECES ||
        piece_index >= piece_count || piece_index != expected_piece_index ||
        !validPose(data)) {
        setError(SCARA_VISION_ERROR_INVALID_FRAME);
        return;
    }

    if (vision_result.expected_piece_count == 0U) {
        vision_result.expected_piece_count = piece_count;
    }
    if (vision_result.expected_piece_count != piece_count ||
        (vision_result.received_mask & (1U << piece_index)) != 0U) {
        setError(SCARA_VISION_ERROR_INVALID_FRAME);
        return;
    }

    std::memcpy(
        &vision_result.poses[piece_index], data,
        sizeof(ScaraVisionPose));
    vision_result.received_mask = static_cast<uint8_t>(
        vision_result.received_mask | (1U << piece_index));

    vision_state = SCARA_VISION_STATE_READY;
    vision_result.state = SCARA_VISION_STATE_READY;
}

extern "C" void Vision_OnErrorFrame(
    const uint16_t flags, const uint16_t error_code) {
    if (vision_state != SCARA_VISION_STATE_WAITING ||
        sequenceFromFlags(flags) != vision_result.sequence) {
        return;
    }
    vision_error = static_cast<ScaraVisionError>(error_code);
    vision_state = SCARA_VISION_STATE_ERROR;
    vision_result.state = SCARA_VISION_STATE_ERROR;
    vision_result.error = vision_error;
}

extern "C" ScaraVisionState Vision_GetState(void) {
    return vision_state;
}

extern "C" ScaraVisionError Vision_GetError(void) {
    return vision_error;
}

extern "C" bool Vision_GetResult(ScaraVisionResult *result) {
    if (result == nullptr || vision_state != SCARA_VISION_STATE_READY) return false;
    taskENTER_CRITICAL();
    *result = vision_result;
    result->state = vision_state;
    result->error = vision_error;
    vision_state = SCARA_VISION_STATE_IDLE;
    vision_result.state = SCARA_VISION_STATE_IDLE;
    taskEXIT_CRITICAL();
    return true;
}

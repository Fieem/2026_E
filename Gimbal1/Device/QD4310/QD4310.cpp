#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include "QD4310.h"

namespace {
constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

uint16_t readU16Le(const uint8_t *data) {
    return static_cast<uint16_t>(data[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8U);
}

int16_t readI16Le(const uint8_t *data) {
    return static_cast<int16_t>(readU16Le(data));
}
}

bool QD4310::sendCommand(const Command cmd, const uint16_t raw_value) {
    if (hfdcan_ == nullptr) {
        last_tx_status_ = HAL_ERROR;
        return false;
    }

    uint8_t tx_buffer[3]{};

    tx_buffer[0] = static_cast<uint8_t>(cmd);
    tx_buffer[1] = static_cast<uint8_t>(raw_value & 0xFFU);
    tx_buffer[2] = static_cast<uint8_t>((raw_value >> 8U) & 0xFFU);

    FDCAN_TxHeaderTypeDef tx_header{};
    tx_header.Identifier = commandCanId();
    tx_header.IdType = FDCAN_STANDARD_ID;
    tx_header.TxFrameType = FDCAN_DATA_FRAME;
    tx_header.DataLength = FDCAN_DLC_BYTES_3;
    tx_header.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
    tx_header.BitRateSwitch = FDCAN_BRS_OFF;
    tx_header.FDFormat = FDCAN_CLASSIC_CAN;
    tx_header.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
    tx_header.MessageMarker = 0;

    last_tx_status_ = HAL_FDCAN_AddMessageToTxFifoQ(hfdcan_, &tx_header, tx_buffer);
    return last_tx_status_ == HAL_OK;
}

bool QD4310::enable() {
    return sendCommand(Command::ENABLE, 0U);
}

bool QD4310::disable() {
    return sendCommand(Command::DISABLE, 0U);
}

bool QD4310::update(const uint8_t feedback[8]) {
    if (feedback == nullptr) return false;

    enabled = (feedback[0] & 0x01U) != 0U;
    current = static_cast<float>(readI16Le(feedback + 2)) * 10.0f /
              static_cast<float>(std::numeric_limits<int16_t>::max());
    speed = static_cast<float>(readI16Le(feedback + 4)) * 1000.0f /
            static_cast<float>(std::numeric_limits<int16_t>::max());
    angle = static_cast<float>(readU16Le(feedback + 6)) * kTwoPi /
            static_cast<float>(std::numeric_limits<uint16_t>::max());
    last_feedback_tick_ = HAL_GetTick();
    return true;
}

bool QD4310::setAngle(float angle_rad) {
    if (!std::isfinite(angle_rad)) return false;

    angle_rad = std::clamp(angle_rad, 0.0f, kTwoPi);
    const auto raw = static_cast<uint16_t>(std::lround(
        angle_rad / kTwoPi * static_cast<float>(std::numeric_limits<uint16_t>::max())));
    return sendCommand(Command::ANGLE, raw);
}

bool QD4310::setSpeed(float speed_rpm) {
    if (!std::isfinite(speed_rpm)) return false;

    speed_rpm = std::clamp(speed_rpm, -1000.0f, 1000.0f);
    const auto raw = static_cast<int16_t>(std::lround(
        speed_rpm / 1000.0f * static_cast<float>(std::numeric_limits<int16_t>::max())));
    return sendCommand(Command::SPEED, static_cast<uint16_t>(raw));
}

bool QD4310::setLowSpeed(float speed_rpm) {
    if (!std::isfinite(speed_rpm)) return false;

    speed_rpm = std::clamp(speed_rpm, -1000.0f, 1000.0f);
    const auto raw = static_cast<int16_t>(std::lround(
        speed_rpm / 1000.0f * static_cast<float>(std::numeric_limits<int16_t>::max())));
    return sendCommand(Command::LOW_SPEED, static_cast<uint16_t>(raw));
}

bool QD4310::setCurrent(float current_a) {
    if (!std::isfinite(current_a)) return false;

    current_a = std::clamp(current_a, -10.0f, 10.0f);
    const auto raw = static_cast<int16_t>(std::lround(
        current_a / 10.0f * static_cast<float>(std::numeric_limits<int16_t>::max())));
    return sendCommand(Command::CURRENT, static_cast<uint16_t>(raw));
}

bool QD4310::feedbackFresh(const uint32_t now_ms, const uint32_t timeout_ms) const {
    return last_feedback_tick_ != 0U && (now_ms - last_feedback_tick_) <= timeout_ms;
}

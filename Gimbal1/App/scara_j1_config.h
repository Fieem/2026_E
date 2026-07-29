#ifndef SCARA_J1_CONFIG_H
#define SCARA_J1_CONFIG_H

#include <cstdint>
#include <numbers>

namespace scara::j1_config {

// QD4310 node ID 0x01: command CAN ID 0x401, feedback CAN ID 0x501.
inline constexpr uint8_t kMotorNodeId = 0x01;

// These values describe the mechanical J1 joint and should be calibrated
// after the arm, reducer and hard stops are installed.
inline constexpr float kZeroOffsetRad = 0.0f;
inline constexpr bool kDirectionInverted = false;
inline constexpr float kMinJointAngleRad = -2.8f;
inline constexpr float kMaxJointAngleRad = 2.8f;
inline constexpr float kMaxManualSpeedRpm = 100.0f;

inline constexpr uint32_t kControlPeriodMs = 5U;
inline constexpr uint32_t kFeedbackTimeoutMs = 300U;
inline constexpr uint32_t kHostWatchdogMs = 1500U;

inline constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

} // namespace scara::j1_config

#endif

#ifndef SCARA_J1_CONFIG_H
#define SCARA_J1_CONFIG_H

#include <cstdint>
#include <numbers>

namespace scara::j1_config {

// QD4310 node ID 0x01: command CAN ID 0x401, feedback CAN ID 0x501.
inline constexpr uint8_t kMotorNodeId = 0x01;

// Calibrate these values after the arm, reducer and hard stops are installed.
inline constexpr float kZeroOffsetRad = 0.0f;
inline constexpr bool kDirectionInverted = false;
inline constexpr float kMinJointAngleRad = -3.1416f;
inline constexpr float kMaxJointAngleRad = 3.1416f;
inline constexpr float kMinJ2JointAngleRad = -3.1416f;
inline constexpr float kMaxJ2JointAngleRad = 3.1416f;
inline constexpr float kMaxManualSpeedRpm = 100.0f;

// Power-on homing drives J1 to the configured joint zero before normal control.
inline constexpr bool kStartupHomeEnabled = true;
inline constexpr float kStartupHomeJointAngleRad = 0.0f;
inline constexpr float kStartupHomeToleranceRad = 0.03f;
inline constexpr uint32_t kStartupHomeStableMs = 200U;
inline constexpr uint32_t kStartupHomeTimeoutMs = 4000U;

inline constexpr uint32_t kControlPeriodMs = 5U;
inline constexpr uint32_t kFeedbackTimeoutMs = 300U;
inline constexpr uint32_t kHostWatchdogMs = 1500U;
inline constexpr uint32_t kVisionTimeoutMs = 30000U;
inline constexpr uint32_t kVisionStartMinExecuteDelayMs = 3000U;
inline constexpr float kExecuteJ1ToleranceRad = 0.03f;

inline constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

} // namespace scara::j1_config

#endif

#include <algorithm>
#include <cmath>

#include "FreeRTOS.h"
#include "cmsis_os.h"
#include "fdcan.h"
#include "gimbal2_link.h"
#include "queue.h"
#include "task.h"

#include "QD4310.h"
#include "scara_j1_config.h"
#include "scara_protocol.h"

namespace {

using namespace scara::j1_config;

QD4310 j1_motor(&hfdcan1, kMotorNodeId);
QueueHandle_t j1_command_queue = nullptr;

ScaraJ1State j1_state = SCARA_J1_STATE_DISABLED;
float target_joint_angle_rad = 0.0f;
uint32_t last_host_command_tick = 0U;
uint32_t active_since_tick = 0U;
bool startup_homing_active = false;
bool startup_home_command_sent = false;
uint32_t startup_home_started_tick = 0U;
uint32_t startup_home_stable_since_tick = 0U;

enum class ExecuteState : uint8_t {
    Idle = 0,
    MoveJ1ToPick,
    WaitJ1Pick,
    WaitGimbal2Pick,
    MoveJ1ToPlace,
    WaitJ1Place,
    WaitGimbal2Finish,
};

ExecuteState execute_state = ExecuteState::Idle;
ScaraVisionResult execute_result{};
uint8_t execute_piece_index = 0U;
uint16_t last_completed_sequence = 0U;
uint32_t execute_stage_started_tick = 0U;
uint32_t vision_start_min_execute_tick = 0U;

float wrapToTwoPi(float angle_rad) {
    angle_rad = std::fmod(angle_rad, kTwoPi);
    return angle_rad < 0.0f ? angle_rad + kTwoPi : angle_rad;
}

float wrapToPi(float angle_rad) {
    angle_rad = wrapToTwoPi(angle_rad + std::numbers::pi_v<float>);
    return angle_rad - std::numbers::pi_v<float>;
}

float jointToMotorAngle(float joint_angle_rad) {
    const float direction = kDirectionInverted ? -1.0f : 1.0f;
    return wrapToTwoPi(kZeroOffsetRad + direction * joint_angle_rad);
}

float motorToJointAngle(float motor_angle_rad) {
    const float direction = kDirectionInverted ? -1.0f : 1.0f;
    return wrapToPi(direction * (motor_angle_rad - kZeroOffsetRad));
}

bool isActive() {
    return j1_state == SCARA_J1_STATE_HOMING ||
           j1_state == SCARA_J1_STATE_POSITION ||
           j1_state == SCARA_J1_STATE_SPEED;
}

bool hostWatchdogEnabled() {
    return false;
}

void disableMotor(ScaraJ1State next_state) {
    (void)j1_motor.setCurrent(0.0f);
    (void)j1_motor.disable();
    j1_state = next_state;
}

void enterFault() {
    disableMotor(SCARA_J1_STATE_FAULT);
    startup_homing_active = false;
    startup_home_command_sent = false;
    startup_home_stable_since_tick = 0U;
    execute_state = ExecuteState::Idle;
    execute_piece_index = 0U;
    execute_stage_started_tick = 0U;
}

bool enableAtZeroSpeed(uint32_t now_ms) {
    const bool was_active = isActive();
    if (!j1_motor.enable()) return false;
    if (!j1_motor.setSpeed(0.0f)) return false;

    j1_state = SCARA_J1_STATE_SPEED;
    if (!was_active) active_since_tick = now_ms;
    return true;
}

void processCommand(const ScaraJ1Command &command, uint32_t now_ms) {
    if (startup_homing_active) return;

    switch (command.type) {
    case SCARA_J1_COMMAND_ENABLE:
        if (j1_state == SCARA_J1_STATE_FAULT) break;
        if (!enableAtZeroSpeed(now_ms)) enterFault();
        break;

    case SCARA_J1_COMMAND_DISABLE:
        disableMotor(SCARA_J1_STATE_DISABLED);
        break;

    case SCARA_J1_COMMAND_SET_ANGLE: {
        if (j1_state == SCARA_J1_STATE_FAULT || !std::isfinite(command.value)) break;
        const bool was_active = isActive();

        target_joint_angle_rad = std::clamp(
            command.value, kMinJointAngleRad, kMaxJointAngleRad);

        if (!j1_motor.enable() ||
            !j1_motor.setAngle(jointToMotorAngle(target_joint_angle_rad))) {
            enterFault();
            break;
        }

        j1_state = SCARA_J1_STATE_POSITION;
        if (!was_active) active_since_tick = now_ms;
        break;
    }

    case SCARA_J1_COMMAND_SET_SPEED: {
        if (j1_state == SCARA_J1_STATE_FAULT || !std::isfinite(command.value)) break;
        const bool was_active = isActive();

        const float speed_rpm = std::clamp(
            command.value, -kMaxManualSpeedRpm, kMaxManualSpeedRpm);
        if (!j1_motor.enable() || !j1_motor.setLowSpeed(speed_rpm)) {
            enterFault();
            break;
        }

        j1_state = SCARA_J1_STATE_SPEED;
        if (!was_active) active_since_tick = now_ms;
        break;
    }

    case SCARA_J1_COMMAND_STOP:
        disableMotor(SCARA_J1_STATE_DISABLED);
        break;

    case SCARA_J1_COMMAND_CLEAR_FAULT:
        disableMotor(SCARA_J1_STATE_DISABLED);
        break;

    case SCARA_J1_COMMAND_HEARTBEAT:
        break;

    default:
        break;
    }
}

float radToDegrees(float angle_rad) {
    return angle_rad * 180.0f / std::numbers::pi_v<float>;
}

bool j1AtTarget(float target_rad) {
    if (!j1_motor.feedbackFresh(HAL_GetTick(), kFeedbackTimeoutMs)) return false;
    const float measured_joint_angle = motorToJointAngle(j1_motor.angle);
    return std::fabs(measured_joint_angle - target_rad) <= kExecuteJ1ToleranceRad;
}

void commandJ1JointAngle(float joint_angle_rad, uint32_t now_ms) {
    processCommand(
        ScaraJ1Command{SCARA_J1_COMMAND_SET_ANGLE, joint_angle_rad}, now_ms);
}

void startExecuteStage(ExecuteState next_state, uint32_t now_ms) {
    execute_state = next_state;
    execute_stage_started_tick = now_ms;
}

void finishSequence(void) {
    last_completed_sequence = execute_result.sequence;
    execute_result = ScaraVisionResult{};
    execute_piece_index = 0U;
    execute_stage_started_tick = 0U;
    vision_start_min_execute_tick = 0U;
    execute_state = ExecuteState::Idle;
}

void startExecution(const ScaraVisionResult &result, uint32_t now_ms) {
    execute_result = result;
    execute_piece_index = 0U;
    while (execute_piece_index < execute_result.expected_piece_count &&
           (execute_result.received_mask & (1U << execute_piece_index)) != 0U) {
        ++execute_piece_index;
    }
    if (execute_piece_index > 0U) {
        --execute_piece_index;
    }
    Gimbal2Link_ClearFlags();
    startExecuteStage(ExecuteState::MoveJ1ToPick, now_ms);
}

void updateVisionStartTrigger(uint32_t now_ms) {
    if (startup_homing_active || j1_state == SCARA_J1_STATE_FAULT) return;
    if (execute_state != ExecuteState::Idle) return;

    Gimbal2LinkStatus gimbal2_status{};
    (void)Gimbal2Link_GetStatus(&gimbal2_status);
    if (!gimbal2_status.ready_flag) return;
    if (Vision_GetState() == SCARA_VISION_STATE_WAITING) return;

    if (Vision_RequestStart()) {
        vision_start_min_execute_tick = now_ms + kVisionStartMinExecuteDelayMs;
    }
    Gimbal2Link_ClearFlags();
    last_host_command_tick = now_ms;
}

void updateExecution(uint32_t now_ms) {
    if (startup_homing_active || j1_state == SCARA_J1_STATE_FAULT) return;

    if (execute_state == ExecuteState::Idle) {
        if (Vision_GetState() != SCARA_VISION_STATE_READY) return;
        if (vision_start_min_execute_tick != 0U &&
            static_cast<int32_t>(now_ms - vision_start_min_execute_tick) < 0) {
            return;
        }

        ScaraVisionResult pending{};
        if (!Vision_GetResult(&pending)) return;
        if (pending.sequence == 0U || pending.sequence == last_completed_sequence) return;

        vision_start_min_execute_tick = 0U;
        startExecution(pending, now_ms);
    }

    last_host_command_tick = now_ms;

    Gimbal2LinkStatus gimbal2_status{};
    (void)Gimbal2Link_GetStatus(&gimbal2_status);
    if (gimbal2_status.error_flag) {
        enterFault();
        return;
    }

    if (execute_piece_index >= execute_result.expected_piece_count ||
        execute_piece_index >= SCARA_MAX_VISION_PIECES) {
        finishSequence();
        return;
    }

    const ScaraVisionPose &pose = execute_result.poses[execute_piece_index];

    switch (execute_state) {
    case ExecuteState::Idle:
        break;

    case ExecuteState::MoveJ1ToPick:
        Gimbal2Link_ClearFlags();
        if (!Gimbal2Link_SendTask(
                radToDegrees(pose.pick_j2_rad),
                radToDegrees(pose.pick_wrist_rad))) {
            enterFault();
            return;
        }
        commandJ1JointAngle(pose.pick_j1_rad, now_ms);
        startExecuteStage(ExecuteState::WaitGimbal2Pick, now_ms);
        break;

    case ExecuteState::WaitJ1Pick:
        startExecuteStage(ExecuteState::WaitGimbal2Pick, now_ms);
        break;

    case ExecuteState::WaitGimbal2Pick:
        if (!gimbal2_status.pick_flag) break;
        Gimbal2Link_ClearFlags();
        startExecuteStage(ExecuteState::MoveJ1ToPlace, now_ms);
        break;

    case ExecuteState::MoveJ1ToPlace:
        Gimbal2Link_ClearFlags();
        if (!Gimbal2Link_SendTask(
                radToDegrees(pose.place_j2_rad),
                radToDegrees(pose.place_wrist_rad))) {
            enterFault();
            return;
        }
        commandJ1JointAngle(pose.place_j1_rad, now_ms);
        startExecuteStage(ExecuteState::WaitGimbal2Finish, now_ms);
        break;

    case ExecuteState::WaitJ1Place:
        startExecuteStage(ExecuteState::WaitGimbal2Finish, now_ms);
        break;

    case ExecuteState::WaitGimbal2Finish:
        if (!gimbal2_status.finish_flag) break;
        Gimbal2Link_ClearFlags();
        if ((execute_piece_index + 1U) >= execute_result.expected_piece_count) {
            finishSequence();
        } else {
            execute_result = ScaraVisionResult{};
            execute_piece_index = 0U;
            execute_stage_started_tick = 0U;
            execute_state = ExecuteState::Idle;
            if (!Vision_RequestNext()) {
                enterFault();
                return;
            }
        }
        break;
    }
}

void startStartupHome(uint32_t now_ms) {
    startup_homing_active = true;
    startup_home_command_sent = false;
    startup_home_started_tick = now_ms;
    startup_home_stable_since_tick = 0U;
    target_joint_angle_rad = kStartupHomeJointAngleRad;
    j1_state = SCARA_J1_STATE_HOMING;
}

void updateStartupHome(uint32_t now_ms) {
    if (!startup_homing_active) return;

    if ((now_ms - startup_home_started_tick) > kStartupHomeTimeoutMs) {
        startup_homing_active = false;
        startup_home_command_sent = false;
        startup_home_stable_since_tick = 0U;
        j1_state = SCARA_J1_STATE_POSITION;
        return;
    }

    if (!startup_home_command_sent) {
        if (!j1_motor.enable() ||
            !j1_motor.setAngle(jointToMotorAngle(kStartupHomeJointAngleRad))) {
            enterFault();
            return;
        }
        startup_home_command_sent = true;
        active_since_tick = now_ms;
    }

    if (!j1_motor.feedbackFresh(now_ms, kFeedbackTimeoutMs)) return;

    const float measured_joint_angle = motorToJointAngle(j1_motor.angle);
    const float error_rad = std::fabs(measured_joint_angle - kStartupHomeJointAngleRad);
    if (error_rad > kStartupHomeToleranceRad) {
        startup_home_stable_since_tick = 0U;
        return;
    }

    if (startup_home_stable_since_tick == 0U) {
        startup_home_stable_since_tick = now_ms;
        return;
    }

    if ((now_ms - startup_home_stable_since_tick) < kStartupHomeStableMs) return;

    startup_homing_active = false;
    startup_home_command_sent = false;
    startup_home_stable_since_tick = 0U;
    j1_state = SCARA_J1_STATE_POSITION;
}

void monitorSafety(uint32_t now_ms) {
    (void)now_ms;
}

void CAN_InterfaceInit() {
    FDCAN_FilterTypeDef filter{};
    filter.IdType = FDCAN_STANDARD_ID;
    filter.FilterIndex = 0;
    filter.FilterType = FDCAN_FILTER_MASK;
    filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    filter.FilterID1 = j1_motor.feedbackCanId();
    filter.FilterID2 = 0x7FFU;

    if (HAL_FDCAN_ConfigFilter(&hfdcan1, &filter) != HAL_OK) Error_Handler();
    if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK) Error_Handler();
    if (HAL_FDCAN_ActivateNotification(
            &hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0U) != HAL_OK) {
        Error_Handler();
    }
}

} // namespace

extern "C" bool ScaraJ1_SubmitCommand(const ScaraJ1Command command) {
    if (j1_command_queue == nullptr) return false;
    return xQueueSend(j1_command_queue, &command, 0U) == pdPASS;
}

extern "C" bool ScaraJ1_GetStatus(ScaraJ1Status *status) {
    if (status == nullptr) return false;

    taskENTER_CRITICAL();
    status->state = j1_state;
    status->motor_enabled = j1_motor.enabled;
    status->feedback_online = j1_motor.feedbackFresh(
        HAL_GetTick(), kFeedbackTimeoutMs);
    status->target_angle_rad = target_joint_angle_rad;
    status->measured_angle_rad = motorToJointAngle(j1_motor.angle);
    status->measured_speed_rpm = j1_motor.speed;
    status->measured_current_a = j1_motor.current;
    status->last_feedback_tick_ms = j1_motor.lastFeedbackTick();
    taskEXIT_CRITICAL();
    return true;
}

extern "C" void StartGimbalTask(void *argument) {
    (void)argument;

    j1_command_queue = xQueueCreate(12U, sizeof(ScaraJ1Command));
    if (j1_command_queue == nullptr) Error_Handler();

    CAN_InterfaceInit();
    last_host_command_tick = HAL_GetTick();
    if (kStartupHomeEnabled) {
        startStartupHome(last_host_command_tick);
    }

    TickType_t last_wake_tick = xTaskGetTickCount();
    for (;;) {
        const uint32_t now_ms = HAL_GetTick();
        ScaraJ1Command command{};
        while (xQueueReceive(j1_command_queue, &command, 0U) == pdPASS) {
            processCommand(command, now_ms);
        }

        updateStartupHome(now_ms);
        updateVisionStartTrigger(now_ms);
        updateExecution(now_ms);
        Vision_Poll();
        monitorSafety(now_ms);
        vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(kControlPeriodMs));
    }
}

extern "C" void HAL_FDCAN_RxFifo0Callback(
    FDCAN_HandleTypeDef *hfdcan, uint32_t rx_fifo0_its) {
    (void)rx_fifo0_its;
    if (hfdcan != &hfdcan1) return;

    while (HAL_FDCAN_GetRxFifoFillLevel(hfdcan, FDCAN_RX_FIFO0) > 0U) {
        FDCAN_RxHeaderTypeDef rx_header{};
        uint8_t rx_data[8]{};
        if (HAL_FDCAN_GetRxMessage(
                hfdcan, FDCAN_RX_FIFO0, &rx_header, rx_data) != HAL_OK) {
            break;
        }

        if (rx_header.IdType == FDCAN_STANDARD_ID &&
            rx_header.Identifier == j1_motor.feedbackCanId() &&
            rx_header.DataLength == FDCAN_DLC_BYTES_8) {
            (void)j1_motor.update(rx_data);
        }
    }
}

#ifndef QD4310_H
#define QD4310_H

#include <cstdint>
#include "fdcan.h"

class QD4310 {
public:
    static constexpr uint16_t kCommandIdBase = 0x400;
    static constexpr uint16_t kFeedbackIdBase = 0x500;

    explicit QD4310(FDCAN_HandleTypeDef *hfdcan, uint8_t node_id) :
        node_id_(node_id), hfdcan_(hfdcan) {}

    bool enable();
    bool disable();
    bool update(const uint8_t feedback[8]);

    /**
     * @brief 设置电机角度
     * @param _angle 设置的角度,[0,2pi]
     */
    bool setAngle(float angle_rad);
    /**
     * @brief 设置电机转速
     * @param _speed 设置的转速,[-1000,1000]
     */
    bool setSpeed(float speed_rpm);
    /**
     * @brief 设置电机转速
     * @param _speed 设置的转速,[-1000,1000]
     */
    bool setLowSpeed(float speed_rpm);
    /**
     * @brief 设置电机电流
     * @param _current 设置的转速,[-10,10]
     */
    bool setCurrent(float current_a);

    [[nodiscard]] uint8_t nodeId() const { return node_id_; }
    [[nodiscard]] uint16_t commandCanId() const { return kCommandIdBase + node_id_; }
    [[nodiscard]] uint16_t feedbackCanId() const { return kFeedbackIdBase + node_id_; }
    [[nodiscard]] bool feedbackFresh(uint32_t now_ms, uint32_t timeout_ms) const;
    [[nodiscard]] uint32_t lastFeedbackTick() const { return last_feedback_tick_; }
    [[nodiscard]] HAL_StatusTypeDef lastTxStatus() const { return last_tx_status_; }

    bool enabled{};
    float speed{};   // in rpm
    float angle{};   // in rad
    float current{}; // in A
private:
    enum class Command :uint8_t {
        NOP = 0x00,
        ENABLE = 0x01,
        DISABLE = 0x02,
        CURRENT = 0x03,
        SPEED = 0x04,
        ANGLE = 0x05,
        LOW_SPEED = 0x06
    };

    uint8_t node_id_{};
    FDCAN_HandleTypeDef *hfdcan_{};
    volatile uint32_t last_feedback_tick_{};
    HAL_StatusTypeDef last_tx_status_{HAL_OK};

    bool sendCommand(Command cmd, uint16_t raw_value);
};

#endif

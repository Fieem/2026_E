#include <cmath>
#include <cstring>

#include "FreeRTOS.h"
#include "queue.h"

#include "main.h"
#include "scara_protocol.h"
#include "usart.h"

namespace {

constexpr uint8_t kFrameFixedBytes = 4U; // cmd_id + flags
constexpr uint8_t kMaxPayloadBytes = kFrameFixedBytes + SCARA_MAX_FLOATS * sizeof(float);
constexpr uint32_t kInterByteTimeoutMs = 50U;

struct ReceiveFrame {
    uint16_t cmd_id;
    uint16_t flags;
    uint8_t float_num;
    float data[SCARA_MAX_FLOATS];
};

enum class RxState : uint8_t {
    WaitHeader,
    WaitLength,
    ReceivePayload,
};

struct RawPacket {
    uint8_t data[kMaxPayloadBytes];
    uint8_t len;
};

QueueHandle_t uart_queue = nullptr;
uint8_t uart1_rx_byte[1]{};

bool submit(ScaraJ1CommandType type, float value = 0.0f) {
    return ScaraJ1_SubmitCommand(ScaraJ1Command{type, value});
}

bool decodeFrame(const RawPacket &packet, ReceiveFrame &frame) {
    if (packet.len < kFrameFixedBytes ||
        ((packet.len - kFrameFixedBytes) % sizeof(float)) != 0U) {
        return false;
    }

    frame.cmd_id = static_cast<uint16_t>(packet.data[0]) |
                   static_cast<uint16_t>(packet.data[1] << 8U);
    frame.flags = static_cast<uint16_t>(packet.data[2]) |
                  static_cast<uint16_t>(packet.data[3] << 8U);
    frame.float_num = static_cast<uint8_t>(
        (packet.len - kFrameFixedBytes) / sizeof(float));
    if (frame.float_num > SCARA_MAX_FLOATS) return false;

    std::memset(frame.data, 0, sizeof(frame.data));
    std::memcpy(frame.data, packet.data + kFrameFixedBytes,
                frame.float_num * sizeof(float));

    for (uint8_t i = 0U; i < frame.float_num; ++i) {
        if (!std::isfinite(frame.data[i])) return false;
    }
    return true;
}

void processFrame(const ReceiveFrame &frame) {
    (void)frame.flags;

    switch (frame.cmd_id) {
    case SCARA_CMD_HEARTBEAT:
        (void)submit(SCARA_J1_COMMAND_HEARTBEAT);
        break;
    case SCARA_CMD_J1_ENABLE:
        (void)submit(SCARA_J1_COMMAND_ENABLE);
        break;
    case SCARA_CMD_J1_DISABLE:
        (void)submit(SCARA_J1_COMMAND_DISABLE);
        break;
    case SCARA_CMD_J1_SET_ANGLE:
        if (frame.float_num >= 1U) {
            (void)submit(SCARA_J1_COMMAND_SET_ANGLE, frame.data[0]);
        }
        break;
    case SCARA_CMD_J1_SET_SPEED:
        if (frame.float_num >= 1U) {
            (void)submit(SCARA_J1_COMMAND_SET_SPEED, frame.data[0]);
        }
        break;
    case SCARA_CMD_J1_STOP:
        (void)submit(SCARA_J1_COMMAND_STOP);
        break;
    case SCARA_CMD_J1_CLEAR_FAULT:
        (void)submit(SCARA_J1_COMMAND_CLEAR_FAULT);
        break;
    default:
        break;
    }
}

} // namespace

extern "C" void StartReceiveTask(void *argument) {
    (void)argument;

    RawPacket raw_packet{};
    ReceiveFrame frame{};

    uart_queue = xQueueCreate(10U, sizeof(RawPacket));
    if (uart_queue == nullptr) Error_Handler();

    if (HAL_UART_Receive_IT(&huart1, uart1_rx_byte, 1U) != HAL_OK) Error_Handler();

    for (;;) {
        if (xQueueReceive(uart_queue, &raw_packet, portMAX_DELAY) == pdPASS &&
            decodeFrame(raw_packet, frame)) {
            processFrame(frame);
        }
    }
}

extern "C" void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    static RxState state = RxState::WaitHeader;
    static RawPacket packet{};
    static uint8_t rx_count = 0U;
    static uint32_t last_byte_tick = 0U;

    if (huart->Instance == USART1) {
        const uint8_t byte = uart1_rx_byte[0];
        const uint32_t now_ms = HAL_GetTick();
        BaseType_t higher_priority_task_woken = pdFALSE;

        if ((now_ms - last_byte_tick) > kInterByteTimeoutMs) {
            state = RxState::WaitHeader;
            rx_count = 0U;
        }
        last_byte_tick = now_ms;

        switch (state) {
        case RxState::WaitHeader:
            if (byte == SCARA_FRAME_HEADER) state = RxState::WaitLength;
            break;

        case RxState::WaitLength:
            if (byte >= kFrameFixedBytes && byte <= kMaxPayloadBytes) {
                packet.len = byte;
                rx_count = 0U;
                state = RxState::ReceivePayload;
            } else if (byte != SCARA_FRAME_HEADER) {
                state = RxState::WaitHeader;
            }
            break;

        case RxState::ReceivePayload:
            packet.data[rx_count++] = byte;
            if (rx_count >= packet.len) {
                if (uart_queue != nullptr) {
                    (void)xQueueSendFromISR(
                        uart_queue, &packet, &higher_priority_task_woken);
                }
                state = RxState::WaitHeader;
                rx_count = 0U;
            }
            break;
        }

        (void)HAL_UART_Receive_IT(huart, uart1_rx_byte, 1U);
        portYIELD_FROM_ISR(higher_priority_task_woken);
    }
}

extern "C" void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    __HAL_UART_CLEAR_FLAG(huart, UART_CLEAR_OREF | UART_CLEAR_NEF | UART_CLEAR_FEF);

    if (huart->Instance == USART1) {
        (void)HAL_UART_Receive_IT(huart, uart1_rx_byte, 1U);
    }
}

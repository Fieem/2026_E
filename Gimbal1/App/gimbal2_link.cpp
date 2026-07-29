#include "gimbal2_link.h"

#include <cstdio>
#include <cstring>

#include "main.h"
#include "usart.h"

namespace {

uint8_t rx_ring[GIMBAL2_LINK_RX_RING_SIZE];
volatile uint16_t rx_head = 0U;
uint16_t rx_tail = 0U;

char frame_buf[GIMBAL2_LINK_FRAME_MAX_LEN];
uint8_t frame_len = 0U;

Gimbal2LinkStatus link_status{};

bool ringPop(uint8_t *out) {
    if (out == nullptr || rx_tail == rx_head) {
        return false;
    }
    *out = rx_ring[rx_tail];
    rx_tail = static_cast<uint16_t>((rx_tail + 1U) % GIMBAL2_LINK_RX_RING_SIZE);
    return true;
}

void updateLastLine(const char *line) {
    std::strncpy(link_status.last_line, line, sizeof(link_status.last_line) - 1U);
    link_status.last_line[sizeof(link_status.last_line) - 1U] = '\0';
    link_status.last_rx_tick_ms = HAL_GetTick();
}

bool equalsUpper(const char *text, const char *expected_upper) {
    if (text == nullptr || expected_upper == nullptr) return false;

    while (*text != '\0' && *expected_upper != '\0') {
        char lhs = *text;
        if (lhs >= 'a' && lhs <= 'z') {
            lhs = static_cast<char>(lhs - ('a' - 'A'));
        }
        if (lhs != *expected_upper) return false;
        ++text;
        ++expected_upper;
    }
    return *text == '\0' && *expected_upper == '\0';
}

void parseLine(const char *line) {
    if (line == nullptr || line[0] == '\0') return;

    updateLastLine(line);

    char work[GIMBAL2_LINK_FRAME_MAX_LEN];
    std::strncpy(work, line, sizeof(work) - 1U);
    work[sizeof(work) - 1U] = '\0';

    char *context = nullptr;
    char *cmd = strtok_r(work, ",", &context);
    if (cmd == nullptr) return;

    if (equalsUpper(cmd, "READY")) {
        link_status.state = GIMBAL2_LINK_STATE_READY;
        link_status.ready_flag = true;
        return;
    }

    if (equalsUpper(cmd, "NEW")) {
        link_status.new_flag = true;
        return;
    }

    if (equalsUpper(cmd, "FINISH")) {
        link_status.state = GIMBAL2_LINK_STATE_FINISH;
        link_status.finish_flag = true;
        return;
    }

    if (equalsUpper(cmd, "BUSY")) {
        link_status.state = GIMBAL2_LINK_STATE_BUSY;
        link_status.busy_flag = true;
        return;
    }

    if (equalsUpper(cmd, "ERROR")) {
        link_status.state = GIMBAL2_LINK_STATE_ERROR;
        link_status.error_flag = true;
    }
}

void feedByte(uint8_t byte) {
    if (byte == '\r') return;

    if (byte == '\n') {
        frame_buf[frame_len] = '\0';
        if (frame_len > 0U) {
            parseLine(frame_buf);
        }
        frame_len = 0U;
        return;
    }

    if (frame_len < (GIMBAL2_LINK_FRAME_MAX_LEN - 1U)) {
        frame_buf[frame_len++] = static_cast<char>(byte);
        return;
    }

    frame_len = 0U;
}

} // namespace

extern "C" uint8_t gimbal2_link_rx_byte;
uint8_t gimbal2_link_rx_byte = 0U;

extern "C" void Gimbal2Link_Init(void) {
    rx_head = 0U;
    rx_tail = 0U;
    frame_len = 0U;
    std::memset(frame_buf, 0, sizeof(frame_buf));
    std::memset(&link_status, 0, sizeof(link_status));

    (void)HAL_UART_Receive_IT(&huart10, &gimbal2_link_rx_byte, 1U);
}

extern "C" void Gimbal2Link_Poll(void) {
    uint8_t value = 0U;
    while (ringPop(&value)) {
        feedByte(value);
    }
}

extern "C" void Gimbal2Link_RingPush(uint8_t byte) {
    const uint16_t next = static_cast<uint16_t>((rx_head + 1U) % GIMBAL2_LINK_RX_RING_SIZE);
    if (next == rx_tail) {
        return;
    }
    rx_ring[rx_head] = byte;
    rx_head = next;
}

extern "C" bool Gimbal2Link_GetStatus(Gimbal2LinkStatus *status) {
    if (status == nullptr) return false;
    *status = link_status;
    return true;
}

extern "C" bool Gimbal2Link_ClearFlags(void) {
    link_status.new_flag = false;
    link_status.ready_flag = false;
    link_status.finish_flag = false;
    link_status.busy_flag = false;
    link_status.error_flag = false;
    return true;
}

extern "C" bool Gimbal2Link_SendTask(
    const int32_t p1, const int32_t p2, const int32_t p3, const int32_t p4) {
    char buffer[64];
    const int length = std::snprintf(
        buffer, sizeof(buffer), "TASK,%ld,%ld,%ld,%ld\n",
        static_cast<long>(p1), static_cast<long>(p2),
        static_cast<long>(p3), static_cast<long>(p4));
    if (length <= 0 || length >= static_cast<int>(sizeof(buffer))) return false;

    return HAL_UART_Transmit(
               &huart10,
               reinterpret_cast<uint8_t *>(buffer),
               static_cast<uint16_t>(length),
               100U) == HAL_OK;
}

extern "C" bool Gimbal2Link_SendLine(const char *line) {
    if (line == nullptr) return false;

    const size_t length = std::strlen(line);
    if (length == 0U || length > UINT16_MAX) return false;

    return HAL_UART_Transmit(
               &huart10,
               reinterpret_cast<uint8_t *>(const_cast<char *>(line)),
               static_cast<uint16_t>(length),
               100U) == HAL_OK;
}

//
// Created by Administrator on 2026/7/9.
//

#include "cmsis_os2.h"
#include "Test/test.h"

void StartScreenTask(void *argument) {
    printsf_clear(0);
    test_vofa_init();           //启动 USART3 单字节中断接收
    for (;;) {
        test_vofa_poll();
        osDelay(10);
    }

}

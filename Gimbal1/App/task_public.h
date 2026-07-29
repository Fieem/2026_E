#ifndef TASK_PUBLIC_H
#define TASK_PUBLIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cmsis_os2.h"
#include "scara_protocol.h"

void StartReceiveTask(void *argument);
void StartGimbalTask(void *argument);

extern osThreadId_t GimbalTaskHandle;

#ifdef __cplusplus
}
#endif

#endif //TASK_PUBLIC_H

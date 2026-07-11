/* Force-include shim for building pywhispercpp/whisper.cpp on Windows-on-ARM64
 * with the rtools45 MinGW-w64 clang toolchain.
 *
 * rtools45's MinGW-w64 headers define the PROCESS-level PROCESS_POWER_THROTTLING_STATE
 * and the ThreadPowerThrottling enum, but omit the THREAD-level struct + constants
 * that recent whisper.cpp/ggml (ggml-cpu.c) uses. Supply them here and force-include
 * this file (clang -include ...) so the vendored source is left untouched.
 *
 * Not needed when building with a real MSVC-targeting clang-cl + the Windows SDK
 * (the SDK headers already define these). See BUILD_ARM.md.
 */
#include <windows.h>
#ifndef THREAD_POWER_THROTTLING_CURRENT_VERSION
typedef struct _THREAD_POWER_THROTTLING_STATE {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
} THREAD_POWER_THROTTLING_STATE, *PTHREAD_POWER_THROTTLING_STATE;
#define THREAD_POWER_THROTTLING_CURRENT_VERSION 1
#define THREAD_POWER_THROTTLING_EXECUTION_SPEED 0x1
#endif

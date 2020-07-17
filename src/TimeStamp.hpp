#pragma once

#include <functional>

/**
 * @brief timestamp_ms allows to measure time of something
 */
class timestamp_ms
{
public:
    using tOnStopCallback = std::function<void(uint64_t takenTime)>;

public:
    explicit timestamp_ms(tOnStopCallback&& onStop) noexcept;
    ~timestamp_ms();

private:
    static uint64_t getTimeStampMs() noexcept;

private:
    tOnStopCallback m_OnStop;
    uint64_t        m_Start;
};

#define TAKEN_TIME() auto _ = timestamp_ms([](uint64_t takenTime){ std::cout << __FILE__ << ":" << __LINE__ << ", taken: " << takenTime << "us" << std::endl; })

#include "TimeStamp.hpp"

#include "time.h"

timestamp_ms::timestamp_ms(timestamp_ms::tOnStopCallback&& onStop) noexcept
    : m_OnStop{std::move(onStop)}
    , m_Start{getTimeStampMs()}
{
}

timestamp_ms::~timestamp_ms()
{
  m_OnStop(getTimeStampMs() - m_Start);
}

uint64_t timestamp_ms::getTimeStampMs() noexcept
{
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return (tp.tv_sec * 1000000LL) + (tp.tv_nsec / 1000/*000*/);
}
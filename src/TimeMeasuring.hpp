#pragma once

#include <chrono>
#include <functional>
#include <iostream>

template<typename Units>
class TimeMeasuring
{
public:
   using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
   using OnFinishedCallback = std::function<void(int64_t)>;

public:
   TimeMeasuring(OnFinishedCallback&& onFinished) noexcept
      : _onFinished{std::move(onFinished)}
      , _startTime{std::chrono::steady_clock::now()}
   {
   }

   ~TimeMeasuring()
   {
      using namespace ::std::chrono;
      _onFinished(duration_cast<Units>(steady_clock::now() - _startTime).count());
   }

private:
   OnFinishedCallback _onFinished;
   TimePoint _startTime;
};

#define TAKEN_TIME_FN()                                                                                           \
   const char* __$$$__fn = __FUNCTION__;                                                                       \
   auto __$$$__tm = TimeMeasuring<::std::chrono::microseconds>([__$$$__fn](uint64_t takenTime) {               \
      std::cout << "[" << __$$$__fn << "]" << " taken " << (static_cast<double>(takenTime) / 1000.0) << " ms" << std::endl; \
   })

#define TAKEN_TIME() auto _$$$_ = TimeMeasuring<::std::chrono::microseconds>([](uint64_t takenTime){ std::cout << __FILE__ << ":" << __LINE__ << ", taken: " << takenTime << "us" << std::endl; })

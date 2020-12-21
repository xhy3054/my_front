#pragma once

#include <iostream>

namespace basalt {

#define UNUSED(x) (void)(x)

inline void assertion_failed(char const* expr, char const* function,
                             char const* file, long line) {
  std::cerr << "***** Assertion (" << expr << ") failed in " << function
            << ":\n"
            << file << ':' << line << ":" << std::endl;
  std::abort();
}

inline void assertion_failed_msg(char const* expr, char const* msg,
                                 char const* function, char const* file,
                                 long line) {
  std::cerr << "***** Assertion (" << expr << ") failed in " << function
            << ":\n"
            << file << ':' << line << ": " << msg << std::endl;
  std::abort();
}
}  // namespace basalt

#define BASALT_LIKELY(x) __builtin_expect(x, 1)

#if defined(BASALT_DISABLE_ASSERTS)

#define BASALT_ASSERT(expr) ((void)0)

#define BASALT_ASSERT_MSG(expr, msg) ((void)0)

#define BASALT_ASSERT_STREAM(expr, msg) ((void)0)

#else

#define BASALT_ASSERT(expr)                                               \
  (BASALT_LIKELY(!!(expr))                                                \
       ? ((void)0)                                                        \
       : ::basalt::assertion_failed(#expr, __PRETTY_FUNCTION__, __FILE__, \
                                    __LINE__))

#define BASALT_ASSERT_MSG(expr, msg)                                     \
  (BASALT_LIKELY(!!(expr))                                               \
       ? ((void)0)                                                       \
       : ::basalt::assertion_failed_msg(#expr, msg, __PRETTY_FUNCTION__, \
                                        __FILE__, __LINE__))

#define BASALT_ASSERT_STREAM(expr, msg)                                    \
  (BASALT_LIKELY(!!(expr))                                                 \
       ? ((void)0)                                                         \
       : (std::cerr << msg << std::endl,                                   \
          ::basalt::assertion_failed(#expr, __PRETTY_FUNCTION__, __FILE__, \
                                     __LINE__)))

#endif

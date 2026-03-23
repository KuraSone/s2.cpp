#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)

#ifdef S2_LIBRARY
#define S2_Export __declspec(dllexport)
static bool SuppressNonEssentialVerbosity = true;
#else
#define S2_Export 
static bool SuppressNonEssentialVerbosity = false;
#endif
#else

// On Unix-like systems, check for GCC 4+ visibility support
#if __GNUC__ >= 4
#define S2_Export __attribute__((visibility("default")))
#else
#define S2_Export
#endif
#endif
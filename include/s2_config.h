#pragma once

#define S2_LIBRARY //Just here as a test for now.

#if defined S2_LIBRARY
static bool SuppressNonEssentialVerbosity = true;
#else
static bool SuppressNonEssentialVerbosity = false;
#endif
// TimelineProfiler.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "timeline_profiler.hpp"

int main()
{
	{
		SCOPED_PROFILE("main");
		SET_PROFILE_DESC("additional info");

		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		{
			BEG_PROFILE("sub");
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			END_PROFILE();
		}
	}
	SAVE_PROFILE("profile.json");
	return 0;
}

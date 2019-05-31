#pragma once
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <stack>
#include <fstream>
#include <iomanip>
#include "json.hpp"

#define ENABLE_PROFILE 1

class ChromeTracingProfiler {
public:
	using clock_type = std::chrono::high_resolution_clock;

	ChromeTracingProfiler() :_origin(clock_type::now()) {

	}
	static ChromeTracingProfiler &instance() {
		static ChromeTracingProfiler profiler;
		return profiler;
	}
	uint32_t thread_id_int() {
		auto identifier = std::this_thread::get_id();
		std::lock_guard<std::recursive_mutex> scoped_lock(_mutex);
		auto it = _to_id_int.find(identifier);
		if (it == _to_id_int.end()) {
			uint32_t new_id = _thread_id_counter++;
			_to_id_int[identifier] = new_id;
			return new_id;
		}
		return it->second;
	}

	void beg_profile(const char *name, const char *file, int32_t line) {
		std::lock_guard<std::recursive_mutex> scoped_lock(_mutex);
		uint32_t itd = thread_id_int();
		TimeRange tr;
		tr.name = name;
		tr.tid = itd;
		tr.ts = microseconds_from_origin();
		tr.file = file;
		tr.line = line;
		_time_ranges[itd].push(tr);
	}
	void end_profile() {
		std::lock_guard<std::recursive_mutex> scoped_lock(_mutex);
		uint32_t itd = thread_id_int();
		std::stack<TimeRange> &timerange_stack = _time_ranges[itd];
		TimeRange tr = timerange_stack.top();
		timerange_stack.pop();
		tr.dur = microseconds_from_origin() - tr.ts;
		_events.emplace_back(tr);
	}
	void set_profile_desc(const char *desc) {
		std::lock_guard<std::recursive_mutex> scoped_lock(_mutex);
		uint32_t itd = thread_id_int();
		std::stack<TimeRange> &timerange_stack = _time_ranges[itd];
		timerange_stack.top().desc = desc;
	}

	uint64_t microseconds_from_origin() const {
		return std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - _origin).count();
	}

	void save(const char *dst) {
		std::lock_guard<std::recursive_mutex> scoped_lock(_mutex);
		// _events
		using namespace nlohmann;
		json trace_events = json::array();
		for (auto tr : _events) {
			json j;
			j["pid"] = 1;
			j["tid"] = tr.tid;
			j["ts"] = tr.ts;
			j["dur"] = tr.dur;
			j["ph"] = "X";
			j["name"] = tr.name;
			j["args"] = {
				{"file", tr.file},
				{"line", tr.line},
				{"desc", tr.desc},
			};
			trace_events.emplace_back(j);
		}
		json trace = {
			{"traceEvents", trace_events}
		};
		std::ofstream o(dst);
		o << std::setw(4) << trace;
	}
	void clear() {
		std::lock_guard<std::recursive_mutex> scoped_lock(_mutex);
		_origin = clock_type::now();
		_thread_id_counter = 1;
		_to_id_int.clear();
		_time_ranges.clear();
		_events.clear();
	}
private:
	clock_type::time_point _origin;

	// microseconds
	struct TimeRange {
		uint32_t tid = 0;
		std::string name;
		uint64_t ts = 0;
		uint64_t dur = 0;

		// args
		std::string file;
		uint32_t line;
		std::string desc;
	};

	std::recursive_mutex _mutex;
	uint32_t _thread_id_counter = 1;
	std::map<std::thread::id, uint32_t> _to_id_int;
	std::map<uint32_t, std::stack<TimeRange>> _time_ranges;
	std::vector<TimeRange> _events;
};

class ScopedChromeTracingProfile {
public:
	ScopedChromeTracingProfile(const char *name, const char *file, int32_t line) {
		ChromeTracingProfiler::instance().beg_profile(name, file, line);
	}
	~ScopedChromeTracingProfile() {
		ChromeTracingProfiler::instance().end_profile();
	}
	ScopedChromeTracingProfile(const ScopedChromeTracingProfile &) = delete;
	ScopedChromeTracingProfile& operator=(const ScopedChromeTracingProfile &) = delete;
};

#if ENABLE_PROFILE

#define BEG_PROFILE(name)    ChromeTracingProfiler::instance().beg_profile(name, __FILE__, __LINE__);
#define END_PROFILE()        ChromeTracingProfiler::instance().end_profile();

#define SCOPED_PROFILE(name) ScopedChromeTracingProfile _scoped_chrome_tracing_(name, __FILE__, __LINE__);

#define SET_PROFILE_DESC(desc) ChromeTracingProfiler::instance().set_profile_desc(desc);

#define SAVE_PROFILE(file) ChromeTracingProfiler::instance().save(file);
#define CLEAR_PROFILE()    ChromeTracingProfiler::instance().clear();

#else 

#define BEG_PROFILE(name)    ;
#define END_PROFILE()        ;

#define SCOPED_PROFILE(name) ;

#define SET_PROFILE_DESC(desc) ;

#define SAVE_PROFILE(file) ;
#define CLEAR_PROFILE()    ;

#endif
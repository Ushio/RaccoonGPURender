#pragma once
#include <memory>
#include <iostream>

namespace rt {
	class GNUPlotBase {
	public:
		void ytics(double tics) {
			_ytics = tics;
		}
	protected:
		void appy_ytics(FILE *fp) {
			if (0.0 < _ytics) {
				fprintf(fp, "set ytics %f\n", _ytics);
			}
		}
		double _ytics = -1.0;
	};
	class GNUPlot : public GNUPlotBase {
	public:
		GNUPlot() {
			_fp = std::shared_ptr<FILE>(_popen("gnuplot", "w"), _pclose);
			fprintf(_fp.get(), "$data << EOD\n");
		}
		void add(double x, double y) {
			fprintf(_fp.get(), "%f %f\n", x, y);
		}
		void show(const char *label) {
			fprintf(_fp.get(), "EOD\n");
			fprintf(_fp.get(), "set grid\n");
			appy_ytics(_fp.get());
			fprintf(_fp.get(), "plot '$data' using 1:2 with lines title '%s'\n", label);
			fflush(_fp.get());
		}
	private:
		std::shared_ptr<FILE> _fp;
	};

	class GNUPlot2 : public GNUPlotBase {
	public:
		GNUPlot2() {
			_fp = std::shared_ptr<FILE>(_popen("gnuplot", "w"), _pclose);
			fprintf(_fp.get(), "$data << EOD\n");
		}
		void add(double x, double y, double z) {
			fprintf(_fp.get(), "%f %f %f\n", x, y, z);
		}
		void show(const char *label_a, const char *label_b) {
			fprintf(_fp.get(), "EOD\n");
			fprintf(_fp.get(), "set grid\n");
			appy_ytics(_fp.get());
			fprintf(_fp.get(), "plot '$data' using 1:2 with lines title '%s', '$data' using 1:3 with lines title '%s'\n", label_a, label_b);
			fflush(_fp.get());
		}
	private:
		std::shared_ptr<FILE> _fp;
	};

	class GNUPlot3 : public GNUPlotBase {
	public:
		GNUPlot3() {
			_fp = std::shared_ptr<FILE>(_popen("gnuplot", "w"), _pclose);
			fprintf(_fp.get(), "$data << EOD\n");
		}
		void add(double x, double y, double z, double w) {
			fprintf(_fp.get(), "%f %f %f %f\n", x, y, z, w);
		}
		void show(const char *label_a, const char *label_b, const char *label_c) {
			fprintf(_fp.get(), "EOD\n");
			fprintf(_fp.get(), "set grid\n");
			appy_ytics(_fp.get());
			fprintf(_fp.get(), "plot '$data' using 1:2 with lines title '%s', '$data' using 1:3 with lines title '%s', '$data' using 1:4 with lines title '%s'\n", label_a, label_b, label_c);
			fflush(_fp.get());
		}
	private:
		std::shared_ptr<FILE> _fp;
	};

	class GNUPlot4 : public GNUPlotBase {
	public:
		GNUPlot4() {
			_fp = std::shared_ptr<FILE>(_popen("gnuplot", "w"), _pclose);
			fprintf(_fp.get(), "$data << EOD\n");
		}
		void add(double x, double y, double z, double w, double a) {
			fprintf(_fp.get(), "%f %f %f %f %f\n", x, y, z, w, a);
		}
		void show(const char *label_a, const char *label_b, const char *label_c, const char *label_d) {
			fprintf(_fp.get(), "EOD\n");
			fprintf(_fp.get(), "set grid\n");
			appy_ytics(_fp.get());
			fprintf(_fp.get(), "plot '$data' using 1:2 with lines title '%s', '$data' using 1:3 with lines title '%s', '$data' using 1:4 with lines title '%s', '$data' using 1:5 with lines title '%s'\n", label_a, label_b, label_c, label_d);
			fflush(_fp.get());
		}
	private:
		std::shared_ptr<FILE> _fp;
	};
}
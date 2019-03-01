#pragma once
#include "ofMain.h"

#include <imgui.h>

namespace ofxRaccoonImGui {
	void initialize();
	void shutdown();

	void beginGui();
	void endGui();

	class ScopedImGui {
	public:
		ScopedImGui() { beginGui(); }
		~ScopedImGui() { endGui(); }

		ScopedImGui(const ScopedImGui&) = delete;
		void operator=(const ScopedImGui&) = delete;
	};

	// ofDisableArbTex(); on create image is highly recommended for performance.
	void image(const ofImage &ofimage);
	void image(const ofImage &ofimage, float width, float height);

	template <class T>
	void Tree(const char *name, bool isOpen, T f) {
		if (isOpen) {
			ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
		}
		if (ImGui::TreeNode(name)) {
			f();
			ImGui::TreePop();
		}
	}
}
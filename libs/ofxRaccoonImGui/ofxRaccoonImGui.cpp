#include "ofxRaccoonImGui.hpp"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl2.h"

namespace ofxRaccoonImGui {
	void initialize() {
		ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = nullptr;

		ofAppGLFWWindow *window = dynamic_cast<ofAppGLFWWindow *>(ofGetWindowPtr());
		GLFWwindow *glfwWindow = window->getGLFWWindow();
		ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);

		if (ofIsGLProgrammableRenderer()) {
			ImGui_ImplOpenGL3_Init();
		}
		else {
			ImGui_ImplOpenGL2_Init();
		}
	}
	void shutdown() {
		if (ofIsGLProgrammableRenderer()) {
			ImGui_ImplOpenGL3_Shutdown();
		}
		else {
			ImGui_ImplOpenGL2_Shutdown();
		}
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	static std::vector<std::shared_ptr<ofTexture>> textureStorage;

	void beginGui() {
		if (ofIsGLProgrammableRenderer()) {
			ImGui_ImplOpenGL3_NewFrame();
		}
		else {
			ImGui_ImplOpenGL2_NewFrame();
		}

		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}
	void endGui() {
		ImGui::Render();

		if (ofIsGLProgrammableRenderer()) {
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
		else {
			ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
		}

		textureStorage.clear();
	}
	void image(const ofImage &ofimage) {
		image(ofimage, ofimage.getWidth(), ofimage.getHeight());
	}
	void image(const ofImage &ofimage, float width, float height) {
		if (ofimage.getTexture().getTextureData().textureTarget != GL_TEXTURE_2D) {
			bool arb = ofGetUsingArbTex();
			ofDisableArbTex();

			std::shared_ptr<ofTexture> texture = std::shared_ptr<ofTexture>(new ofTexture());
			textureStorage.push_back(texture);

			texture->loadData(ofimage.getPixels());
			ImGui::Image((ImTextureID)texture->getTextureData().textureID, ImVec2(width, height));

			if (arb) {
				ofEnableArbTex();
			}
		}
		else {
			ImGui::Image((ImTextureID)ofimage.getTexture().getTextureData().textureID, ImVec2(width, height));
		}
	}
}
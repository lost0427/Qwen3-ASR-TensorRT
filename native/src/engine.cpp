#include "qasr.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace qasr {

void Logger::log(Severity severity, const char* message) noexcept {
  if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", message);
}

void cuda_check(int code, const char* what) {
  if (code != 0) throw std::runtime_error(std::string(what) + " failed: cuda error " + std::to_string(code));
}

Engine::Engine(const std::string& path) {
  std::vector<char> blob = read_file(path);
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + path);
  context_ = engine_->createExecutionContext();
  if (!context_) throw std::runtime_error("createExecutionContext failed");
}

Engine::~Engine() {
  delete context_;
  delete engine_;
  delete runtime_;
}

const char* Engine::name(int index) const { return engine_->getIOTensorName(index); }

bool Engine::is_input(int index) const {
  return engine_->getTensorIOMode(name(index)) == nvinfer1::TensorIOMode::kINPUT;
}

nvinfer1::Dims Engine::dims(int index) const { return engine_->getTensorShape(name(index)); }

nvinfer1::DataType Engine::dtype(int index) const { return engine_->getTensorDataType(name(index)); }

void Engine::set_shape(const char* tensor, nvinfer1::Dims value) {
  if (!context_->setInputShape(tensor, value)) {
    throw std::runtime_error(std::string("setInputShape failed for ") + tensor);
  }
}

nvinfer1::Dims Engine::shape_after(const char* tensor) const { return context_->getTensorShape(tensor); }

void Engine::bind(const char* tensor, void* pointer) {
  if (!context_->setTensorAddress(tensor, pointer)) {
    throw std::runtime_error(std::string("setTensorAddress failed for ") + tensor);
  }
}

void Engine::run() {
  if (!context_->enqueueV3(nullptr)) throw std::runtime_error("enqueueV3 failed");
}

}  // namespace qasr

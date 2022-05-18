#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <boost/algorithm/clamp.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include "logging.h"
#include <string>
// Macro definition checks CUDA errors
#define CHECK(status) \
	do\
	{\
		auto ret = (status);\
		if (ret != 0)\
		{\
			std::cerr << "Cuda failure: " << ret << std::endl;\
			abort();\
		}\
	} while(0)

static const int INPUT_H = 720;
static const int INPUT_W = 1280;
static const int OUTPUT_H = 720;
static const int OUTPUT_W = 1280;

// Yours engine's input name; output name.
const char* INPUT_BLOB_NAME = "input.1";
const char* OUTPUT_BLOB_NAME1 = "950";
const char* OUTPUT_BLOB_NAME2 = "1037";
const char* OUTPUT_BLOB_NAME3 = "1122";

using namespace nvinfer1;
using namespace nvonnxparser;
static Logger gLogger;

//ICudaEngine* 
void createEngine(IHostMemory** modelStream, std::string onnxFilename) {
	//Create tensorrt builder
	/*
	 * createInferBuilder: create an instance of an IBuilder class
	 * IBuilder: Builder an engine from a network definition.
	 * */
	IBuilder* builder = createInferBuilder(gLogger);

  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	/*
	 * createBuilderConfig: create a builder configuration object.
	 * IBuilderConfig: Holds properties for configuring a builder to produce an engine.
	 * */
	IBuilderConfig* config = builder->createBuilderConfig();

	// Create a network definition object.
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
 
	auto parser = nvonnxparser::createParser(*network, gLogger);

  parser->parseFromFile(onnxFilename.c_str(), 2);
  
  for (int i = 0; i < parser->getNbErrors(); i++) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  std::cout << "TensorRT load onnx model sucessful!\n";

  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  
	(*modelStream) = engine->serialize();
	network->destroy();
  parser->destroy();
	engine->destroy();
	builder->destroy();
	config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output1, float* output2,float* output3,int batchSize) {
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
    	// Engine requires exactly IEngine::getNbBindings() number of buffers.
     std::cout << engine.getNbBindings() <<std::endl;
    	assert(engine.getNbBindings() == 4);
     for (int i=0; i<engine.getNbBindings(); i++ ) {
       if (engine.bindingIsInput(i) == true) {
         printf("Bind %d (%s): input.\n", i, engine.getBindingName(i));
       } else {
         printf("Bind %d  (%s):  Output.\n", i, engine.getBindingName(i));
       }
     }
	    void* buffers[4];

	    // In order to bind the buffers, we need to know the names of the input and output tensors.
	    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
	    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	    const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);
      const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
      const int outputIndex3 = engine.getBindingIndex(OUTPUT_BLOB_NAME3);
      //std::cout << outputIndex3 <<std::endl;

	    // Create GPU buffers on device
	    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
	    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 3 * OUTPUT_H * OUTPUT_W * sizeof(float)));
      CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * 3 * OUTPUT_H * OUTPUT_W * sizeof(float)));
      CHECK(cudaMalloc(&buffers[outputIndex3], batchSize * 3 * OUTPUT_H * OUTPUT_W * sizeof(float)));

	    // Create stream
	    cudaStream_t stream;
	    CHECK(cudaStreamCreate(&stream));

	    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	    context.enqueue(batchSize, buffers, stream, nullptr);
	    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], batchSize * 3 * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
      CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2], batchSize * 3 * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
      CHECK(cudaMemcpyAsync(output3, buffers[outputIndex3], batchSize * 3 * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
	    cudaStreamSynchronize(stream);

	    // Release stream and buffers
	    cudaStreamDestroy(stream);
	    CHECK(cudaFree(buffers[inputIndex]));
	    CHECK(cudaFree(buffers[outputIndex1]));
      CHECK(cudaFree(buffers[outputIndex2]));
      CHECK(cudaFree(buffers[outputIndex3]));
}

int main(int argc, char** argv) {
	char *trtModelStream{nullptr};
	size_t size{0};
	std::ifstream file("../weights/MIMO.engine", std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
  }

	// inference
	// Image data and initialize data;
  static float data[1 * 3 * OUTPUT_H * OUTPUT_W];
	for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    data[i] = 1;

	// Create an instance of an IRuntime class.
	// IRuntime: Allows a serialized functionally unsafe engine to be deserialized.
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	
	// Deserialize an engine from a stream when plugin factory is not uesd.
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	
	// Create an execution context.
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	delete[] trtModelStream;
	
	// Run inference.
	static float prob1[1 * 3 * OUTPUT_H * OUTPUT_W];
  static float prob2[1 * 3 * OUTPUT_H * OUTPUT_W];
  static float prob3[1 * 3 * OUTPUT_H * OUTPUT_W];
	for (int i = 0; i < 1; i++) {
		auto start = std::chrono::system_clock::now();
		doInference(*context, data, prob1, prob2, prob3, 1);
		auto end = std::chrono::system_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	}
	
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	return 0;
}

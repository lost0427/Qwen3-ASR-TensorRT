// Build a TensorRT engine from a published ONNX graph, without Python or PyTorch.
//
// The exports this project publishes are strongly typed and only their sequence
// axes are dynamic, so nothing from the original checkpoint is needed: the static
// dimensions (mel bins, hidden size) come from the parsed network, and the only
// thing the caller has to supply is the longest input the engine must accept.
//
// The graph fixes the upper bound of every dynamic axis at export time, so a
// larger window or a longer prompt needs a freshly exported ONNX. This tool can
// only choose a profile inside the range the graph already allows.

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

enum class Kind
{
    kAuto,
    kEncoder,
    kPrefill,
    kAlignerDecoder
};

char const* kindName(Kind kind)
{
    switch (kind)
    {
    case Kind::kEncoder:
        return "encoder";
    case Kind::kPrefill:
        return "prefill";
    case Kind::kAlignerDecoder:
        return "aligner-decoder";
    default:
        return "auto";
    }
}

struct Args
{
    std::string onnx;
    std::string out;
    std::string timingCache;
    Kind kind = Kind::kAuto;
    int32_t maxFrames = 0;
    int32_t optFrames = 0;
    int32_t maxSeq = 0;
    int32_t optSeq = 0;
    int32_t maxTimestamps = 0;
    int32_t optTimestamps = 0;
    int32_t workspaceGiB = 0;
    bool verbose = false;
    bool noTf32 = false;
    bool parseOnly = false;
};

char const* kUsage =
    "build-engine - build a TensorRT engine from an exported ONNX graph\n"
    "\n"
    "usage:\n"
    "  build-engine --onnx FILE --out FILE [options]\n"
    "\n"
    "required:\n"
    "  --onnx FILE             ONNX graph; a sibling FILE.onnx.data is picked up\n"
    "                          automatically when the graph uses external weights\n"
    "  --out FILE              engine file to write\n"
    "\n"
    "shape limits (the graph's maximum is fixed at export time):\n"
    "  --max-frames N          encoder: longest mel input, 100 frames per second\n"
    "                          (default 3600 = 36 s)\n"
    "  --opt-frames N          encoder: optimum mel input (default min(800, max))\n"
    "  --max-seq N             prefill: longest prompt in tokens (default 512,\n"
    "                          about 39 s of audio); aligner decoder: longest\n"
    "                          sequence (default 1024)\n"
    "  --opt-seq N             optimum prompt/sequence length\n"
    "                          (default min(256, max) for prefill, min(512, max)\n"
    "                          for the aligner decoder)\n"
    "  --max-timestamps N      aligner decoder: timestamp slots, two per word\n"
    "                          (default 256)\n"
    "  --opt-timestamps N      optimum timestamp slots (default min(64, max))\n"
    "  --workspace-gib N       TensorRT workspace ceiling (default 2, prefill 1)\n"
    "\n"
    "other:\n"
    "  --kind KIND             encoder | prefill | aligner-decoder; detected from\n"
    "                          the input names when omitted\n"
    "  --timing-cache FILE     reuse tactic timings across rebuilds, then update\n"
    "  --no-tf32               disable TF32 kernels\n"
    "  --parse-only            parse and print the graph, build nothing\n"
    "  --verbose               TensorRT info log plus per-input profile shapes\n"
    "  --version               print the tool and TensorRT version\n"
    "  --help                  this text\n";

class Logger : public nvinfer1::ILogger
{
public:
    explicit Logger(bool verbose) : verbose_(verbose) {}

    void log(Severity severity, char const* message) noexcept override
    {
        if (!verbose_ && severity > Severity::kWARNING)
        {
            return;
        }
        std::fprintf(stderr, "[trt %s] %s\n", severityName(severity), message);
    }

private:
    static char const* severityName(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "internal";
        case Severity::kERROR:
            return "error";
        case Severity::kWARNING:
            return "warn";
        case Severity::kINFO:
            return "info";
        case Severity::kVERBOSE:
            return "debug";
        }
        return "log";
    }

    bool verbose_;
};

[[noreturn]] void fail(char const* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    std::fputs("build-engine: ", stderr);
    std::vfprintf(stderr, format, arguments);
    std::fputc('\n', stderr);
    va_end(arguments);
    std::exit(1);
}

bool parseKind(std::string const& value, Kind& kind)
{
    if (value == "encoder")
    {
        kind = Kind::kEncoder;
    }
    else if (value == "prefill")
    {
        kind = Kind::kPrefill;
    }
    else if (value == "aligner-decoder")
    {
        kind = Kind::kAlignerDecoder;
    }
    else
    {
        return false;
    }
    return true;
}

bool parseInt(char const* flag, std::string const& value, int32_t& out)
{
    try
    {
        std::size_t consumed = 0;
        int const parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed <= 0)
        {
            fail("%s expects a positive integer, got '%s'", flag, value.c_str());
        }
        out = parsed;
    }
    catch (std::exception const&)
    {
        fail("%s expects a positive integer, got '%s'", flag, value.c_str());
    }
    return true;
}

// Returns false when the argument was --help or --version and the process is done.
bool parseArgs(int argc, char** argv, Args& args)
{
    for (int index = 1; index < argc; ++index)
    {
        std::string flag = argv[index];
        std::string value;
        bool const inlineValue = flag.rfind("--", 0) == 0 && flag.find('=') != std::string::npos;
        if (inlineValue)
        {
            value = flag.substr(flag.find('=') + 1);
            flag = flag.substr(0, flag.find('='));
        }
        auto next = [&]() -> std::string {
            if (inlineValue)
            {
                return value;
            }
            if (index + 1 >= argc)
            {
                fail("%s needs a value", flag.c_str());
            }
            return argv[++index];
        };

        if (flag == "--help" || flag == "-h")
        {
            std::fputs(kUsage, stdout);
            return false;
        }
        if (flag == "--version")
        {
            std::printf("build-engine tensorrt %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                NV_TENSORRT_PATCH);
            return false;
        }
        if (flag == "--onnx")
        {
            args.onnx = next();
        }
        else if (flag == "--out")
        {
            args.out = next();
        }
        else if (flag == "--kind")
        {
            std::string const kind = next();
            if (!parseKind(kind, args.kind))
            {
                fail("unknown --kind '%s'", kind.c_str());
            }
        }
        else if (flag == "--max-frames")
        {
            parseInt(flag.c_str(), next(), args.maxFrames);
        }
        else if (flag == "--opt-frames")
        {
            parseInt(flag.c_str(), next(), args.optFrames);
        }
        else if (flag == "--max-seq")
        {
            parseInt(flag.c_str(), next(), args.maxSeq);
        }
        else if (flag == "--opt-seq")
        {
            parseInt(flag.c_str(), next(), args.optSeq);
        }
        else if (flag == "--max-timestamps")
        {
            parseInt(flag.c_str(), next(), args.maxTimestamps);
        }
        else if (flag == "--opt-timestamps")
        {
            parseInt(flag.c_str(), next(), args.optTimestamps);
        }
        else if (flag == "--workspace-gib")
        {
            parseInt(flag.c_str(), next(), args.workspaceGiB);
        }
        else if (flag == "--timing-cache")
        {
            args.timingCache = next();
        }
        else if (flag == "--verbose")
        {
            args.verbose = true;
        }
        else if (flag == "--no-tf32")
        {
            args.noTf32 = true;
        }
        else if (flag == "--parse-only")
        {
            args.parseOnly = true;
        }
        else
        {
            fail("unknown option '%s' (try --help)", flag.c_str());
        }
    }
    return true;
}

bool fileExists(std::string const& path)
{
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::vector<char> readFile(std::string const& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        fail("cannot read %s", path.c_str());
    }
    std::streamsize const size = file.tellg();
    file.seekg(0);
    std::vector<char> blob(static_cast<std::size_t>(size));
    if (size > 0 && !file.read(blob.data(), size))
    {
        fail("short read on %s", path.c_str());
    }
    return blob;
}

void writeFile(std::string const& path, void const* data, std::size_t size)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        fail("cannot write %s", path.c_str());
    }
    file.write(static_cast<char const*>(data), static_cast<std::streamsize>(size));
    file.close();
    if (!file)
    {
        fail("short write on %s", path.c_str());
    }
}

bool hasInput(nvinfer1::INetworkDefinition const& network, char const* name)
{
    for (int32_t index = 0; index < network.getNbInputs(); ++index)
    {
        if (std::strcmp(network.getInput(index)->getName(), name) == 0)
        {
            return true;
        }
    }
    return false;
}

Kind detectKind(nvinfer1::INetworkDefinition const& network)
{
    if (hasInput(network, "inputs_embeds") && hasInput(network, "ts_positions"))
    {
        return Kind::kAlignerDecoder;
    }
    if (hasInput(network, "inputs_embeds") && hasInput(network, "position_ids"))
    {
        return Kind::kPrefill;
    }
    if (hasInput(network, "input_features"))
    {
        return Kind::kEncoder;
    }
    return Kind::kAuto;
}

struct ProfileSpec
{
    std::string name;
    int32_t axis;
    int64_t low;
    int64_t optimum;
    int64_t high;
};

struct Profile
{
    std::vector<ProfileSpec> specs;
    int32_t workspaceGiB = 2;
};

Profile planProfile(Kind kind, Args const& args)
{
    Profile plan;
    if (kind == Kind::kEncoder)
    {
        int64_t const high = args.maxFrames > 0 ? args.maxFrames : 3600;
        int64_t const optimum = args.optFrames > 0
            ? args.optFrames
            : std::min<int64_t>(800, high);
        plan.specs.push_back({"input_features", 2, 100, optimum, high});
        plan.specs.push_back({"input_features_mask", 1, 100, optimum, high});
        plan.workspaceGiB = 2;
    }
    else if (kind == Kind::kPrefill)
    {
        int64_t const high = args.maxSeq > 0 ? args.maxSeq : 512;
        int64_t const optimum = args.optSeq > 0 ? args.optSeq : std::min<int64_t>(256, high);
        plan.specs.push_back({"inputs_embeds", 1, 1, optimum, high});
        plan.specs.push_back({"position_ids", 1, 1, optimum, high});
        plan.workspaceGiB = 1;
    }
    else if (kind == Kind::kAlignerDecoder)
    {
        int64_t const sequence = args.maxSeq > 0 ? args.maxSeq : 1024;
        int64_t const sequenceOptimum
            = args.optSeq > 0 ? args.optSeq : std::min<int64_t>(512, sequence);
        int64_t const timestamps = args.maxTimestamps > 0 ? args.maxTimestamps : 256;
        int64_t const timestampsOptimum = args.optTimestamps > 0
            ? args.optTimestamps
            : std::min<int64_t>(64, timestamps);
        plan.specs.push_back({"inputs_embeds", 1, 8, sequenceOptimum, sequence});
        plan.specs.push_back({"ts_positions", 1, 2, timestampsOptimum, timestamps});
        plan.workspaceGiB = 2;
    }
    else
    {
        fail("cannot tell which graph this is; pass --kind encoder|prefill|aligner-decoder");
    }
    if (args.workspaceGiB > 0)
    {
        plan.workspaceGiB = args.workspaceGiB;
    }
    for (ProfileSpec const& spec : plan.specs)
    {
        if (!(spec.low <= spec.optimum && spec.optimum <= spec.high))
        {
            fail("%s: need min <= opt <= max, got %lld, %lld, %lld", spec.name.c_str(),
                static_cast<long long>(spec.low), static_cast<long long>(spec.optimum),
                static_cast<long long>(spec.high));
        }
    }
    return plan;
}

void printDims(char const* name, nvinfer1::Dims const& dims)
{
    std::printf("  %-24s [", name);
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        std::printf("%s%lld", index == 0 ? "" : ",", static_cast<long long>(dims.d[index]));
    }
    std::puts("]");
}

std::string cudaArch()
{
#ifdef _WIN32
    HMODULE library = LoadLibraryA("nvcuda.dll");
    if (library == nullptr)
    {
        return {};
    }
    using CuInit = int (*)(unsigned int);
    using CuDeviceGet = int (*)(int*, int);
    using CuComputeCapability = int (*)(int*, int*, int);
    auto init = reinterpret_cast<CuInit>(GetProcAddress(library, "cuInit"));
    auto getDevice = reinterpret_cast<CuDeviceGet>(GetProcAddress(library, "cuDeviceGet"));
    auto capability
        = reinterpret_cast<CuComputeCapability>(GetProcAddress(library, "cuDeviceComputeCapability"));
    int device = 0;
    int major = 0;
    int minor = 0;
    bool const ok = init != nullptr && getDevice != nullptr && capability != nullptr
        && init(0) == 0 && getDevice(&device, 0) == 0 && capability(&major, &minor, device) == 0;
    FreeLibrary(library);
    if (!ok)
    {
        return {};
    }
    return std::to_string(major) + "." + std::to_string(minor);
#else
    void* library = dlopen("libcuda.so.1", RTLD_NOW);
    if (library == nullptr)
    {
        return {};
    }
    using CuInit = int (*)(unsigned int);
    using CuDeviceGet = int (*)(int*, int);
    using CuComputeCapability = int (*)(int*, int*, int);
    auto init = reinterpret_cast<CuInit>(dlsym(library, "cuInit"));
    auto getDevice = reinterpret_cast<CuDeviceGet>(dlsym(library, "cuDeviceGet"));
    auto capability
        = reinterpret_cast<CuComputeCapability>(dlsym(library, "cuDeviceComputeCapability"));
    int device = 0;
    int major = 0;
    int minor = 0;
    bool const ok = init != nullptr && getDevice != nullptr && capability != nullptr
        && init(0) == 0 && getDevice(&device, 0) == 0 && capability(&major, &minor, device) == 0;
    dlclose(library);
    if (!ok)
    {
        return {};
    }
    return std::to_string(major) + "." + std::to_string(minor);
#endif
}

std::string jsonEscape(std::string const& text)
{
    std::string escaped;
    for (char character : text)
    {
        if (character == '\\' || character == '"')
        {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    return escaped;
}

void writeSidecar(
    Args const& args, Kind kind, Profile const& plan, std::size_t engineBytes, std::size_t onnxBytes)
{
    std::string const arch = cudaArch();
    FILE* file = std::fopen((args.out + ".json").c_str(), "wb");
    if (file == nullptr)
    {
        std::fprintf(stderr, "build-engine: warning: could not write %s.json\n", args.out.c_str());
        return;
    }
    std::fprintf(file,
        "{\n"
        "  \"engine\": \"%s\",\n"
        "  \"onnx\": \"%s\",\n"
        "  \"onnxBytes\": %zu,\n"
        "  \"engineBytes\": %zu,\n"
        "  \"kind\": \"%s\",\n"
        "  \"tensorrt\": \"%d.%d.%d\",\n"
        "  \"cudaArch\": \"%s\",\n"
        "  \"workspaceGiB\": %d,\n"
        "  \"tf32\": %s,\n"
        "  \"profile\": {\n",
        jsonEscape(args.out).c_str(), jsonEscape(args.onnx).c_str(), onnxBytes, engineBytes,
        kindName(kind), NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, arch.c_str(),
        plan.workspaceGiB, args.noTf32 ? "false" : "true");
    for (std::size_t index = 0; index < plan.specs.size(); ++index)
    {
        ProfileSpec const& spec = plan.specs[index];
        std::fprintf(file, "    \"%s\": [%lld, %lld, %lld]%s\n", jsonEscape(spec.name).c_str(),
            static_cast<long long>(spec.low), static_cast<long long>(spec.optimum),
            static_cast<long long>(spec.high), index + 1 == plan.specs.size() ? "" : ",");
    }
    std::fputs("  }\n}\n", file);
    std::fclose(file);
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(argc, argv, args))
    {
        return 0;
    }
    if (args.onnx.empty() || (args.out.empty() && !args.parseOnly))
    {
        std::fputs(kUsage, stderr);
        return 2;
    }
    if (!fileExists(args.onnx))
    {
        fail("no such file: %s", args.onnx.c_str());
    }

    Logger logger(args.verbose);
    initLibNvInferPlugins(&logger, "");

    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (builder == nullptr)
    {
        fail("could not create the TensorRT builder (is a CUDA device visible?)");
    }
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    if (network == nullptr)
    {
        fail("could not create the network definition");
    }
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (parser == nullptr)
    {
        fail("could not create the ONNX parser");
    }

    std::printf("parsing %s\n", args.onnx.c_str());
    auto const verbosity = args.verbose ? int32_t(nvinfer1::ILogger::Severity::kINFO)
                                        : int32_t(nvinfer1::ILogger::Severity::kWARNING);
    if (!parser->parseFromFile(args.onnx.c_str(), verbosity))
    {
        for (int32_t index = 0; index < parser->getNbErrors(); ++index)
        {
            nvonnxparser::IParserError const* error = parser->getError(index);
            char const* file = error->file();
            char const* description = error->desc();
            std::fprintf(stderr, "  %s:%d: %s\n", file != nullptr ? file : "", error->line(),
                description != nullptr ? description : "");
        }
        if (!fileExists(args.onnx + ".data"))
        {
            fail("TensorRT could not parse %s\n  if the graph uses external weights, copy %s.data "
                 "next to it",
                args.onnx.c_str(), args.onnx.c_str());
        }
        fail("TensorRT could not parse %s", args.onnx.c_str());
    }

    Kind const kind = args.kind == Kind::kAuto ? detectKind(*network) : args.kind;
    if (kind == Kind::kAuto)
    {
        fail("cannot tell which graph this is; pass --kind encoder|prefill|aligner-decoder");
    }

    if (args.verbose || args.parseOnly)
    {
        std::printf("graph: %s (%d inputs, %d outputs)\n", kindName(kind), network->getNbInputs(),
            network->getNbOutputs());
        for (int32_t index = 0; index < network->getNbInputs(); ++index)
        {
            nvinfer1::ITensor const* tensor = network->getInput(index);
            printDims(tensor->getName(), tensor->getDimensions());
        }
    }
    if (args.parseOnly)
    {
        return 0;
    }

    Profile const plan = planProfile(kind, args);
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (config == nullptr)
    {
        fail("could not create the builder configuration");
    }
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(plan.workspaceGiB) << 30);
    if (args.noTf32)
    {
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }

    std::unique_ptr<nvinfer1::ITimingCache> loadedCache;
    if (!args.timingCache.empty() && fileExists(args.timingCache))
    {
        std::vector<char> const blob = readFile(args.timingCache);
        loadedCache.reset(config->createTimingCache(blob.data(), blob.size()));
        if (loadedCache != nullptr)
        {
            config->setTimingCache(*loadedCache, true);
            if (args.verbose)
            {
                std::printf("timing cache: %s\n", args.timingCache.c_str());
            }
        }
    }

    // The builder owns the profile; deleting it would be a double free.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (profile == nullptr)
    {
        fail("could not create an optimization profile");
    }
    for (int32_t index = 0; index < network->getNbInputs(); ++index)
    {
        nvinfer1::ITensor const* tensor = network->getInput(index);
        std::string const name = tensor->getName();
        nvinfer1::Dims const networkDims = tensor->getDimensions();

        ProfileSpec const* spec = nullptr;
        for (ProfileSpec const& candidate : plan.specs)
        {
            if (candidate.name == name)
            {
                spec = &candidate;
                break;
            }
        }

        nvinfer1::Dims low = networkDims;
        nvinfer1::Dims optimum = networkDims;
        nvinfer1::Dims high = networkDims;
        if (spec != nullptr)
        {
            if (spec->axis >= networkDims.nbDims)
            {
                fail("%s: rank %d has no axis %d", name.c_str(), networkDims.nbDims, spec->axis);
            }
            int64_t const fixed = networkDims.d[spec->axis];
            if (fixed >= 0 && fixed != spec->high)
            {
                fail("%s: this graph fixes axis %d at %lld, so its maximum length is %lld, not %lld\n"
                     "  re-export the ONNX with the shape you need",
                    name.c_str(), spec->axis, static_cast<long long>(fixed),
                    static_cast<long long>(fixed), static_cast<long long>(spec->high));
            }
            low.d[spec->axis] = spec->low;
            optimum.d[spec->axis] = spec->optimum;
            high.d[spec->axis] = spec->high;
        }

        char const* selectors[] = {"min", "opt", "max"};
        nvinfer1::Dims const* shapes[] = {&low, &optimum, &high};
        nvinfer1::OptProfileSelector const modes[] = {nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX};
        for (int selector = 0; selector < 3; ++selector)
        {
            for (int32_t axis = 0; axis < shapes[selector]->nbDims; ++axis)
            {
                if (shapes[selector]->d[axis] < 0)
                {
                    fail("%s: axis %d of the %s shape is still dynamic; add a profile rule for "
                         "this graph",
                        name.c_str(), axis, selectors[selector]);
                }
            }
            if (!profile->setDimensions(name.c_str(), modes[selector], *shapes[selector]))
            {
                fail("%s: TensorRT rejected the %s shape", name.c_str(), selectors[selector]);
            }
        }
        if (args.verbose)
        {
            std::printf("profile %-22s %lld -> %lld -> %lld\n", name.c_str(),
                static_cast<long long>(low.d[spec != nullptr ? spec->axis : 0]),
                static_cast<long long>(optimum.d[spec != nullptr ? spec->axis : 0]),
                static_cast<long long>(high.d[spec != nullptr ? spec->axis : 0]));
        }
    }
    if (config->addOptimizationProfile(profile) < 0)
    {
        fail("TensorRT rejected the optimization profile");
    }

    std::printf("building %s engine (%d GiB workspace); this can take several minutes\n",
        kindName(kind), plan.workspaceGiB);
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (serialized == nullptr)
    {
        fail("TensorRT failed to build the engine (see the messages above)");
    }

    writeFile(args.out, serialized->data(), serialized->size());

    if (!args.timingCache.empty())
    {
        nvinfer1::ITimingCache const* cache = config->getTimingCache();
        std::unique_ptr<nvinfer1::IHostMemory> blob(cache != nullptr ? cache->serialize() : nullptr);
        if (blob != nullptr)
        {
            writeFile(args.timingCache, blob->data(), blob->size());
        }
    }

    std::ifstream onnxFile(args.onnx, std::ios::binary | std::ios::ate);
    std::size_t const onnxBytes
        = onnxFile ? static_cast<std::size_t>(onnxFile.tellg()) : std::size_t{0};
    writeSidecar(args, kind, plan, serialized->size(), onnxBytes);

    std::printf("wrote %s (%.1f MB), profile and build environment in %s.json\n", args.out.c_str(),
        double(serialized->size()) / (1024.0 * 1024.0), args.out.c_str());
    return 0;
}

#include <napi.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/FaissException.h>

using namespace Napi;
using idx_t = faiss::idx_t;

class IndexFlatL2 : public Napi::ObjectWrap<IndexFlatL2>
{
public:
  IndexFlatL2(const Napi::CallbackInfo &constructorArgs) : Napi::ObjectWrap<IndexFlatL2>(constructorArgs)
  {
    Napi::Env env = constructorArgs.Env();
    auto firstArg = constructorArgs[0];
    auto argsLength = constructorArgs.Length();
    if (firstArg.IsExternal())
    {
      const std::string importFilename = *firstArg.As<Napi::External<std::string>>().Data();
      try
      {
        auto file = faiss::read_index(importFilename.c_str());
        faissIndexPointer = std::unique_ptr<faiss::IndexFlatL2>(dynamic_cast<faiss::IndexFlatL2 *>(file));
      }
      catch (const faiss::FaissException& ex)
      {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
      }
    }
    else
    {
      if (!constructorArgs.IsConstructCall())
      {
        Napi::Error::New(env, "Class constructors cannot be invoked without 'new'")
            .ThrowAsJavaScriptException();
        return;
      }

      if (argsLength != 1)
      {
        Napi::Error::New(env, "Expected 1 argument, but got " + std::to_string(argsLength) + ".")
            .ThrowAsJavaScriptException();
        return;
      }
      if (!firstArg.IsNumber())
      {
        Napi::TypeError::New(env, "Invalid the first argument type, must be a number.").ThrowAsJavaScriptException();
        return;
      }

      auto dimensions = firstArg.As<Napi::Number>().Uint32Value();
      auto faissIndex = new faiss::IndexFlatL2(dimensions);
      faissIndexPointer = std::unique_ptr<faiss::IndexFlatL2>(faissIndex);
    }
  }

  static Napi::Object Init(Napi::Env env, Napi::Object exports)
  {
    // clang-format off
    Napi::Function func = DefineClass(env, "IndexFlatL2", {
      InstanceMethod("ntotal", &IndexFlatL2::ntotal),
      InstanceMethod("getDimension", &IndexFlatL2::getDimension),
      InstanceMethod("isTrained", &IndexFlatL2::isTrained),
      InstanceMethod("add", &IndexFlatL2::add),
      InstanceMethod("search", &IndexFlatL2::search),
      InstanceMethod("write", &IndexFlatL2::write),
      StaticMethod("read", &IndexFlatL2::read),
    });
    // clang-format on

    Napi::FunctionReference *constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("IndexFlatL2", func);
    return exports;
  }

  static Napi::Value read(const Napi::CallbackInfo &readArgs)
  {
    Napi::Env env = readArgs.Env();

    auto readArgsLength = readArgs.Length();
    if (readArgsLength != 1)
    {
      Napi::Error::New(env, "Expected 1 argument, but got " + std::to_string(readArgsLength) + ".")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    auto importFilename = readArgs[0];
    if (!importFilename.IsString())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be a string.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    std::string *filenamePointer = new std::string(importFilename.As<Napi::String>());
    Napi::FunctionReference *constructor = env.GetInstanceData<Napi::FunctionReference>();
    return constructor->New({ Napi::External<std::string>::New(env, filenamePointer) });
  }

private:
  std::unique_ptr<faiss::IndexFlatL2> faissIndexPointer;
  Napi::Value isTrained(const Napi::CallbackInfo &args)
  {
    return Napi::Boolean::New(args.Env(), faissIndexPointer->is_trained);
  }

  Napi::Value add(const Napi::CallbackInfo & args)
  {
    Napi::Env env = args.Env();

    auto argsLength = args.Length();
    if (argsLength != 1)
    {
      Napi::Error::New(env, "Expected 1 argument, but got " + std::to_string(argsLength) + ".")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    auto embeddingsArg = args[0];
    if (!embeddingsArg.IsArray())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be an Array.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    Napi::Array embeddingsToAdd = embeddingsArg.As<Napi::Array>();
    size_t embeddingsLength = embeddingsToAdd.Length();
    auto divisionResult = std::div(embeddingsLength, faissIndexPointer->d);
    auto embeddingsLengthIsValid = divisionResult.rem == 0;
    if (embeddingsLengthIsValid)
    {
      Napi::Error::New(env, "Invalid the given array length.")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    float *xb = new float[embeddingsLength];
    for (size_t i = 0; i < embeddingsLength; i++)
    {
      Napi::Value val = embeddingsToAdd[i];
      if (!val.IsNumber())
      {
        Napi::Error::New(env, "Expected a Number as array item. (at: " + std::to_string(i) + ")")
            .ThrowAsJavaScriptException();
        return env.Undefined();
      }
      xb[i] = val.As<Napi::Number>().FloatValue();
    }

    faissIndexPointer->add(divisionResult.quot, xb);

    delete[] xb;
    return env.Undefined();
  }

  Napi::Value search(const Napi::CallbackInfo &info)
  {
    Napi::Env env = info.Env();

    if (info.Length() != 2)
    {
      Napi::Error::New(env, "Expected 2 arguments, but got " + std::to_string(info.Length()) + ".")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    if (!info[0].IsArray())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be an Array.").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    if (!info[1].IsNumber())
    {
      Napi::TypeError::New(env, "Invalid the second argument type, must be a Number.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    const uint32_t k = info[1].As<Napi::Number>().Uint32Value();
    if (k > faissIndexPointer->ntotal)
    {
      Napi::Error::New(env, "Invalid the number of k (cannot be given a value greater than `ntotal`: " +
                                std::to_string(faissIndexPointer->ntotal) + ").")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    Napi::Array arr = info[0].As<Napi::Array>();
    size_t length = arr.Length();
    auto dv = std::div(length, faissIndexPointer->d);
    if (dv.rem != 0)
    {
      Napi::Error::New(env, "Invalid the given array length.")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    float *xq = new float[length];
    for (size_t i = 0; i < length; i++)
    {
      Napi::Value val = arr[i];
      if (!val.IsNumber())
      {
        Napi::Error::New(env, "Expected a Number as array item. (at: " + std::to_string(i) + ")")
            .ThrowAsJavaScriptException();
        return env.Undefined();
      }
      xq[i] = val.As<Napi::Number>().FloatValue();
    }

    auto nq = dv.quot;
    idx_t *I = new idx_t[k * nq];
    float *D = new float[k * nq];

    faissIndexPointer->search(nq, xq, k, D, I);

    Napi::Array arr_distances = Napi::Array::New(env, k * nq);
    Napi::Array arr_labels = Napi::Array::New(env, k * nq);
    for (size_t i = 0; i < k * nq; i++)
    {
      idx_t label = I[i];
      float distance = D[i];
      arr_distances[i] = Napi::Number::New(env, distance);
      arr_labels[i] = Napi::Number::New(env, label);
    }
    delete[] I;
    delete[] D;

    Napi::Object results = Napi::Object::New(env);
    results.Set("distances", arr_distances);
    results.Set("labels", arr_labels);
    return results;
  }

  Napi::Value ntotal(const Napi::CallbackInfo &info)
  {
    return Napi::Number::New(info.Env(), faissIndexPointer->ntotal);
  }

  Napi::Value getDimension(const Napi::CallbackInfo &info)
  {
    return Napi::Number::New(info.Env(), faissIndexPointer->d);
  }

  Napi::Value write(const Napi::CallbackInfo &info)
  {
    Napi::Env env = info.Env();

    if (info.Length() != 1)
    {
      Napi::Error::New(env, "Expected 1 argument, but got " + std::to_string(info.Length()) + ".")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    if (!info[0].IsString())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be a string.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    const std::string fname = info[0].As<Napi::String>().Utf8Value();

    faiss::write_index(faissIndexPointer.get(), fname.c_str());

    return env.Undefined();
  }
};

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
  IndexFlatL2::Init(env, exports);
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, Init)
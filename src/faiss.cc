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
    const Napi::Value *importFilename = &readArgs[0];
    if (!importFilename->IsString())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be a string.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    std::string *filenamePointer = new std::string(importFilename->As<Napi::String>());
    Napi::FunctionReference *constructor = env.GetInstanceData<Napi::FunctionReference>();
    return constructor->New({ Napi::External<std::string>::New(env, filenamePointer) });
  }

private:
  std::unique_ptr<faiss::IndexFlatL2> faissIndexPointer;
  Napi::Value isTrained(const Napi::CallbackInfo &args)
  {
    return Napi::Boolean::New(args.Env(), faissIndexPointer->is_trained);
  }

  Napi::Value add(const Napi::CallbackInfo &args)
  {
    Napi::Env env = args.Env();

    auto argsLength = args.Length();
    if (argsLength != 1)
    {
      Napi::Error::New(env, "Expected 1 argument, but got " + std::to_string(argsLength) + ".")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    const Napi::Value* vectorToAddArg = &args[0];
    if (!vectorToAddArg->IsArray())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be an Array.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    Napi::Array vectorToAddArray = vectorToAddArg->As<Napi::Array>();
    size_t vectorToAddLength = vectorToAddArray.Length();
    auto divisionResult = std::div(vectorToAddLength, faissIndexPointer->d);
    auto vectorHasCorrectDimension = divisionResult.rem == 0;
    if (!vectorHasCorrectDimension)
    {
      Napi::Error::New(env, "Invalid the given array length.")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    float* vectorToAddPointer = new float[vectorToAddLength];
    for (size_t i = 0; i < vectorToAddLength; i++)
    {
      Napi::Value vectorValue = vectorToAddArray[i];
      if (!vectorValue.IsNumber())
      {
        Napi::Error::New(env, "Expected a Number as array item. (at: " + std::to_string(i) + ")")
            .ThrowAsJavaScriptException();
        return env.Undefined();
      }
      vectorToAddPointer[i] = vectorValue.As<Napi::Number>().FloatValue();
    }

    faissIndexPointer->add(divisionResult.quot, vectorToAddPointer);

    delete[] vectorToAddPointer;
    return env.Undefined();
  }


  Napi::Value search(const Napi::CallbackInfo& searchArgs)
  {
      Napi::Env env = searchArgs.Env();
      auto searchArgs_length = searchArgs.Length();
      if (searchArgs_length != 2)
      {
          Napi::Error::New(env, "Expected 2 arguments, but got " + std::to_string(searchArgs_length) + ".")
              .ThrowAsJavaScriptException();
          return env.Undefined();
      }

      const Napi::Value* searchArgs_vectorsToSearchFor = &searchArgs[0];
      if (!searchArgs_vectorsToSearchFor->IsArray())
      {
          Napi::TypeError::New(env, "Invalid the first argument type, must be an Array.").ThrowAsJavaScriptException();
          return env.Undefined();
      }

      const Napi::Value* searchArgs_searchResultLimit = &searchArgs[1];
      if (!searchArgs_searchResultLimit->IsNumber())
      {
          Napi::TypeError::New(env, "Invalid the second argument type, must be a Number.").ThrowAsJavaScriptException();
          return env.Undefined();
      }

      const uint32_t searchResultLimit = searchArgs_searchResultLimit->As<Napi::Number>().Uint32Value();
      if (searchResultLimit > faissIndexPointer->ntotal)
      {
          Napi::Error::New(env, "Invalid the number of k (cannot be given a value greater than `ntotal`: " +
              std::to_string(faissIndexPointer->ntotal) + ").")
              .ThrowAsJavaScriptException();
          return env.Undefined();
      }

      Napi::Array vectorToSearchFor_asArray = searchArgs_vectorsToSearchFor->As<Napi::Array>();
      size_t vectorToSearchFor_asArray_length = vectorToSearchFor_asArray.Length();
      auto divisionResult = std::div(vectorToSearchFor_asArray_length, faissIndexPointer->d);
      if (divisionResult.rem != 0)
      {
          Napi::Error::New(env, "Invalid the given array length.")
              .ThrowAsJavaScriptException();
          return env.Undefined();
      }

      float* vectorToSearchFor = new float[vectorToSearchFor_asArray_length];
      for (size_t i = 0; i < vectorToSearchFor_asArray_length; i++)
      {
          Napi::Value vectorValue = vectorToSearchFor_asArray[i];
          if (!vectorValue.IsNumber())
          {
              Napi::Error::New(env, "Expected a Number as array item. (at: " + std::to_string(i) + ")")
                  .ThrowAsJavaScriptException();
              return env.Undefined();
          }
          vectorToSearchFor[i] = vectorValue.As<Napi::Number>().FloatValue();
      }

      auto vectorDimension = divisionResult.quot;
      const auto totalNumberOfResultVectorValues = searchResultLimit * vectorDimension;
      idx_t* indexLabelResults = new idx_t[totalNumberOfResultVectorValues];
      float* indexDistanceResults = new float[totalNumberOfResultVectorValues];

      faissIndexPointer->search(vectorDimension, vectorToSearchFor, searchResultLimit, indexDistanceResults, indexLabelResults);

      Napi::Array distances = Napi::Array::New(env, totalNumberOfResultVectorValues);
      Napi::Array labels = Napi::Array::New(env, totalNumberOfResultVectorValues);
      for (size_t i = 0; i < totalNumberOfResultVectorValues; i++)
      {
          idx_t label = indexLabelResults[i];
          float distance = indexDistanceResults[i];
          distances[i] = Napi::Number::New(env, distance);
          labels[i] = Napi::Number::New(env, label);
      }
      delete[] indexLabelResults;
      delete[] indexDistanceResults;

      Napi::Object results = Napi::Object::New(env);
      results.Set("distances", distances);
      results.Set("labels", labels);
      return results;
  }

  Napi::Value ntotal(const Napi::CallbackInfo &args)
  {
    return Napi::Number::New(args.Env(), faissIndexPointer->ntotal);
  }

  Napi::Value getDimension(const Napi::CallbackInfo &args)
  {
    return Napi::Number::New(args.Env(), faissIndexPointer->d);
  }

  Napi::Value write(const Napi::CallbackInfo &args)
  {
    Napi::Env env = args.Env();

    if (args.Length() != 1)
    {
      Napi::Error::New(env, "Expected 1 argument, but got " + std::to_string(args.Length()) + ".")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }

    const Napi::Value* filennameValue = &args[0];
    if (!filennameValue->IsString())
    {
      Napi::TypeError::New(env, "Invalid the first argument type, must be a string.").ThrowAsJavaScriptException();
      return env.Undefined();
    }

    const std::string filename = filennameValue->As<Napi::String>().Utf8Value();

    faiss::write_index(faissIndexPointer.get(), filename.c_str());

    return env.Undefined();
  }
};

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
  IndexFlatL2::Init(env, exports);
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, Init)
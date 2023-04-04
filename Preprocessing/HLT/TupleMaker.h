#pragma once

#include <map>
#include <vector>
#include <thread>
#include <typeinfo>

#include "EntryQueue.h"

namespace analysis {

struct Entry {
  bool valid{false};
  std::map<int, int> int_values;
  std::map<int, float> float_values;
  std::map<int, std::vector<int>> vint_values;
  std::map<int, std::vector<float>> vfloat_values;

  void Add(int index, float value)
  {
    CheckIndex(index);
    float_values[index] = value;
  }

  void Add(int index, int value)
  {
    CheckIndex(index);
    int_values[index] = value;
  }

  void Add(int index, const RVecI& value)
  {
    AddToMap(index, value, vint_values);
  }

  void Add(int index, const RVecF& value)
  {
    AddToMap(index, value, vfloat_values);
  }

private:
  template<typename In, typename Out>
  void AddToMap(int index, const ROOT::VecOps::RVec<In>& input, std::map<int, std::vector<Out>>& output)
  {
    CheckIndex(index);
    auto& out = output[index];
    for(auto x : input)
      out.push_back(x);
  }

  void CheckIndex(int index) const
  {
    if (int_values.count(index) || float_values.count(index) || vint_values.count(index) || vfloat_values.count(index))
      throw std::runtime_error("Entry::Add: index already exists");
  }
};

struct StopLoop {};

namespace detail {
inline void putEntry(std::vector<Entry>& entries, int index, const RVecB& sel) {}

template<typename T, typename ...Args>
void putEntry(std::vector<Entry>& entries, int var_index, const RVecB& sel,
              const ROOT::VecOps::RVec<T>& value, Args&& ...args)
{
  if(entries.empty()) {
    size_t n = 0;
    for(size_t i = 0; i < sel.size(); ++i) {
      if(sel[i])
        ++n;
    }
    entries.resize(n);
  }
  // std::cout << "putEntry: var_index=" << var_index << " value.size()=" << value.size() << " entries.size()="
  //           << entries.size() <<  " type_name=" << typeid(T).name() << std::endl;
  for(size_t entry_index = 0, out_index = 0; entry_index < value.size(); ++entry_index) {
    if(sel.at(entry_index)) {
      entries.at(out_index).Add(var_index, value[entry_index]);
      ++out_index;
    }
  }
  putEntry(entries, var_index + 1, sel, std::forward<Args>(args)...);
}

} // namespace detail

template<typename ...Args>
struct TupleMaker {
  TupleMaker(size_t queue_size, size_t max_entries)
    : queue(queue_size, max_entries)
  {
  }

  TupleMaker(const TupleMaker&) = delete;
  TupleMaker& operator= (const TupleMaker&) = delete;

  ROOT::RDF::RNode process(ROOT::RDF::RNode df_in, ROOT::RDF::RNode df_out, const std::vector<std::string>& var_names,
                           int start_idx, int stop_idx, int batch_size)
  {
    thread = std::make_unique<std::thread>([=]() {
      std::cout << "TupleMaker::process: foreach started." << std::endl;
      try {
        ROOT::RDF::RNode df = df_in;
        df.Foreach([&](const Args& ...args) {
          std::vector<Entry> entries;
          // std::cout << "TupleMaker::process: running detail::putEntry." << std::endl;
          detail::putEntry(entries, 0, args...);
          // std::cout << "TupleMaker::process: running detail::putEntry done." << std::endl;
          for(auto& entry : entries) {
            // std::cout << "TupleMaker::process: push entry." << std::endl;
            entry.valid = true;
            if(!queue.Push(entry)) {
              std::cout << "TupleMaker::process: queue is full." << std::endl;
              throw StopLoop();
            }
            // std::cout << "TupleMaker::process: push entry done." << std::endl;
          }
          // std::cout << "TupleMaker::process: Foreach step done." << std::endl;
        }, var_names);
      } catch(StopLoop) {
        // std::cout << "TupleMaker::process: StopLoop exception." << std::endl;
      }
      queue.SetAllDone();
      std::cout << "TupleMaker::process: foreach done." << std::endl;
    });
    df_out = df_out.Define("_entry", [=](ULong64_t rdfentry) {
      Entry entry;
      const int index = rdfentry % batch_size;
      if(index >= start_idx && index < stop_idx) {
        // std::cout << "TupleMaker::process: pop entry " << rdfentry << " index=" << index << " start_idx="
        //           << start_idx << " stop_idx=" << stop_idx << " batch_size=" << batch_size << std::endl;
        if(!queue.Pop(entry)) {
          // std::cout << "TupleMaker::process: queue is empty" << std::endl;
          // throw std::runtime_error("TupleMaker::process: queue is empty");
        }
        // std::cout << "TupleMaker::process: pop entry done " << rdfentry << std::endl;
      } else {
        // std::cout << "TupleMaker::process: skip entry " << rdfentry << " index=" << index << " start_idx="
        //           << start_idx << " stop_idx=" << stop_idx << std::endl;
      }
      return entry;
    }, { "rdfentry_" });
    return df_out;
  }

  void join()
  {
    if(thread)
      thread->join();
  }

  EntryQueue<Entry> queue;
  std::unique_ptr<std::thread> thread;
};

} // namespace analysis
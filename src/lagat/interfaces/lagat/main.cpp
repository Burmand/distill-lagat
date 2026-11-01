#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <planner.hpp>
#include <policy.hpp>
#include <utils.hpp>

#include <torch/torch.h>

const auto tmpdir = std::filesystem::temp_directory_path();
const auto map_name = tmpdir / "tmp.map";
const auto scene_name = tmpdir / "tmp.scene";
const auto record_name = tmpdir / "recorded_x.pt";  // Recorded x for RL training in Python

extern "C" {
const char* run_lagat(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec, int seed,
                      const char* model_path_cstr, int deadlock_detection,
                      int deadlock_depth, int lns, int plns_num_refiners,
                      int verbose, int sampling, float tau); // Add sampling, tau
}

const char* run_lagat(const char* map_content_cstr,
                      const char* scene_content_cstr, int N,
                      float time_limit_sec, int seed,
                      const char* model_path_cstr, int deadlock_detection,
                      int deadlock_depth, int lns, int plns_num_refiners,
                      int verbose, int sampling, float tau) // Add sampling, tau
{
  std::string map_content(map_content_cstr);
  std::string scene_content(scene_content_cstr);
  std::string model_path = model_path_cstr ? model_path_cstr : "";

  // setup instance
  std::ofstream ins_file;
  ins_file.open(scene_name, std::ios::out);
  ins_file << scene_content;
  ins_file.close();

  // setup instance
  std::ofstream map_file;
  map_file.open(map_name, std::ios::out);
  map_file << map_content;
  map_file.close();

  const auto ins = Instance(scene_name, map_name, N);
  if (!ins.is_valid(1)) return "ERROR_SCENE";

  AgentPolicy::MODEL_FILEPATH = model_path;

  torch::manual_seed(seed);
  // Sampling on = Exploration, Sampling off = argmax
  AgentPolicy::SAMPLING_STRATEGY = sampling ? AgentPolicy::Probablistic : AgentPolicy::Deterministic;
  // Higher temperature = more random rankings
  AgentPolicy::SAMPLING_TEMPERTURE = tau;

  LaCAM::DEADLOCK_DETECTION = deadlock_detection != 0;
  LaCAM::DEADLOCK_DEPTH = deadlock_depth;
  LNS::ON = lns != 0;
  PLNS::NUM_REFINERS = plns_num_refiners;

  // Empty the recorded x from the previous run
  AgentPolicy::RECORDED_NODE_FEATURES.clear();
  AgentPolicy::RECORDED_EDGE_INDEX.clear();
  AgentPolicy::RECORDED_EDGE_ATTR.clear();
  AgentPolicy::RECORDED_L.clear();

  // solve
  auto deadline = Deadline(time_limit_sec * 1000);
  set_print_options(deadline, verbose);
  const auto solution = solve(ins, verbose - 1, &deadline, seed);

  // failure
  if (solution.empty()) return "ERROR_EMPTY";

  // check feasibility
  if (!is_feasible_solution(ins, solution)) return "ERROR_SOLUTION";

  // Write the recorded x (views, edges, edge info, top moves) to a file Python can read with torch.load
  auto recorded = c10::ivalue::Tuple::create(
      {AgentPolicy::RECORDED_NODE_FEATURES, AgentPolicy::RECORDED_EDGE_INDEX,
       AgentPolicy::RECORDED_EDGE_ATTR, AgentPolicy::RECORDED_L});
  std::vector<char> bytes = torch::pickle_save(recorded);  // Turn the 4 lists into bytes
  std::ofstream record_file(record_name, std::ios::binary);
  record_file.write(bytes.data(), bytes.size());
  record_file.close();

  auto get_x = [&](int k) { return k % ins.G->width; };
  auto get_y = [&](int k) { return k / ins.G->width; };

  std::ostringstream result_string;
  for (size_t t = 0; t < solution.size(); ++t) {
    auto C = solution[t];
    for (auto v : C) {
      result_string << get_x(v->index) << "," << get_y(v->index) << "|";
    }
    result_string << "\n";
  }

  static std::string result;
  result = result_string.str();

  return result.c_str();
}

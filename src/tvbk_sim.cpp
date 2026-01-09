/* Standalone CLI for running tvbk simulations */

#include "tvbk.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tvbk;

// CSV parsing utilities
std::vector<std::vector<float>> read_csv_matrix(const std::string &filename) {
  std::vector<std::vector<float>> matrix;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    exit(1);
  }

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      row.push_back(std::stof(value));
    }
    if (!row.empty()) {
      matrix.push_back(row);
    }
  }
  return matrix;
}

struct ParamData {
  std::vector<std::string> names;
  std::vector<std::vector<float>> values; // [node][param]
};

ParamData read_csv_params(const std::string &filename) {
  ParamData data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    exit(1);
  }

  std::string line;
  // Read header
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string name;
    while (std::getline(ss, name, ',')) {
      data.names.push_back(name);
    }
  }

  // Read values
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      row.push_back(std::stof(value));
    }
    if (!row.empty()) {
      data.values.push_back(row);
    }
  }
  return data;
}

void write_csv_timeseries(const std::string &filename, const float *data,
                          uint32_t num_steps, uint32_t num_nodes,
                          uint32_t num_svars, float dt) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open output file " << filename << std::endl;
    exit(1);
  }

  // Write header
  file << "time";
  for (uint32_t node = 0; node < num_nodes; node++) {
    for (uint32_t svar = 0; svar < num_svars; svar++) {
      file << ",node_" << node << "_var_" << svar;
    }
  }
  file << "\n";

  // Write data
  for (uint32_t t = 0; t < num_steps; t++) {
    file << (t * dt);
    for (uint32_t svar = 0; svar < num_svars; svar++) {
      for (uint32_t node = 0; node < num_nodes; node++) {
        // data layout: (num_svar, num_node, width=1)
        uint32_t idx = svar * num_nodes + node;
        file << "," << data[idx];
      }
    }
    file << "\n";
  }
}

// Helper to compute CSR connectivity from dense matrix
struct ConnData {
  std::vector<float> weights;
  std::vector<uint32_t> indices;
  std::vector<uint32_t> indptr;
  std::vector<uint32_t> idelays;
  uint32_t num_nonzero;
};

ConnData matrix_to_csr(const std::vector<std::vector<float>> &weights,
                       const std::vector<std::vector<float>> &lengths,
                       float speed, float dt) {
  ConnData data;
  uint32_t num_nodes = weights.size();
  data.indptr.push_back(0);

  for (uint32_t i = 0; i < num_nodes; i++) {
    for (uint32_t j = 0; j < num_nodes; j++) {
      if (weights[i][j] != 0.0f) {
        data.weights.push_back(weights[i][j]);
        data.indices.push_back(j);
        // Convert length to delay in time steps
        float delay_ms = lengths[i][j] / speed;
        uint32_t delay_steps = (uint32_t)std::ceil(delay_ms / dt);
        data.idelays.push_back(delay_steps);
      }
    }
    data.indptr.push_back(data.weights.size());
  }
  data.num_nonzero = data.weights.size();
  return data;
}

// Template function to run simulation for a given model
template <typename Model>
void run_simulation(const ConnData &conn_data, const ParamData &param_data,
                    uint32_t num_nodes, uint32_t num_steps, float dt,
                    uint32_t tavg_period, float noise_level, uint64_t seed,
                    const std::string &output_file) {
  constexpr int width = 1; // scalar mode
  constexpr uint32_t num_svar = Model::num_svar;
  constexpr uint32_t num_parm = Model::num_parm;

  std::cout << "Running " << Model::name << " simulation..." << std::endl;
  std::cout << "  Nodes: " << num_nodes << std::endl;
  std::cout << "  State variables: " << num_svar << std::endl;
  std::cout << "  Parameters: " << num_parm << std::endl;
  std::cout << "  Time steps: " << num_steps << std::endl;
  std::cout << "  dt: " << dt << std::endl;
  std::cout << "  Temporal averaging period: " << tavg_period << std::endl;

  // Determine horizon (next power of 2 greater than max delay)
  uint32_t max_delay = 0;
  for (auto d : conn_data.idelays) {
    max_delay = std::max(max_delay, d);
  }
  uint32_t horizon = 1;
  while (horizon <= max_delay) {
    horizon *= 2;
  }
  std::cout << "  Max delay: " << max_delay << " steps" << std::endl;
  std::cout << "  Horizon: " << horizon << std::endl;

  // Create connectivity structure
  conn c(num_nodes, conn_data.num_nonzero);
  std::copy(conn_data.weights.begin(), conn_data.weights.end(),
            const_cast<float *>(c.weights));
  std::copy(conn_data.indices.begin(), conn_data.indices.end(),
            const_cast<uint32_t *>(c.indices));
  std::copy(conn_data.indptr.begin(), conn_data.indptr.end(),
            const_cast<uint32_t *>(c.indptr));
  std::copy(conn_data.idelays.begin(), conn_data.idelays.end(),
            const_cast<uint32_t *>(c.idelays));

  // Create coupling buffer
  cxb<width> cx(num_nodes, horizon);

  // Initialize states (zeros)
  std::vector<float> states(num_svar * num_nodes * width, 0.0f);

  // Initialize temporal average buffer
  std::vector<float> tavg(num_svar * num_nodes * width, 0.0f);

  // Initialize noise scaling
  std::vector<float> noise_scale(num_svar * width, noise_level);

  // Initialize parameters
  std::vector<float> params(num_parm * num_nodes * width);
  bool p_varies_node = param_data.values.size() > 1;

  if (p_varies_node) {
    // Parameters vary per node
    if (param_data.values.size() != num_nodes) {
      std::cerr << "Error: Parameter file has " << param_data.values.size()
                << " rows but network has " << num_nodes << " nodes"
                << std::endl;
      exit(1);
    }
    for (uint32_t node = 0; node < num_nodes; node++) {
      for (uint32_t p = 0; p < num_parm; p++) {
        params[node * num_parm * width + p * width] =
            param_data.values[node][p];
      }
    }
  } else {
    // Global parameters
    for (uint32_t node = 0; node < num_nodes; node++) {
      for (uint32_t p = 0; p < num_parm; p++) {
        params[node * num_parm * width + p * width] = param_data.values[0][p];
      }
    }
  }

  // Initialize RNG seed
  std::vector<uint64_t> rng_seed(width * 4);
  for (int i = 0; i < width; i++) {
    rng_seed[i * 4 + 0] = seed + i;
    rng_seed[i * 4 + 1] = seed + i + 1000;
    rng_seed[i * 4 + 2] = seed + i + 2000;
    rng_seed[i * 4 + 3] = seed + i + 3000;
  }

  // Run simulation in chunks with temporal averaging
  uint32_t num_outputs = num_steps / tavg_period;
  std::vector<float> output_data(num_outputs * num_svar * num_nodes);

  for (uint32_t chunk = 0; chunk < num_outputs; chunk++) {
    uint32_t t0 = chunk * tavg_period;
    step_batch<Model, width>(cx, c, states.data(), tavg.data(),
                             noise_scale.data(), params.data(), p_varies_node,
                             t0, tavg_period, dt, rng_seed.data());

    // Copy temporal average to output
    for (uint32_t i = 0; i < num_svar * num_nodes; i++) {
      output_data[chunk * num_svar * num_nodes + i] = tavg[i];
    }

    // Progress bar
    if (chunk % (num_outputs / 20 + 1) == 0 || chunk == num_outputs - 1) {
      float progress = (float)(chunk + 1) / num_outputs;
      int barWidth = 40;
      std::cout << "\r[" << std::string(int(barWidth * progress), '=')
                << std::string(barWidth - int(barWidth * progress), ' ') << "] "
                << int(progress * 100.0) << "%" << std::flush;
    }
  }

  std::cout << "\nSimulation complete. Writing output..." << std::endl;

  // Write output
  write_csv_timeseries(output_file, output_data.data(), num_outputs, num_nodes,
                       num_svar, dt * tavg_period);

  std::cout << "Output written to " << output_file << std::endl;
}

// Model dispatcher
void dispatch_model(const std::string &model_name, const ConnData &conn_data,
                    const ParamData &param_data, uint32_t num_nodes,
                    uint32_t num_steps, float dt, uint32_t tavg_period,
                    float noise_level, uint64_t seed,
                    const std::string &output_file) {
  // Neural mass models
  if (model_name == "wilson_cowan") {
    run_simulation<wilson_cowan>(conn_data, param_data, num_nodes, num_steps,
                                 dt, tavg_period, noise_level, seed,
                                 output_file);
  } else if (model_name == "reduced_wong_wang") {
    run_simulation<reduced_wong_wang>(conn_data, param_data, num_nodes,
                                      num_steps, dt, tavg_period, noise_level,
                                      seed, output_file);
  } else if (model_name == "reduced_wong_wang_exc_inh") {
    run_simulation<reduced_wong_wang_exc_inh>(conn_data, param_data, num_nodes,
                                              num_steps, dt, tavg_period,
                                              noise_level, seed, output_file);
  } else if (model_name == "deco_balanced_exc_inh") {
    run_simulation<deco_balanced_exc_inh>(conn_data, param_data, num_nodes,
                                          num_steps, dt, tavg_period,
                                          noise_level, seed, output_file);

    // Oscillator models
  } else if (model_name == "kuramoto") {
    run_simulation<kuramoto>(conn_data, param_data, num_nodes, num_steps, dt,
                             tavg_period, noise_level, seed, output_file);
  } else if (model_name == "generic_2d") {
    run_simulation<generic_2d>(conn_data, param_data, num_nodes, num_steps, dt,
                               tavg_period, noise_level, seed, output_file);
  } else if (model_name == "sup_hopf") {
    run_simulation<sup_hopf>(conn_data, param_data, num_nodes, num_steps, dt,
                             tavg_period, noise_level, seed, output_file);

    // Epileptor family
  } else if (model_name == "epileptor") {
    run_simulation<epileptor>(conn_data, param_data, num_nodes, num_steps, dt,
                              tavg_period, noise_level, seed, output_file);
  } else if (model_name == "epileptor_rs") {
    run_simulation<epileptor_rs>(conn_data, param_data, num_nodes, num_steps,
                                 dt, tavg_period, noise_level, seed,
                                 output_file);
  } else if (model_name == "epileptor_codim3") {
    run_simulation<epileptor_codim3>(conn_data, param_data, num_nodes,
                                     num_steps, dt, tavg_period, noise_level,
                                     seed, output_file);
  } else if (model_name == "epileptor_codim3_slow_mod") {
    run_simulation<epileptor_codim3_slow_mod>(conn_data, param_data, num_nodes,
                                              num_steps, dt, tavg_period,
                                              noise_level, seed, output_file);

    // Neural field models
  } else if (model_name == "coombes_byrne") {
    run_simulation<coombes_byrne>(conn_data, param_data, num_nodes, num_steps,
                                  dt, tavg_period, noise_level, seed,
                                  output_file);
  } else if (model_name == "gast_schmidt_knosche_sd") {
    run_simulation<gast_schmidt_knosche_sd>(conn_data, param_data, num_nodes,
                                            num_steps, dt, tavg_period,
                                            noise_level, seed, output_file);
  } else if (model_name == "gast_schmidt_knosche_sf") {
    run_simulation<gast_schmidt_knosche_sf>(conn_data, param_data, num_nodes,
                                            num_steps, dt, tavg_period,
                                            noise_level, seed, output_file);
  } else if (model_name == "zetterberg_jansen") {
    run_simulation<zetterberg_jansen>(conn_data, param_data, num_nodes,
                                      num_steps, dt, tavg_period, noise_level,
                                      seed, output_file);

    // Jansen-Rit and variants
  } else if (model_name == "jr") {
    run_simulation<jr>(conn_data, param_data, num_nodes, num_steps, dt,
                       tavg_period, noise_level, seed, output_file);

    // Mean-field models
  } else if (model_name == "mpr") {
    run_simulation<mpr>(conn_data, param_data, num_nodes, num_steps, dt,
                        tavg_period, noise_level, seed, output_file);
  } else if (model_name == "infinite_theta") {
    run_simulation<infinite_theta>(conn_data, param_data, num_nodes, num_steps,
                                   dt, tavg_period, noise_level, seed,
                                   output_file);
  } else if (model_name == "dumont_gutkin") {
    run_simulation<dumont_gutkin>(conn_data, param_data, num_nodes, num_steps,
                                  dt, tavg_period, noise_level, seed,
                                  output_file);

    // Reduced set models
  } else if (model_name == "reduced_set_fitz_hugh_nagumo") {
    run_simulation<reduced_set_fitz_hugh_nagumo>(
        conn_data, param_data, num_nodes, num_steps, dt, tavg_period,
        noise_level, seed, output_file);
  } else if (model_name == "reduced_set_hindmarsh_rose") {
    run_simulation<reduced_set_hindmarsh_rose>(conn_data, param_data, num_nodes,
                                               num_steps, dt, tavg_period,
                                               noise_level, seed, output_file);

    // Zerlaut models
  } else if (model_name == "zerlaut_adaptation_first_order") {
    run_simulation<zerlaut_adaptation_first_order>(
        conn_data, param_data, num_nodes, num_steps, dt, tavg_period,
        noise_level, seed, output_file);
  } else if (model_name == "zerlaut_adaptation_second_order") {
    run_simulation<zerlaut_adaptation_second_order>(
        conn_data, param_data, num_nodes, num_steps, dt, tavg_period,
        noise_level, seed, output_file);

    // Other models
  } else if (model_name == "larter_breakspear") {
    run_simulation<larter_breakspear>(conn_data, param_data, num_nodes,
                                      num_steps, dt, tavg_period, noise_level,
                                      seed, output_file);
  } else if (model_name == "hopfield") {
    run_simulation<hopfield>(conn_data, param_data, num_nodes, num_steps, dt,
                             tavg_period, noise_level, seed, output_file);
  } else if (model_name == "hopfield_dynamic") {
    run_simulation<hopfield_dynamic>(conn_data, param_data, num_nodes,
                                     num_steps, dt, tavg_period, noise_level,
                                     seed, output_file);
  } else if (model_name == "linear") {
    run_simulation<linear>(conn_data, param_data, num_nodes, num_steps, dt,
                           tavg_period, noise_level, seed, output_file);
  } else if (model_name == "kionex") {
    run_simulation<kionex>(conn_data, param_data, num_nodes, num_steps, dt,
                           tavg_period, noise_level, seed, output_file);
  } else if (model_name == "kionex2") {
    run_simulation<kionex2>(conn_data, param_data, num_nodes, num_steps, dt,
                            tavg_period, noise_level, seed, output_file);
  } else {
    std::cerr << "Error: Unknown model '" << model_name << "'" << std::endl;
    std::cerr << "Available models:\n"
              << "  Neural mass: wilson_cowan, reduced_wong_wang, "
                 "reduced_wong_wang_exc_inh, deco_balanced_exc_inh\n"
              << "  Oscillators: kuramoto, generic_2d, sup_hopf\n"
              << "  Epileptor: epileptor, epileptor_rs, epileptor_codim3, "
                 "epileptor_codim3_slow_mod\n"
              << "  Neural fields: coombes_byrne, gast_schmidt_knosche_sd, "
                 "gast_schmidt_knosche_sf, zetterberg_jansen\n"
              << "  Jansen-Rit: jr\n"
              << "  Mean-field: mpr, infinite_theta, dumont_gutkin\n"
              << "  Reduced sets: reduced_set_fitz_hugh_nagumo, "
                 "reduced_set_hindmarsh_rose\n"
              << "  Zerlaut: zerlaut_adaptation_first_order, "
                 "zerlaut_adaptation_second_order\n"
              << "  Other: larter_breakspear, hopfield, hopfield_dynamic, "
                 "linear, kionex, kionex2\n"
              << std::endl;
    exit(1);
  }
}

void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --model MODEL          Model name (required)\n";
  std::cout << "  --weights FILE         Weights CSV file (required)\n";
  std::cout << "  --lengths FILE         Lengths CSV file (optional, defaults "
               "to zeros)\n";
  std::cout << "  --params FILE          Parameters CSV file (required)\n";
  std::cout
      << "  --dt FLOAT             Integration time step (default: 0.1)\n";
  std::cout << "  --steps INT            Number of simulation steps (default: "
               "1000)\n";
  std::cout << "  --tavg-period INT      Temporal averaging period in steps "
               "(default: 1)\n";
  std::cout
      << "  --output FILE          Output CSV file (default: output.csv)\n";
  std::cout << "  --speed FLOAT          Conduction speed in mm/ms (default: "
               "3.0)\n";
  std::cout << "  --noise FLOAT          Noise level sigma (default: 0.0)\n";
  std::cout << "  --seed INT             Random seed (default: 42)\n";
  std::cout << "  --help                 Show this help message\n";
  std::cout << "\nAvailable models (28 total):\n";
  std::cout << "  Neural mass: wilson_cowan, reduced_wong_wang, "
               "reduced_wong_wang_exc_inh,\n";
  std::cout << "               deco_balanced_exc_inh\n";
  std::cout << "  Oscillators: kuramoto, generic_2d, sup_hopf\n";
  std::cout << "  Epileptor: epileptor, epileptor_rs, epileptor_codim3, "
               "epileptor_codim3_slow_mod\n";
  std::cout << "  Neural fields: coombes_byrne, gast_schmidt_knosche_sd,\n";
  std::cout << "                 gast_schmidt_knosche_sf, zetterberg_jansen\n";
  std::cout << "  Jansen-Rit: jr\n";
  std::cout << "  Mean-field: mpr, infinite_theta, dumont_gutkin\n";
  std::cout << "  Reduced sets: reduced_set_fitz_hugh_nagumo, "
               "reduced_set_hindmarsh_rose\n";
  std::cout << "  Zerlaut: zerlaut_adaptation_first_order, "
               "zerlaut_adaptation_second_order\n";
  std::cout
      << "  Other: larter_breakspear, hopfield, hopfield_dynamic, linear,\n";
  std::cout << "         kionex, kionex2\n";
}

int main(int argc, char **argv) {
  // Default parameters
  std::string model_name;
  std::string weights_file;
  std::string lengths_file;
  std::string params_file;
  std::string output_file = "output.csv";
  float dt = 0.1f;
  uint32_t num_steps = 1000;
  uint32_t tavg_period = 1;
  float speed = 3.0f;
  float noise_level = 0.0f;
  uint64_t seed = 42;

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--weights" && i + 1 < argc) {
      weights_file = argv[++i];
    } else if (arg == "--lengths" && i + 1 < argc) {
      lengths_file = argv[++i];
    } else if (arg == "--params" && i + 1 < argc) {
      params_file = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--dt" && i + 1 < argc) {
      dt = std::stof(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      num_steps = std::stoul(argv[++i]);
    } else if (arg == "--tavg-period" && i + 1 < argc) {
      tavg_period = std::stoul(argv[++i]);
    } else if (arg == "--speed" && i + 1 < argc) {
      speed = std::stof(argv[++i]);
    } else if (arg == "--noise" && i + 1 < argc) {
      noise_level = std::stof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  // Validate required arguments
  if (model_name.empty()) {
    std::cerr << "Error: --model is required" << std::endl;
    print_usage(argv[0]);
    return 1;
  }
  if (weights_file.empty()) {
    std::cerr << "Error: --weights is required" << std::endl;
    print_usage(argv[0]);
    return 1;
  }
  if (params_file.empty()) {
    std::cerr << "Error: --params is required" << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // Read connectivity
  std::cout << "Reading weights from " << weights_file << "..." << std::endl;
  auto weights = read_csv_matrix(weights_file);
  uint32_t num_nodes = weights.size();

  std::vector<std::vector<float>> lengths;
  if (!lengths_file.empty()) {
    std::cout << "Reading lengths from " << lengths_file << "..." << std::endl;
    lengths = read_csv_matrix(lengths_file);
    if (lengths.size() != num_nodes) {
      std::cerr << "Error: Lengths matrix size mismatch" << std::endl;
      return 1;
    }
  } else {
    // Default to zero lengths
    lengths.resize(num_nodes, std::vector<float>(num_nodes, 0.0f));
  }

  // Read parameters
  std::cout << "Reading parameters from " << params_file << "..." << std::endl;
  auto params = read_csv_params(params_file);

  // Convert to CSR format
  auto conn_data = matrix_to_csr(weights, lengths, speed, dt);

  // Run simulation
  dispatch_model(model_name, conn_data, params, num_nodes, num_steps, dt,
                 tavg_period, noise_level, seed, output_file);

  return 0;
}

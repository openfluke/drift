package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/openfluke/drift"
	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== DRIFT: Multi-Agent Neural Network Config ===")
	fmt.Println()

	// Create a DRIFT config with two neural network models
	cfg := drift.NewConfig("MultiAgentSwarm")

	// Example 1: Heterogeneous Agent Swarm (LSTM, MHA, Dense ensemble, RNN)
	example1 := json.RawMessage(`{
		"batch_size": 2,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 20,
				"output_size": 32,
				"activation": "relu",
				"comment": "Shared perception layer"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 2,
				"grid_output_cols": 2,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0},
					{"branch_index": 2, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 3, "target_row": 1, "target_col": 1, "target_layer": 0}
				],
				"branches": [
					{
						"type": "lstm",
						"input_size": 32,
						"hidden_size": 10,
						"seq_length": 1,
						"comment": "Agent 0: Scout (temporal memory)"
					},
					{
						"type": "mha",
						"d_model": 32,
						"num_heads": 4,
						"seq_length": 1,
						"comment": "Agent 1: Analyzer (attention-based)"
					},
					{
						"type": "parallel",
						"combine_mode": "add",
						"branches": [
							{"type": "dense", "input_size": 32, "output_size": 10, "activation": "relu"},
							{"type": "dense", "input_size": 32, "output_size": 10, "activation": "gelu"},
							{"type": "dense", "input_size": 32, "output_size": 10, "activation": "tanh"}
						],
						"comment": "Agent 2: Executor (ensemble decision)"
					},
					{
						"type": "rnn",
						"input_size": 32,
						"hidden_size": 10,
						"seq_length": 1,
						"comment": "Agent 3: Coordinator (sequential processing)"
					}
				]
			}
		]
	}`)

	// Example 2: Multi-Scale Processing (LayerNorm, RMSNorm, SwiGLU)
	example2 := json.RawMessage(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 24,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 3,
				"grid_output_cols": 1,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 2, "target_row": 2, "target_col": 0, "target_layer": 0}
				],
				"branches": [
					{
						"type": "layer_norm",
						"norm_size": 24,
						"epsilon": 1e-5,
						"comment": "Agent 0: LayerNorm processor"
					},
					{
						"type": "rms_norm",
						"norm_size": 24,
						"epsilon": 1e-5,
						"comment": "Agent 1: RMSNorm processor (Llama-style)"
					},
					{
						"type": "swiglu",
						"input_size": 24,
						"output_size": 24,
						"comment": "Agent 2: SwiGLU gated processor"
					}
				]
			}
		]
	}`)

	// Add models to config
	cfg.Models["agent_swarm"] = example1
	cfg.Models["multi_scale"] = example2

	// Save to file
	err := cfg.SaveToFile("drift_config.json")
	if err != nil {
		log.Fatalf("Failed to save config: %v", err)
	}
	fmt.Println("✓ Saved config to drift_config.json")

	// Load from file
	loaded, err := drift.LoadFromFile("drift_config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	fmt.Printf("✓ Loaded config: %s (%d models)\n", loaded.GetName(), len(loaded.Models))
	fmt.Println()

	// Build and run Network 1: Agent Swarm
	fmt.Println("=== Network 1: Heterogeneous Agent Swarm ===")
	net1, err := nn.BuildNetworkFromJSON(string(loaded.Models["agent_swarm"]))
	if err != nil {
		log.Fatalf("Failed to build network 1: %v", err)
	}
	net1.InitializeWeights()

	input1 := make([]float32, 2*20)
	for i := range input1 {
		input1[i] = float32(i%20) * 0.05
	}

	output1, _ := net1.ForwardCPU(input1)
	fmt.Printf("Input: [batch=2, sensor_data=20]\n")
	fmt.Printf("Agent Grid (2x2):\n")
	fmt.Printf("  ┌──────────────────┬──────────────────┐\n")
	fmt.Printf("  │ Scout (LSTM)     │ Analyzer (MHA)   │\n")
	fmt.Printf("  ├──────────────────┼──────────────────┤\n")
	fmt.Printf("  │ Executor (Dense) │ Coordinator (RNN)│\n")
	fmt.Printf("  └──────────────────┴──────────────────┘\n")
	fmt.Printf("Output: %d values\n", len(output1))
	fmt.Printf("Scout (batch 1): %v\n", output1[0:10])
	fmt.Println()

	// Build and run Network 2: Multi-Scale
	fmt.Println("=== Network 2: Multi-Scale Processing ===")
	net2, err := nn.BuildNetworkFromJSON(string(loaded.Models["multi_scale"]))
	if err != nil {
		log.Fatalf("Failed to build network 2: %v", err)
	}
	net2.InitializeWeights()

	input2 := make([]float32, 24)
	for i := range input2 {
		input2[i] = float32(i)*0.1 - 1.0
	}

	output2, _ := net2.ForwardCPU(input2)
	fmt.Printf("Input: [24 features]\n")
	fmt.Printf("Agent Grid (3x1 vertical):\n")
	fmt.Printf("  ┌──────────────────────┐\n")
	fmt.Printf("  │ LayerNorm (stable)   │\n")
	fmt.Printf("  ├──────────────────────┤\n")
	fmt.Printf("  │ RMSNorm (efficient)  │\n")
	fmt.Printf("  ├──────────────────────┤\n")
	fmt.Printf("  │ SwiGLU (gated)       │\n")
	fmt.Printf("  └──────────────────────┘\n")
	fmt.Printf("Output: %d values\n", len(output2))
	fmt.Printf("LayerNorm: %v\n", output2[0:10])
	fmt.Println()

	fmt.Println("=== DRIFT Demo Complete ===")
}

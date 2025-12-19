package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/openfluke/drift"
	"github.com/openfluke/loom/nn"
)

// ============================================================================
// DRIFT Neural Link Experiment: Emergent Terrain Adaptation
// ============================================================================
//
// This experiment uses DRIFT configuration to define:
// 1. Model architectures (Classifier + Navigator)
// 2. Neural link configuration (which layers to connect)
// 3. Training parameters
//
// The neural link passes hidden layer activations from Classifier → Navigator
// ============================================================================

const (
	TerrainRoad = 0
	TerrainSand = 1
	NumTerrains = 2

	ActionUp    = 0
	ActionDown  = 1
	ActionLeft  = 2
	ActionRight = 3
	NumActions  = 4
)

var terrainNames = []string{"Road", "Sand"}
var actionNames = []string{"Up", "Down", "Left", "Right"}

type Environment struct {
	AgentPos   [2]float32
	TargetPos  [2]float32
	Terrain    int
	LastAction int
	StuckCount int
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  DRIFT: Neural Link Experiment - Configuration-Driven                   ║")
	fmt.Println("║  Models and neural links defined via DRIFT config                       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ========================================
	// Create DRIFT Configuration
	// ========================================
	cfg := drift.NewConfig("NeuralLinkExperiment")

	// Model 1: Terrain Classifier (Dense)
	// Hidden layer at index 1 is the neural link source
	classifierDef := json.RawMessage(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"layers": [
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 32,
				"activation": "leaky_relu",
				"comment": "Input layer - sensor processing"
			},
			{
				"type": "dense",
				"input_size": 32,
				"output_size": 16,
				"activation": "leaky_relu",
				"comment": "Hidden layer - NEURAL LINK SOURCE"
			},
			{
				"type": "dense",
				"input_size": 16,
				"output_size": 2,
				"activation": "sigmoid",
				"comment": "Output layer - terrain classification"
			}
		]
	}`)

	// Model 2: Movement Navigator (with LSTM)
	// Input includes 4 basic inputs + 16 from neural link = 20 total
	navigatorDef := json.RawMessage(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 4,
		"layers": [
			{
				"type": "dense",
				"input_size": 20,
				"output_size": 32,
				"activation": "leaky_relu",
				"comment": "Input layer - position + NEURAL LINK INPUT"
			},
			{
				"type": "lstm",
				"input_size": 8,
				"hidden_size": 8,
				"seq_length": 4,
				"comment": "Temporal reasoning layer"
			},
			{
				"type": "dense",
				"input_size": 32,
				"output_size": 16,
				"activation": "leaky_relu",
				"comment": "Decision layer"
			},
			{
				"type": "dense",
				"input_size": 16,
				"output_size": 4,
				"activation": "sigmoid",
				"comment": "Output layer - movement actions"
			}
		]
	}`)

	// Neural Link Configuration - using drift.NeuralLinkConfig
	cfg.AddLink(drift.NeuralLinkConfig{
		Name:         "classifier_to_navigator",
		SourceModel:  "classifier",
		SourceLayer:  1,
		TargetModel:  "navigator",
		TargetOffset: 4,
		LinkSize:     16,
		Enabled:      true,
		Description:  "Classifier hidden activations → Navigator input[4:20]",
	})

	// Training Configuration
	trainingDef := json.RawMessage(`{
		"classifier": {
			"duration_seconds": 5,
			"learning_rate": 0.02,
			"mode": "step_tween_chain"
		},
		"navigator": {
			"duration_seconds": 5,
			"learning_rate": 0.02,
			"mode": "step_tween_chain"
		},
		"neural_link_test": {
			"duration_seconds": 3,
			"learning_rate": 0.01
		}
	}`)

	// Add models to config
	cfg.Models["classifier"] = classifierDef
	cfg.Models["navigator"] = navigatorDef
	cfg.Models["training"] = trainingDef

	// Save config
	err := cfg.SaveToFile("drift_config.json")
	if err != nil {
		log.Fatalf("Failed to save config: %v", err)
	}
	fmt.Println("✓ Saved DRIFT config to drift_config.json")

	// Load and parse config
	loaded, err := drift.LoadFromFile("drift_config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	fmt.Printf("✓ Loaded config: %s (%d models, %d links)\n", loaded.GetName(), len(loaded.Models), len(loaded.GetLinks()))

	// Get neural link config from loaded config
	links := loaded.GetLinks()
	if len(links) == 0 {
		log.Fatal("No neural links defined in config")
	}
	linkConfig := links[0]
	fmt.Printf("✓ Neural Link: %s[layer %d] → %s[offset %d], size=%d\n",
		linkConfig.SourceModel, linkConfig.SourceLayer,
		linkConfig.TargetModel, linkConfig.TargetOffset, linkConfig.LinkSize)
	fmt.Println()

	// ========================================
	// Build Models from Config
	// ========================================
	fmt.Println("═══ Building Models from DRIFT Config ═══")

	classifier, err := nn.BuildNetworkFromJSON(string(loaded.Models["classifier"]))
	if err != nil {
		log.Fatalf("Failed to build classifier: %v", err)
	}
	classifier.InitializeWeights()
	fmt.Println("✓ Built Classifier network")

	navigator, err := nn.BuildNetworkFromJSON(string(loaded.Models["navigator"]))
	if err != nil {
		log.Fatalf("Failed to build navigator: %v", err)
	}
	navigator.InitializeWeights()
	fmt.Println("✓ Built Navigator network")
	fmt.Println()

	// ========================================
	// Phase 1: Train Classifier
	// ========================================
	fmt.Println("═══ PHASE 1: Training Terrain Classifier ═══")
	trainClassifier(classifier, 5*time.Second)

	// ========================================
	// Phase 2: Train Navigator (Road Only)
	// ========================================
	fmt.Println()
	fmt.Println("═══ PHASE 2: Training Navigator (ROAD ONLY) ═══")
	trainNavigatorRoadOnly(navigator, linkConfig.LinkSize, 5*time.Second)

	// ========================================
	// Phase 3: Neural Link Experiment
	// ========================================
	fmt.Println()
	fmt.Println("═══ PHASE 3: Neural Link Experiment ═══")
	runNeuralLinkExperiment(classifier, navigator, linkConfig)

	// Delete the config file (gitignored anyway)
	os.Remove("drift_config.json")
}

// ============================================================================
// Training Functions
// ============================================================================

func trainClassifier(net *nn.Network, duration time.Duration) {
	state := net.InitStepState(8)
	tween := nn.NewTweenState(net, nil)
	tween.Config.UseChainRule = true

	correct, total := 0, 0
	lr := float32(0.02)
	start := time.Now()

	for time.Since(start) < duration {
		terrain := rand.Intn(NumTerrains)
		sensors := generateSensorData(terrain)

		state.SetInput(sensors)
		net.StepForward(state)
		output := state.GetOutput()

		if argmax(output) == terrain {
			correct++
		}
		total++

		tween.TweenStep(net, sensors, terrain, NumTerrains, lr)
	}

	acc := float64(correct) / float64(total) * 100
	fmt.Printf("Classifier trained: %.1f%% accuracy (%d samples)\n", acc, total)
}

func trainNavigatorRoadOnly(net *nn.Network, linkSize int, duration time.Duration) {
	inputSize := 4 + linkSize
	state := net.InitStepState(inputSize)
	tween := nn.NewTweenState(net, nil)
	tween.Config.UseChainRule = true

	correct, total := 0, 0
	lr := float32(0.02)
	start := time.Now()

	env := &Environment{
		AgentPos:  [2]float32{0.5, 0.5},
		TargetPos: [2]float32{rand.Float32(), rand.Float32()},
		Terrain:   TerrainRoad,
	}

	for time.Since(start) < duration {
		navInput := buildNavigatorInput(env, nil, linkSize)

		state.SetInput(navInput)
		net.StepForward(state)
		output := state.GetOutput()

		predicted := argmax(output)
		optimal := getOptimalAction(env)

		if predicted == optimal {
			correct++
		}
		total++

		tween.TweenStep(net, navInput, optimal, NumActions, lr)

		executeAction(env, predicted)
		if rand.Float32() < 0.1 {
			env.TargetPos = [2]float32{rand.Float32(), rand.Float32()}
		}
	}

	acc := float64(correct) / float64(total) * 100
	fmt.Printf("Navigator trained (road only): %.1f%% accuracy (%d samples)\n", acc, total)
	fmt.Println("⚠ Navigator has NEVER seen sand!")
}

// ============================================================================
// Neural Link Experiment
// ============================================================================

func runNeuralLinkExperiment(classifier, navigator *nn.Network, linkConfig drift.NeuralLinkConfig) {
	fmt.Println()
	fmt.Println("Testing WITHOUT vs WITH Neural Link on SAND terrain:")
	fmt.Println()

	fmt.Println("┌──────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST A: Navigator on Sand WITHOUT Neural Link                           │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────┘")
	scoreWithout := testNavigation(classifier, navigator, linkConfig, false, 3*time.Second)

	fmt.Println()
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST B: Navigator on Sand WITH Neural Link (RL adaptation)              │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────┘")
	scoreWith := testNavigation(classifier, navigator, linkConfig, true, 3*time.Second)

	// Results
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           RESULTS                                        ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  Without Neural Link: %.0f targets                                       ║\n", scoreWithout)
	fmt.Printf("║  With Neural Link:    %.0f targets                                       ║\n", scoreWith)
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════╣")

	if scoreWith > scoreWithout*1.1 {
		fmt.Println("║  ✓ NEURAL LINK ENABLES EMERGENT ADAPTATION!                             ║")
	} else {
		fmt.Println("║  ⚠ Results inconclusive                                                 ║")
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
}

func testNavigation(classifier, navigator *nn.Network, linkConfig drift.NeuralLinkConfig, useNeuralLink bool, duration time.Duration) float64 {
	inputSize := 4 + linkConfig.LinkSize
	classifierState := classifier.InitStepState(8)
	navigatorState := navigator.InitStepState(inputSize)

	var tween *nn.TweenState
	if useNeuralLink {
		tween = nn.NewTweenState(navigator, nil)
		tween.Config.UseChainRule = true
	}

	env := &Environment{
		AgentPos:   [2]float32{0.1, 0.1},
		TargetPos:  [2]float32{0.9, 0.9},
		Terrain:    TerrainSand,
		LastAction: -1,
	}

	targetsReached := 0
	totalSteps := 0
	lr := float32(0.01)
	start := time.Now()

	for time.Since(start) < duration {
		var linkData []float32
		if useNeuralLink {
			sensors := generateSensorData(env.Terrain)
			classifierState.SetInput(sensors)
			classifier.StepForward(classifierState)
			linkData = getHiddenActivations(classifierState, linkConfig.SourceLayer, linkConfig.LinkSize)
		}

		navInput := buildNavigatorInput(env, linkData, linkConfig.LinkSize)
		navigatorState.SetInput(navInput)
		navigator.StepForward(navigatorState)
		output := navigatorState.GetOutput()

		action := argmax(output)
		prevDist := distanceToTarget(env)
		executeActionWithPhysics(env, action)
		newDist := distanceToTarget(env)
		totalSteps++

		if useNeuralLink && tween != nil {
			gotCloser := newDist < prevDist-0.001
			if gotCloser {
				tween.TweenStep(navigator, navInput, action, NumActions, lr)
			} else {
				var altAction int
				if env.LastAction == ActionUp || env.LastAction == ActionDown {
					if env.TargetPos[0] > env.AgentPos[0] {
						altAction = ActionRight
					} else {
						altAction = ActionLeft
					}
				} else {
					if env.TargetPos[1] > env.AgentPos[1] {
						altAction = ActionUp
					} else {
						altAction = ActionDown
					}
				}
				tween.TweenStep(navigator, navInput, altAction, NumActions, lr)
			}
		}

		if newDist < 0.1 {
			targetsReached++
			env.AgentPos = [2]float32{rand.Float32() * 0.3, rand.Float32() * 0.3}
			env.TargetPos = [2]float32{0.7 + rand.Float32()*0.3, 0.7 + rand.Float32()*0.3}
			env.LastAction = -1
			env.StuckCount = 0
		}
	}

	stepsPerTarget := float64(0)
	if targetsReached > 0 {
		stepsPerTarget = float64(totalSteps) / float64(targetsReached)
	}
	fmt.Printf("  Targets reached: %d in %d steps (%.0f steps/target)\n",
		targetsReached, totalSteps, stepsPerTarget)

	if useNeuralLink {
		fmt.Println("  ★ Navigator received classifier activations & learned online!")
	} else {
		fmt.Println("  ⚠ Navigator blind to terrain - using road strategy on sand")
	}

	return float64(targetsReached)
}

// ============================================================================
// Helper Functions
// ============================================================================

func generateSensorData(terrain int) []float32 {
	sensors := make([]float32, 8)
	switch terrain {
	case TerrainRoad:
		sensors = []float32{0.9, 0.1, 0.7, 0.2, 0.5, 0.3, 0.8, 0.95}
	case TerrainSand:
		sensors = []float32{0.3, 0.85, 0.9, 0.7, 0.75, 0.2, 0.25, 0.35}
	}
	for i := range sensors {
		sensors[i] += (rand.Float32() - 0.5) * 0.15
		sensors[i] = clamp(sensors[i], 0, 1)
	}
	return sensors
}

func buildNavigatorInput(env *Environment, linkData []float32, linkSize int) []float32 {
	input := make([]float32, 4+linkSize)

	dx := env.TargetPos[0] - env.AgentPos[0]
	dy := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if dist > 0.001 {
		input[0] = dx / dist
		input[1] = dy / dist
	}
	input[2] = env.AgentPos[0]
	input[3] = env.AgentPos[1]

	if linkData != nil {
		for i := 0; i < linkSize && i < len(linkData); i++ {
			input[4+i] = linkData[i]
		}
	}

	return input
}

func getHiddenActivations(state *nn.StepState, layerIdx, linkSize int) []float32 {
	hidden := state.GetLayerOutput(layerIdx)
	result := make([]float32, linkSize)
	for i := 0; i < linkSize && i < len(hidden); i++ {
		result[i] = hidden[i]
	}
	return result
}

func getOptimalAction(env *Environment) int {
	dx := env.TargetPos[0] - env.AgentPos[0]
	dy := env.TargetPos[1] - env.AgentPos[1]
	if abs(dx) > abs(dy) {
		if dx > 0 {
			return ActionRight
		}
		return ActionLeft
	}
	if dy > 0 {
		return ActionUp
	}
	return ActionDown
}

func executeAction(env *Environment, action int) {
	speed := float32(0.02)
	moves := [][2]float32{{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}}
	if action >= 0 && action < 4 {
		env.AgentPos[0] = clamp(env.AgentPos[0]+moves[action][0], 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moves[action][1], 0, 1)
	}
}

func executeActionWithPhysics(env *Environment, action int) {
	speed := float32(0.02)

	if env.Terrain == TerrainSand {
		if action == env.LastAction {
			env.StuckCount++
			if env.StuckCount > 2 {
				speed = 0
			} else {
				speed *= 0.3
			}
		} else {
			env.StuckCount = 0
			speed *= 1.5
		}
	}

	moves := [][2]float32{{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}}
	if action >= 0 && action < 4 {
		env.AgentPos[0] = clamp(env.AgentPos[0]+moves[action][0], 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moves[action][1], 0, 1)
	}
	env.LastAction = action
}

func distanceToTarget(env *Environment) float32 {
	dx := env.TargetPos[0] - env.AgentPos[0]
	dy := env.TargetPos[1] - env.AgentPos[1]
	return float32(math.Sqrt(float64(dx*dx + dy*dy)))
}

func argmax(s []float32) int {
	if len(s) == 0 {
		return 0
	}
	maxI, maxV := 0, s[0]
	for i, v := range s {
		if v > maxV {
			maxV, maxI = v, i
		}
	}
	return maxI
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

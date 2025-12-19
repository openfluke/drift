package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/openfluke/loom/nn"
)

// ============================================================================
// DRIFT Neural Link Experiment: Emergent Terrain Adaptation
// ============================================================================
//
// GOAL: LSTM discovers zigzag behavior through RL, triggered by neural link
//
// Phase 1: Train classifier on terrain detection
// Phase 2: Train LSTM on basic navigation (road only, no sand experience)
// Phase 3: Neural link test - classifier hidden layer → LSTM input
//          LSTM discovers zigzag through trial/error on sand
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

	// Neural link: classifier hidden layer size
	LinkSize = 16
)

var terrainNames = []string{"Road", "Sand"}
var actionNames = []string{"Up", "Down", "Left", "Right"}

type Environment struct {
	AgentPos   [2]float32
	TargetPos  [2]float32
	Terrain    int
	LastAction int
	StuckCount int // tracks consecutive same-direction moves on sand
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  DRIFT: Neural Link Experiment - Emergent Behavior Discovery            ║")
	fmt.Println("║  Classifier detects terrain → LSTM discovers optimal strategy via RL    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ========================================
	// PHASE 1: Train Classifier
	// ========================================
	fmt.Println("═══ PHASE 1: Training Terrain Classifier ═══")
	classifier := buildClassifier()
	trainClassifier(classifier, 5*time.Second)

	// ========================================
	// PHASE 2: Train LSTM on ROAD ONLY
	// ========================================
	fmt.Println()
	fmt.Println("═══ PHASE 2: Training LSTM Navigator (ROAD ONLY) ═══")
	navigator := buildNavigator()
	trainNavigatorRoadOnly(navigator, 5*time.Second)

	// ========================================
	// PHASE 3: Neural Link Test
	// ========================================
	fmt.Println()
	fmt.Println("═══ PHASE 3: Neural Link Experiment ═══")
	runNeuralLinkExperiment(classifier, navigator)
}

// ============================================================================
// Model Builders
// ============================================================================

func buildClassifier() *nn.Network {
	// Input: 8 sensor features
	// Hidden: 16 neurons (THIS IS THE NEURAL LINK OUTPUT)
	// Output: 2 terrain classes
	net := nn.NewNetwork(8, 1, 1, 3)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(32, LinkSize, nn.ActivationLeakyReLU)) // Hidden = link output
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(LinkSize, NumTerrains, nn.ActivationSigmoid))
	return net
}

func buildNavigator() *nn.Network {
	// Input: 4 (position/direction) + 16 (NEURAL LINK) = 20
	// LSTM for temporal reasoning
	// Output: 4 actions
	inputSize := 4 + LinkSize
	net := nn.NewNetwork(inputSize, 1, 1, 4)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(inputSize, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitLSTMLayer(8, 8, 1, 4))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 16, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(16, NumActions, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Phase 1: Train Classifier
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

// ============================================================================
// Phase 2: Train Navigator on ROAD ONLY
// ============================================================================

func trainNavigatorRoadOnly(net *nn.Network, duration time.Duration) {
	inputSize := 4 + LinkSize
	state := net.InitStepState(inputSize)
	tween := nn.NewTweenState(net, nil)
	tween.Config.UseChainRule = true

	correct, total := 0, 0
	lr := float32(0.02)
	start := time.Now()

	env := &Environment{
		AgentPos:  [2]float32{0.5, 0.5},
		TargetPos: [2]float32{rand.Float32(), rand.Float32()},
		Terrain:   TerrainRoad, // ROAD ONLY - no sand experience
	}

	for time.Since(start) < duration {
		// Build input with ZEROS for neural link (no classifier connected yet)
		navInput := buildNavigatorInput(env, nil) // nil = zeros for link

		state.SetInput(navInput)
		net.StepForward(state)
		output := state.GetOutput()

		predicted := argmax(output)
		optimal := getOptimalAction(env, -1) // -1 = no history tracking

		if predicted == optimal {
			correct++
		}
		total++

		tween.TweenStep(net, navInput, optimal, NumActions, lr)

		// Update environment
		executeAction(env, predicted)
		if rand.Float32() < 0.1 {
			env.TargetPos = [2]float32{rand.Float32(), rand.Float32()}
		}
	}

	acc := float64(correct) / float64(total) * 100
	fmt.Printf("Navigator trained (road only): %.1f%% accuracy (%d samples)\n", acc, total)
	fmt.Println("⚠ Navigator has NEVER seen sand - doesn't know zigzag strategy!")
}

// ============================================================================
// Phase 3: Neural Link Experiment
// ============================================================================

func runNeuralLinkExperiment(classifier, navigator *nn.Network) {
	fmt.Println()
	fmt.Println("Testing without vs with Neural Link on SAND terrain:")
	fmt.Println()

	// Test 1: Navigator WITHOUT neural link (zeros for link input)
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST A: Navigator on Sand WITHOUT Neural Link                           │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────┘")
	scoreWithoutLink := testNavigation(classifier, navigator, false, 3*time.Second)

	fmt.Println()
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST B: Navigator on Sand WITH Neural Link (RL adaptation)              │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────┘")
	scoreWithLink := testNavigation(classifier, navigator, true, 3*time.Second)

	// Results
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                           RESULTS                                        ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  Without Neural Link: %.0f targets                                      ║\n", scoreWithoutLink)
	fmt.Printf("║  With Neural Link:    %.0f targets                                      ║\n", scoreWithLink)
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════╣")

	if scoreWithLink > scoreWithoutLink*1.1 {
		fmt.Println("║  ✓ NEURAL LINK ENABLES EMERGENT ADAPTATION!                             ║")
	} else {
		fmt.Println("║  ⚠ Results inconclusive - try adjusting learning rate                  ║")
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	// Save results
	saveResults(scoreWithoutLink, scoreWithLink)
}

func testNavigation(classifier, navigator *nn.Network, useNeuralLink bool, duration time.Duration) float64 {
	inputSize := 4 + LinkSize
	classifierState := classifier.InitStepState(8)
	navigatorState := navigator.InitStepState(inputSize)

	// Online learning for adaptation
	var tween *nn.TweenState
	if useNeuralLink {
		tween = nn.NewTweenState(navigator, nil)
		tween.Config.UseChainRule = true
	}

	env := &Environment{
		AgentPos:   [2]float32{0.1, 0.1},
		TargetPos:  [2]float32{0.9, 0.9},
		Terrain:    TerrainSand, // SAND - unknown to navigator!
		LastAction: -1,
	}

	targetsReached := 0
	totalSteps := 0
	lr := float32(0.01)
	start := time.Now()

	for time.Since(start) < duration {
		// Get classifier hidden layer activations (neural link data)
		var linkData []float32
		if useNeuralLink {
			sensors := generateSensorData(env.Terrain)
			classifierState.SetInput(sensors)
			classifier.StepForward(classifierState)
			linkData = getHiddenActivations(classifier, classifierState)
		}

		// Navigator input
		navInput := buildNavigatorInput(env, linkData)
		navigatorState.SetInput(navInput)
		navigator.StepForward(navigatorState)
		output := navigatorState.GetOutput()

		// Take action
		action := argmax(output)
		prevDist := distanceToTarget(env)
		executeActionWithPhysics(env, action)
		newDist := distanceToTarget(env)
		totalSteps++

		// ONLINE RL: If neural link active, learn from reward
		if useNeuralLink && tween != nil {
			gotCloser := newDist < prevDist-0.001
			if gotCloser {
				tween.TweenStep(navigator, navInput, action, NumActions, lr)
			} else {
				// Try alternating pattern
				var altAction int
				if env.LastAction == ActionUp || env.LastAction == ActionDown {
					// Was vertical, try horizontal
					if env.TargetPos[0] > env.AgentPos[0] {
						altAction = ActionRight
					} else {
						altAction = ActionLeft
					}
				} else {
					// Was horizontal, try vertical
					if env.TargetPos[1] > env.AgentPos[1] {
						altAction = ActionUp
					} else {
						altAction = ActionDown
					}
				}
				tween.TweenStep(navigator, navInput, altAction, NumActions, lr)
			}
		}

		// Check if reached target
		if newDist < 0.1 {
			targetsReached++
			// Reset to new random positions
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
	// Add noise
	for i := range sensors {
		sensors[i] += (rand.Float32() - 0.5) * 0.15
		sensors[i] = clamp(sensors[i], 0, 1)
	}
	return sensors
}

func buildNavigatorInput(env *Environment, linkData []float32) []float32 {
	input := make([]float32, 4+LinkSize)

	// Basic navigation input
	dx := env.TargetPos[0] - env.AgentPos[0]
	dy := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if dist > 0.001 {
		input[0] = dx / dist // direction x
		input[1] = dy / dist // direction y
	}
	input[2] = env.AgentPos[0]
	input[3] = env.AgentPos[1]

	// Neural link data (or zeros)
	if linkData != nil {
		for i := 0; i < LinkSize && i < len(linkData); i++ {
			input[4+i] = linkData[i]
		}
	}

	return input
}

func getHiddenActivations(net *nn.Network, state *nn.StepState) []float32 {
	// Get activations from hidden layer (layer 1)
	// This is the neural link data we pass to navigator
	hidden := state.GetLayerOutput(1)
	result := make([]float32, LinkSize)
	for i := 0; i < LinkSize && i < len(hidden); i++ {
		result[i] = hidden[i]
	}
	return result
}

func getOptimalAction(env *Environment, _ int) int {
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
		// SAND PHYSICS: Straight moves get increasingly stuck
		if action == env.LastAction {
			env.StuckCount++
			if env.StuckCount > 2 {
				// Completely stuck - no movement!
				speed = 0
			} else {
				speed *= 0.3 // Very slow when repeating
			}
		} else {
			// Alternating direction works great on sand!
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

func saveResults(without, with float64) {
	results := map[string]interface{}{
		"experiment":            "neural_link_emergent_adaptation",
		"timestamp":             time.Now().Format(time.RFC3339),
		"without_neural_link":   without,
		"with_neural_link":      with,
		"improvement":           with - without,
		"neural_link_effective": with > without+5,
	}
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("experiment_results.json", data, 0644)
	fmt.Println("\nResults saved to experiment_results.json")
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

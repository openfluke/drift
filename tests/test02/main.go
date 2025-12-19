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
// DRIFT Neural Link Experiment: Multi-Terrain Adaptation Benchmark
// ============================================================================
//
// Terrains: Road, Sand, Ice, Grass (each requires different movement strategy)
// Sequence: Road → Sand → Road → Grass → Road → Ice → Road
//
// Training Modes:
// 1. LSTM Only (baseline - no learning during test)
// 2. LSTM + RL (reinforcement learning, no neural link)
// 3. LSTM + Neural Link (link active, no RL)
// 4. LSTM + Neural Link + RL (full system)
//
// Metrics: Accumulated every 500ms windows
// ============================================================================

const (
	TerrainRoad = iota
	TerrainSand
	TerrainIce
	TerrainGrass
	NumTerrains
)

const (
	ActionUp = iota
	ActionDown
	ActionLeft
	ActionRight
	NumActions
)

var terrainNames = []string{"Road", "Sand", "Ice", "Grass"}
var actionNames = []string{"Up", "Down", "Left", "Right"}

// Terrain sequence for the experiment
var terrainSequence = []int{
	TerrainRoad, TerrainSand, TerrainRoad, TerrainGrass,
	TerrainRoad, TerrainIce, TerrainRoad,
}

type Environment struct {
	AgentPos   [2]float32
	TargetPos  [2]float32
	Terrain    int
	LastAction int
	StuckCount int
	IceVelX    float32 // momentum for ice
	IceVelY    float32
}

// WindowMetrics tracks performance in 500ms windows
type WindowMetrics struct {
	WindowNum      int     `json:"window"`
	Terrain        string  `json:"terrain"`
	TargetsReached int     `json:"targets"`
	TotalSteps     int     `json:"steps"`
	EffectiveMoves int     `json:"effective_moves"`
	Accuracy       float64 `json:"accuracy_pct"`
}

// ExperimentResult holds results for one training mode
type ExperimentResult struct {
	Mode           string          `json:"mode"`
	Windows        []WindowMetrics `json:"windows"`
	TotalTargets   int             `json:"total_targets"`
	TotalSteps     int             `json:"total_steps"`
	FinalAccuracy  float64         `json:"final_accuracy_pct"`
	TerrainResults map[string]int  `json:"targets_by_terrain"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  DRIFT: Multi-Terrain Neural Link Benchmark                             ║")
	fmt.Println("║  Terrains: Road, Sand, Ice, Grass | 4 Training Modes                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ========================================
	// Create DRIFT Configuration
	// ========================================
	cfg := createDriftConfig()

	// Save and reload config
	cfg.SaveToFile("drift_config.json")
	loaded, err := drift.LoadFromFile("drift_config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	fmt.Printf("✓ Loaded config: %s (%d models, %d links)\n",
		loaded.GetName(), len(loaded.Models), len(loaded.GetLinks()))

	linkConfig := loaded.GetLinks()[0]
	fmt.Printf("✓ Neural Link: %s → %s (layer %d → offset %d, size %d)\n",
		linkConfig.SourceModel, linkConfig.TargetModel,
		linkConfig.SourceLayer, linkConfig.TargetOffset, linkConfig.LinkSize)
	fmt.Println()

	// ========================================
	// Build Models
	// ========================================
	fmt.Println("═══ Building Models ═══")
	classifier, _ := nn.BuildNetworkFromJSON(string(loaded.Models["classifier"]))
	classifier.InitializeWeights()
	fmt.Println("✓ Built Classifier (4-terrain detection)")

	// Build 4 separate navigators for fair comparison
	navigators := make([]*nn.Network, 4)
	for i := 0; i < 4; i++ {
		nav, _ := nn.BuildNetworkFromJSON(string(loaded.Models["navigator"]))
		nav.InitializeWeights()
		navigators[i] = nav
	}
	fmt.Println("✓ Built 4 Navigator instances (one per mode)")
	fmt.Println()

	// ========================================
	// Phase 1: Train Classifier
	// ========================================
	fmt.Println("═══ PHASE 1: Training Classifier (all terrains) ═══")
	trainClassifier(classifier, 5*time.Second)

	// ========================================
	// Phase 2: Train Navigators on ROAD ONLY
	// ========================================
	fmt.Println()
	fmt.Println("═══ PHASE 2: Training Navigators (ROAD ONLY) ═══")
	for i, nav := range navigators {
		trainNavigatorRoadOnly(nav, linkConfig.LinkSize, 3*time.Second)
		fmt.Printf("  Navigator %d: trained (road only)\n", i+1)
	}
	fmt.Println("⚠ Navigators have NEVER seen sand/ice/grass!")
	fmt.Println()

	// ========================================
	// Phase 3: Multi-Terrain Benchmark
	// ========================================
	fmt.Println("═══ PHASE 3: Multi-Terrain Benchmark ═══")
	fmt.Println("Terrain sequence: Road → Sand → Road → Grass → Road → Ice → Road")
	fmt.Println()

	modes := []string{
		"LSTM Only (baseline)",
		"LSTM + RL",
		"LSTM + Neural Link",
		"LSTM + Neural Link + RL",
	}

	results := make([]ExperimentResult, 4)
	testDuration := 14 * time.Second // 2 seconds per terrain

	for i, mode := range modes {
		fmt.Printf("Running Mode %d: %s...\n", i+1, mode)
		useRL := (i == 1 || i == 3)
		useLink := (i == 2 || i == 3)
		results[i] = runBenchmark(classifier, navigators[i], linkConfig, mode, useLink, useRL, testDuration)
		fmt.Printf("  → %d targets, %.1f%% accuracy\n", results[i].TotalTargets, results[i].FinalAccuracy)
	}

	// ========================================
	// Print Results
	// ========================================
	fmt.Println()
	printResults(results)

	// Save to JSON
	saveResultsJSON(results)

	os.Remove("drift_config.json")
}

// ============================================================================
// DRIFT Configuration
// ============================================================================

func createDriftConfig() *drift.Config {
	cfg := drift.NewConfig("MultiTerrainNeuralLink")

	// Classifier: 8 sensors → 4 terrain classes
	classifierDef := json.RawMessage(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"layers": [
			{"type": "dense", "input_size": 8, "output_size": 32, "activation": "leaky_relu"},
			{"type": "dense", "input_size": 32, "output_size": 16, "activation": "leaky_relu"},
			{"type": "dense", "input_size": 16, "output_size": 4, "activation": "sigmoid"}
		]
	}`)

	// Navigator: 4 (pos/dir) + 16 (link) = 20 inputs → 4 actions
	navigatorDef := json.RawMessage(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 4,
		"layers": [
			{"type": "dense", "input_size": 20, "output_size": 32, "activation": "leaky_relu"},
			{"type": "lstm", "input_size": 8, "hidden_size": 8, "seq_length": 4},
			{"type": "dense", "input_size": 32, "output_size": 16, "activation": "leaky_relu"},
			{"type": "dense", "input_size": 16, "output_size": 4, "activation": "sigmoid"}
		]
	}`)

	cfg.Models["classifier"] = classifierDef
	cfg.Models["navigator"] = navigatorDef

	// Neural link configuration
	cfg.AddLink(drift.NeuralLinkConfig{
		Name:         "terrain_to_nav",
		SourceModel:  "classifier",
		SourceLayer:  1,
		TargetModel:  "navigator",
		TargetOffset: 4,
		LinkSize:     16,
		Enabled:      true,
		Description:  "Classifier hidden → Navigator input for terrain awareness",
	})

	return cfg
}

// ============================================================================
// Training
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
		tween.TweenStep(net, navInput, optimal, NumActions, lr)

		executeAction(env, predicted)
		if rand.Float32() < 0.1 {
			env.TargetPos = [2]float32{rand.Float32(), rand.Float32()}
		}
	}
}

// ============================================================================
// Benchmark
// ============================================================================

func runBenchmark(classifier, navigator *nn.Network, linkConfig drift.NeuralLinkConfig,
	modeName string, useLink, useRL bool, duration time.Duration) ExperimentResult {

	inputSize := 4 + linkConfig.LinkSize
	classifierState := classifier.InitStepState(8)
	navigatorState := navigator.InitStepState(inputSize)

	var tween *nn.TweenState
	if useRL {
		tween = nn.NewTweenState(navigator, nil)
		tween.Config.UseChainRule = true
	}

	env := &Environment{
		AgentPos:   [2]float32{0.2, 0.2},
		TargetPos:  [2]float32{0.8, 0.8},
		Terrain:    TerrainRoad,
		LastAction: -1,
	}

	windowDuration := 500 * time.Millisecond
	terrainDuration := duration / time.Duration(len(terrainSequence))

	result := ExperimentResult{
		Mode:           modeName,
		Windows:        []WindowMetrics{},
		TerrainResults: make(map[string]int),
	}

	start := time.Now()
	lastWindow := start
	windowNum := 0
	currentTerrainIdx := 0
	lastTerrainChange := start

	windowTargets := 0
	windowSteps := 0
	windowEffective := 0

	lr := float32(0.01)

	for time.Since(start) < duration {
		// Update terrain based on sequence
		if time.Since(lastTerrainChange) >= terrainDuration && currentTerrainIdx < len(terrainSequence)-1 {
			currentTerrainIdx++
			env.Terrain = terrainSequence[currentTerrainIdx]
			lastTerrainChange = time.Now()
			// Reset ice momentum on terrain change
			env.IceVelX = 0
			env.IceVelY = 0
		}

		// Get neural link data if enabled
		var linkData []float32
		if useLink {
			sensors := generateSensorData(env.Terrain)
			classifierState.SetInput(sensors)
			classifier.StepForward(classifierState)
			linkData = getHiddenActivations(classifierState, linkConfig.SourceLayer, linkConfig.LinkSize)
		}

		// Navigator forward pass
		navInput := buildNavigatorInput(env, linkData, linkConfig.LinkSize)
		navigatorState.SetInput(navInput)
		navigator.StepForward(navigatorState)
		output := navigatorState.GetOutput()

		// Execute action
		action := argmax(output)
		prevDist := distanceToTarget(env)
		executeActionWithPhysics(env, action)
		newDist := distanceToTarget(env)
		windowSteps++
		result.TotalSteps++

		gotCloser := newDist < prevDist-0.001
		if gotCloser {
			windowEffective++
		}

		// RL update if enabled
		if useRL && tween != nil {
			if gotCloser {
				tween.TweenStep(navigator, navInput, action, NumActions, lr)
			} else {
				altAction := suggestAlternativeAction(env, action)
				tween.TweenStep(navigator, navInput, altAction, NumActions, lr)
			}
		}

		// Check if reached target
		if newDist < 0.1 {
			windowTargets++
			result.TotalTargets++
			result.TerrainResults[terrainNames[env.Terrain]]++

			// Reset to new position
			env.AgentPos = [2]float32{rand.Float32() * 0.3, rand.Float32() * 0.3}
			env.TargetPos = [2]float32{0.7 + rand.Float32()*0.3, 0.7 + rand.Float32()*0.3}
			env.LastAction = -1
			env.StuckCount = 0
			env.IceVelX = 0
			env.IceVelY = 0
		}

		// Record window metrics every 500ms
		if time.Since(lastWindow) >= windowDuration {
			accuracy := 0.0
			if windowSteps > 0 {
				accuracy = float64(windowEffective) / float64(windowSteps) * 100
			}

			result.Windows = append(result.Windows, WindowMetrics{
				WindowNum:      windowNum,
				Terrain:        terrainNames[env.Terrain],
				TargetsReached: windowTargets,
				TotalSteps:     windowSteps,
				EffectiveMoves: windowEffective,
				Accuracy:       accuracy,
			})

			windowNum++
			windowTargets = 0
			windowSteps = 0
			windowEffective = 0
			lastWindow = time.Now()
		}
	}

	// Final accuracy
	totalEffective := 0
	totalSteps := 0
	for _, w := range result.Windows {
		totalEffective += w.EffectiveMoves
		totalSteps += w.TotalSteps
	}
	if totalSteps > 0 {
		result.FinalAccuracy = float64(totalEffective) / float64(totalSteps) * 100
	}

	return result
}

// ============================================================================
// Sensor & Input Generation
// ============================================================================

func generateSensorData(terrain int) []float32 {
	// [friction, softness, slipperiness, roughness, moisture, density, temperature, stability]
	var base []float32
	switch terrain {
	case TerrainRoad:
		base = []float32{0.9, 0.1, 0.1, 0.3, 0.2, 0.9, 0.5, 0.95}
	case TerrainSand:
		base = []float32{0.4, 0.85, 0.2, 0.8, 0.1, 0.3, 0.7, 0.4}
	case TerrainIce:
		base = []float32{0.05, 0.05, 0.95, 0.1, 0.1, 0.9, 0.1, 0.6}
	case TerrainGrass:
		base = []float32{0.6, 0.4, 0.3, 0.5, 0.6, 0.5, 0.5, 0.7}
	default:
		base = []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	}

	sensors := make([]float32, 8)
	for i := range base {
		sensors[i] = base[i] + (rand.Float32()-0.5)*0.15
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

// ============================================================================
// Actions & Physics
// ============================================================================

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

func suggestAlternativeAction(env *Environment, currentAction int) int {
	// Suggest zigzag pattern
	if env.LastAction == ActionUp || env.LastAction == ActionDown {
		if env.TargetPos[0] > env.AgentPos[0] {
			return ActionRight
		}
		return ActionLeft
	}
	if env.TargetPos[1] > env.AgentPos[1] {
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
	moves := [][2]float32{{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}}

	var moveX, moveY float32
	if action >= 0 && action < 4 {
		moveX = moves[action][0]
		moveY = moves[action][1]
	}

	switch env.Terrain {
	case TerrainRoad:
		// Normal movement
		env.AgentPos[0] = clamp(env.AgentPos[0]+moveX, 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moveY, 0, 1)

	case TerrainSand:
		// Straight moves get stuck, zigzag works
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
		env.AgentPos[0] = clamp(env.AgentPos[0]+moveX*(speed/0.02), 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moveY*(speed/0.02), 0, 1)

	case TerrainIce:
		// Momentum-based: slow to change direction
		friction := float32(0.1)
		env.IceVelX = env.IceVelX*(1-friction) + moveX*friction
		env.IceVelY = env.IceVelY*(1-friction) + moveY*friction
		env.AgentPos[0] = clamp(env.AgentPos[0]+env.IceVelX, 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+env.IceVelY, 0, 1)

	case TerrainGrass:
		// Slightly slower, but consistent
		speed *= 0.8
		env.AgentPos[0] = clamp(env.AgentPos[0]+moveX*(speed/0.02), 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moveY*(speed/0.02), 0, 1)
	}

	env.LastAction = action
}

func distanceToTarget(env *Environment) float32 {
	dx := env.TargetPos[0] - env.AgentPos[0]
	dy := env.TargetPos[1] - env.AgentPos[1]
	return float32(math.Sqrt(float64(dx*dx + dy*dy)))
}

// ============================================================================
// Results Display
// ============================================================================

func printResults(results []ExperimentResult) {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                               MULTI-TERRAIN BENCHMARK RESULTS                                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Mode                          │ Targets │ Accuracy │ Road │ Sand │ Ice  │ Grass                 ║")
	fmt.Println("╠───────────────────────────────┼─────────┼──────────┼──────┼──────┼──────┼───────────────────────╣")

	for _, r := range results {
		road := r.TerrainResults["Road"]
		sand := r.TerrainResults["Sand"]
		ice := r.TerrainResults["Ice"]
		grass := r.TerrainResults["Grass"]
		fmt.Printf("║ %-29s │ %7d │ %7.1f%% │ %4d │ %4d │ %4d │ %4d                  ║\n",
			r.Mode, r.TotalTargets, r.FinalAccuracy, road, sand, ice, grass)
	}

	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Print timeline for best mode
	best := results[0]
	for _, r := range results {
		if r.TotalTargets > best.TotalTargets {
			best = r
		}
	}

	fmt.Println()
	fmt.Printf("Timeline for best mode (%s):\n", best.Mode)
	fmt.Println("┌────────┬─────────┬─────────┬──────────┬────────────┐")
	fmt.Println("│ Window │ Terrain │ Targets │ Accuracy │ Steps      │")
	fmt.Println("├────────┼─────────┼─────────┼──────────┼────────────┤")

	for i, w := range best.Windows {
		if i < 20 { // Show first 20 windows
			fmt.Printf("│ %6d │ %-7s │ %7d │ %7.1f%% │ %10d │\n",
				w.WindowNum, w.Terrain, w.TargetsReached, w.Accuracy, w.TotalSteps)
		}
	}
	fmt.Println("└────────┴─────────┴─────────┴──────────┴────────────┘")
}

func saveResultsJSON(results []ExperimentResult) {
	data := map[string]interface{}{
		"experiment":       "multi_terrain_neural_link",
		"terrain_sequence": []string{"Road", "Sand", "Road", "Grass", "Road", "Ice", "Road"},
		"timestamp":        time.Now().Format(time.RFC3339),
		"results":          results,
	}

	jsonData, _ := json.MarshalIndent(data, "", "  ")
	os.WriteFile("benchmark_results.json", jsonData, 0644)
	fmt.Println("\n✓ Results saved to benchmark_results.json")
}

// ============================================================================
// Utilities
// ============================================================================

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

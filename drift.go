// Package drift provides Decentralized Real-time Integration & Functional Transfer.
//
// DRIFT is an experimental framework exploring the boundaries of Decentralized
// Functional Integration, enabling heterogeneous neural agents to autonomously
// develop communication protocols in real-time.
package drift

import (
	"encoding/json"
	"os"
)

// NeuralLinkConfig defines how to connect two models.
// Source model's layer output is injected into target model's input at specified offset.
type NeuralLinkConfig struct {
	Name         string `json:"name"`          // Unique identifier for this link
	SourceModel  string `json:"source_model"`  // Name of the source model
	SourceLayer  int    `json:"source_layer"`  // Layer index to extract activations from
	TargetModel  string `json:"target_model"`  // Name of the target model
	TargetOffset int    `json:"target_offset"` // Input offset where link data is injected
	LinkSize     int    `json:"link_size"`     // Number of neurons to transfer
	Enabled      bool   `json:"enabled"`       // Whether this link is active
	Description  string `json:"description"`   // Human-readable description
}

// Config holds the configuration for a DRIFT instance.
type Config struct {
	Name   string                     `json:"name"`
	Models map[string]json.RawMessage `json:"models"`
	Links  []NeuralLinkConfig         `json:"links,omitempty"`
}

// NewConfig creates a new Config with the given name.
func NewConfig(name string) *Config {
	return &Config{
		Name:   name,
		Models: make(map[string]json.RawMessage),
		Links:  []NeuralLinkConfig{},
	}
}

// GetName returns the name of the config.
func (c *Config) GetName() string {
	return c.Name
}

// AddModel adds a model (any struct) to the config with a given name.
func (c *Config) AddModel(name string, model interface{}) error {
	data, err := json.Marshal(model)
	if err != nil {
		return err
	}
	c.Models[name] = data
	return nil
}

// GetModel retrieves a model by name and unmarshals it into the provided target.
func (c *Config) GetModel(name string, target interface{}) error {
	data, ok := c.Models[name]
	if !ok {
		return os.ErrNotExist
	}
	return json.Unmarshal(data, target)
}

// AddLink adds a neural link configuration.
func (c *Config) AddLink(link NeuralLinkConfig) {
	c.Links = append(c.Links, link)
}

// GetLinks returns all neural link configurations.
func (c *Config) GetLinks() []NeuralLinkConfig {
	return c.Links
}

// GetLinksBySource returns all links originating from the specified model.
func (c *Config) GetLinksBySource(modelName string) []NeuralLinkConfig {
	var result []NeuralLinkConfig
	for _, link := range c.Links {
		if link.SourceModel == modelName && link.Enabled {
			result = append(result, link)
		}
	}
	return result
}

// GetLinksByTarget returns all links targeting the specified model.
func (c *Config) GetLinksByTarget(modelName string) []NeuralLinkConfig {
	var result []NeuralLinkConfig
	for _, link := range c.Links {
		if link.TargetModel == modelName && link.Enabled {
			result = append(result, link)
		}
	}
	return result
}

// ToJSON serializes the config to a JSON string.
func (c *Config) ToJSON() (string, error) {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// FromJSON deserializes a JSON string into a Config.
func FromJSON(data string) (*Config, error) {
	var c Config
	err := json.Unmarshal([]byte(data), &c)
	if err != nil {
		return nil, err
	}
	return &c, nil
}

// SaveToFile saves the config to a JSON file.
func (c *Config) SaveToFile(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadFromFile loads a config from a JSON file.
func LoadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Config
	err = json.Unmarshal(data, &c)
	if err != nil {
		return nil, err
	}
	return &c, nil
}

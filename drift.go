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

// Config holds the configuration for a DRIFT instance.
type Config struct {
	Name   string                     `json:"name"`
	Models map[string]json.RawMessage `json:"models"`
}

// NewConfig creates a new Config with the given name.
func NewConfig(name string) *Config {
	return &Config{
		Name:   name,
		Models: make(map[string]json.RawMessage),
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

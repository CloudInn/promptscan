package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/CloudInn/promptscan"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/spf13/cobra"
)

const (
	EmbeddingsVersion = "v1.0.0" // Change this to trigger re-generation
	EmbeddingModel    = "text-embedding-3-small"
	OutputDir         = "generated/embeddings"
)

type EmbeddingData struct {
	Version    string               `json:"version"`
	Model      string               `json:"model"`
	Keywords   []string             `json:"keywords"`
	Embeddings map[string][]float64 `json:"embeddings"`
	Metadata   EmbeddingMetadata    `json:"metadata"`
}

type EmbeddingMetadata struct {
	GeneratedAt   string `json:"generated_at"`
	TotalKeywords int    `json:"total_keywords"`
	EmbeddingDim  int    `json:"embedding_dimension"`
}

var rootCmd = &cobra.Command{
	Use:   "embeddings-cli",
	Short: "Generate embeddings for prompt injection detection keywords",
	Long: `This CLI tool generates embeddings for injection detection keywords using OpenAI's embedding API.
The embeddings are stored in JSON files and used for in-memory vector similarity search.`,
}

var generateCmd = &cobra.Command{
	Use:   "generate",
	Short: "Generate embeddings for injection keywords",
	Long: `Generate embeddings for all injection keywords using OpenAI's embedding API.
The embeddings will be stored in the generated/embeddings directory.`,
	Run: func(cmd *cobra.Command, args []string) {
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			log.Fatal("OPENAI_API_KEY environment variable is required")
		}

		batchSize, _ := cmd.Flags().GetInt("batch-size")
		outputPath, _ := cmd.Flags().GetString("output")

		if err := generateEmbeddings(apiKey, batchSize, outputPath); err != nil {
			log.Fatalf("Failed to generate embeddings: %v", err)
		}

		fmt.Println("Embeddings generated successfully!")
	},
}

var checkCmd = &cobra.Command{
	Use:   "check",
	Short: "Check if embeddings need to be regenerated",
	Long: `Check if the current embeddings version matches the code version.
Returns exit code 0 if embeddings are up to date, 1 if they need regeneration.`,
	Run: func(cmd *cobra.Command, args []string) {
		outputPath, _ := cmd.Flags().GetString("output")
		needsUpdate, err := checkEmbeddingsVersion(outputPath)
		if err != nil {
			log.Fatalf("Failed to check embeddings version: %v", err)
		}

		if needsUpdate {
			fmt.Printf("Embeddings need to be regenerated (current version: %s)\n", EmbeddingsVersion)
			os.Exit(1)
		} else {
			fmt.Println("Embeddings are up to date")
			os.Exit(0)
		}
	},
}

func init() {
	generateCmd.Flags().IntP("batch-size", "b", 100, "Number of keywords to process in each batch")
	generateCmd.Flags().StringP("output", "o", OutputDir, "Output directory for embeddings")

	checkCmd.Flags().StringP("output", "o", OutputDir, "Output directory for embeddings")

	rootCmd.AddCommand(generateCmd)
	rootCmd.AddCommand(checkCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
	}
}

func generateEmbeddings(apiKey string, batchSize int, outputPath string) error {
	// Create output directory
	if err := os.MkdirAll(outputPath, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Create OpenAI client
	client := openai.NewClient(option.WithAPIKey(apiKey))

	// Generate injection keywords using the same logic as the main package
	detector, err := promptscan.NewWithDefaults(apiKey)
	if err != nil {
		return fmt.Errorf("failed to create detector: %w", err)
	}

	keywords := detector.GetInjectionKeywords()
	fmt.Printf("Generating embeddings for %d keywords...\n", len(keywords))

	embeddings := make(map[string][]float64)

	// Process keywords in batches to avoid rate limits
	for i := 0; i < len(keywords); i += batchSize {
		end := i + batchSize
		if end > len(keywords) {
			end = len(keywords)
		}

		batch := keywords[i:end]
		fmt.Printf("Processing batch %d/%d (%d keywords)...\n",
			(i/batchSize)+1,
			(len(keywords)+batchSize-1)/batchSize,
			len(batch))

		batchEmbeddings, err := generateBatchEmbeddings(client, batch)
		if err != nil {
			return fmt.Errorf("failed to generate batch embeddings: %w", err)
		}

		// Merge batch results
		for keyword, embedding := range batchEmbeddings {
			embeddings[keyword] = embedding
		}
	}

	// Create embedding data structure
	embeddingData := EmbeddingData{
		Metadata: EmbeddingMetadata{
			GeneratedAt:   fmt.Sprintf("%d", os.Getpid()), // Simple timestamp
			TotalKeywords: len(keywords),
			EmbeddingDim:  len(embeddings[keywords[0]]), // Assume all embeddings have same dimension
		},
	}

	// Save to JSON files (split into chunks for better performance)
	const chunkSize = 1000
	totalChunks := (len(keywords) + chunkSize - 1) / chunkSize

	for i := 0; i < totalChunks; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(keywords) {
			end = len(keywords)
		}

		chunkKeywords := keywords[start:end]
		chunkEmbeddings := make(map[string][]float64)

		for _, keyword := range chunkKeywords {
			chunkEmbeddings[keyword] = embeddings[keyword]
		}

		chunkData := EmbeddingData{
			Version:    EmbeddingsVersion,
			Model:      EmbeddingModel,
			Keywords:   chunkKeywords,
			Embeddings: chunkEmbeddings,
			Metadata:   embeddingData.Metadata,
		}

		filename := filepath.Join(outputPath, fmt.Sprintf("embeddings_chunk_%d.json", i))
		if err := saveEmbeddingsToFile(chunkData, filename); err != nil {
			return fmt.Errorf("failed to save chunk %d: %w", i, err)
		}
	}

	// Save metadata file
	metadataFile := filepath.Join(outputPath, "metadata.json")
	metadataData := struct {
		Version     string            `json:"version"`
		Model       string            `json:"model"`
		TotalChunks int               `json:"total_chunks"`
		Metadata    EmbeddingMetadata `json:"metadata"`
	}{
		Version:     EmbeddingsVersion,
		Model:       EmbeddingModel,
		TotalChunks: totalChunks,
		Metadata:    embeddingData.Metadata,
	}

	metadataJson, err := json.MarshalIndent(metadataData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	if err := os.WriteFile(metadataFile, metadataJson, 0644); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	fmt.Printf("Successfully generated %d embeddings in %d chunks\n", len(keywords), totalChunks)
	return nil
}

func generateBatchEmbeddings(client openai.Client, keywords []string) (map[string][]float64, error) {
	// Convert keywords to input slice
	inputs := make([]string, len(keywords))
	copy(inputs, keywords)

	// Create embedding request
	response, err := client.Embeddings.New(context.Background(), openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: inputs,
		},
		Model: openai.EmbeddingModelTextEmbedding3Small,
	})

	if err != nil {
		return nil, fmt.Errorf("OpenAI API call failed: %w", err)
	}

	if len(response.Data) != len(keywords) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(keywords), len(response.Data))
	}

	// Map keywords to their embeddings
	embeddings := make(map[string][]float64)
	for i, embedding := range response.Data {
		// Convert float32 to float64
		embeddingFloat64 := make([]float64, len(embedding.Embedding))
		for j, val := range embedding.Embedding {
			embeddingFloat64[j] = float64(val)
		}
		embeddings[keywords[i]] = embeddingFloat64
	}

	return embeddings, nil
}

func saveEmbeddingsToFile(data EmbeddingData, filename string) error {
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

func checkEmbeddingsVersion(outputPath string) (bool, error) {
	metadataFile := filepath.Join(outputPath, "metadata.json")

	// Check if metadata file exists
	if _, err := os.Stat(metadataFile); os.IsNotExist(err) {
		return true, nil // Need to generate
	}

	// Read metadata
	data, err := os.ReadFile(metadataFile)
	if err != nil {
		return true, fmt.Errorf("failed to read metadata file: %w", err)
	}

	var metadata struct {
		Version string `json:"version"`
	}

	if err := json.Unmarshal(data, &metadata); err != nil {
		return true, fmt.Errorf("failed to parse metadata: %w", err)
	}

	// Compare versions
	return metadata.Version != EmbeddingsVersion, nil
}

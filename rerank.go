// Copyright 2025 Xavier Portilla Edo
// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// RerankOptions configures a [Rerank] call. Pass it via
// [ai.RerankerRequest.Options].
type RerankOptions struct {
	// TopN caps the number of ranked documents returned. If <= 0, every input
	// document is returned, ordered by descending relevance.
	TopN int `json:"topN,omitempty"`
}

// cohereRerankRequest is the Cohere Rerank InvokeModel body. api_version 2 is
// the current Bedrock contract.
type cohereRerankRequest struct {
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	TopN       int      `json:"top_n"`
	APIVersion int      `json:"api_version"`
}

type cohereRerankResponse struct {
	Results []cohereRerankResult `json:"results"`
}

type cohereRerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

const cohereRerankAPIVersion = 2

// Rerank reranks req.Documents by relevance to req.Query using a Bedrock
// reranking model, such as "cohere.rerank-v3-5:0". It returns documents in
// descending score order and attaches each score to [ai.RankedDocumentMetadata].
//
// Genkit Go does not yet expose a first-class reranker primitive, so this is a
// standalone helper. It looks up the initialized Bedrock plugin on g and reuses
// its configured Bedrock Runtime client.
func Rerank(ctx context.Context, g *genkit.Genkit, modelID string, req *ai.RerankerRequest) (*ai.RerankerResponse, error) {
	if g == nil {
		return nil, errors.New("bedrock.Rerank: Genkit instance required")
	}
	p, _ := genkit.LookupPlugin(g, provider).(*Bedrock)
	if p == nil {
		return nil, errors.New("bedrock.Rerank: bedrock plugin not registered")
	}

	p.mu.Lock()
	initted := p.initted
	client := p.client
	p.mu.Unlock()

	if !initted {
		return nil, errors.New("bedrock.Rerank: plugin not initialized")
	}
	if modelID == "" {
		return nil, errors.New("bedrock.Rerank: model ID required")
	}
	if req == nil {
		return nil, errors.New("bedrock.Rerank: request required")
	}

	return rerank(ctx, client, modelID, req)
}

func rerank(ctx context.Context, client BedrockClient, modelID string, req *ai.RerankerRequest) (*ai.RerankerResponse, error) {
	if client == nil {
		return nil, errors.New("bedrock.Rerank: Bedrock client required")
	}
	if modelID == "" {
		return nil, errors.New("bedrock.Rerank: model ID required")
	}
	if req == nil {
		return nil, errors.New("bedrock.Rerank: request required")
	}

	query := documentText(req.Query)
	if query == "" {
		return nil, errors.New("bedrock.Rerank: query has no text content")
	}

	docs := make([]string, 0, len(req.Documents))
	for i, doc := range req.Documents {
		text := documentText(doc)
		if text == "" {
			return nil, fmt.Errorf("bedrock.Rerank: document %d has no text content", i)
		}
		docs = append(docs, text)
	}
	if len(docs) == 0 {
		return &ai.RerankerResponse{}, nil
	}

	topN := len(docs)
	opts, err := rerankOptions(req.Options)
	if err != nil {
		return nil, err
	}
	if opts != nil && opts.TopN > 0 && opts.TopN < topN {
		topN = opts.TopN
	}

	var resp cohereRerankResponse
	if err := invokeRerankJSON(ctx, client, modelID, cohereRerankRequest{
		Query:      query,
		Documents:  docs,
		TopN:       topN,
		APIVersion: cohereRerankAPIVersion,
	}, &resp); err != nil {
		return nil, err
	}

	return buildRerankResponse(resp, req.Documents)
}

func invokeRerankJSON(ctx context.Context, client BedrockClient, modelID string, req any, resp any) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("bedrock.Rerank: failed to marshal request: %w", err)
	}

	out, err := client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return fmt.Errorf("bedrock.Rerank: failed to invoke model: %w", err)
	}
	if out == nil {
		return errors.New("bedrock.Rerank: empty invoke model response")
	}
	if err := json.Unmarshal(out.Body, resp); err != nil {
		return fmt.Errorf("bedrock.Rerank: failed to unmarshal response: %w", err)
	}
	return nil
}

// buildRerankResponse maps Cohere's score results back onto the original
// documents, preserving reranked order and attaching each relevance score.
func buildRerankResponse(resp cohereRerankResponse, docs []*ai.Document) (*ai.RerankerResponse, error) {
	out := &ai.RerankerResponse{
		Documents: make([]*ai.RankedDocumentData, 0, len(resp.Results)),
	}
	for _, result := range resp.Results {
		if result.Index < 0 || result.Index >= len(docs) {
			return nil, fmt.Errorf("bedrock.Rerank: result index %d out of range for %d documents", result.Index, len(docs))
		}
		if docs[result.Index] == nil {
			return nil, fmt.Errorf("bedrock.Rerank: result index %d references nil document", result.Index)
		}
		out.Documents = append(out.Documents, &ai.RankedDocumentData{
			Content: docs[result.Index].Content,
			Metadata: &ai.RankedDocumentMetadata{
				Score: result.RelevanceScore,
			},
		})
	}
	return out, nil
}

func documentText(doc *ai.Document) string {
	if doc == nil {
		return ""
	}

	var texts []string
	for _, part := range doc.Content {
		if part != nil && part.IsText() && strings.TrimSpace(part.Text) != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.TrimSpace(strings.Join(texts, "\n"))
}

// rerankOptions extracts [RerankOptions] from the request's Options field,
// accepting either a value, pointer, or JSON-deserialized map. It returns nil
// options when options are absent.
func rerankOptions(o any) (*RerankOptions, error) {
	switch v := o.(type) {
	case nil:
		return nil, nil
	case *RerankOptions:
		return v, nil
	case RerankOptions:
		return &v, nil
	case map[string]any:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("bedrock.Rerank: failed to marshal rerank options: %w", err)
		}
		var opts RerankOptions
		if err := json.Unmarshal(b, &opts); err != nil {
			return nil, fmt.Errorf("bedrock.Rerank: failed to unmarshal rerank options: %w", err)
		}
		return &opts, nil
	default:
		return nil, fmt.Errorf("bedrock.Rerank: unsupported rerank options type %T", o)
	}
}

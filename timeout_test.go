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
	"testing"
	"time"
)

func TestWithRequestTimeoutAppliesDeadline(t *testing.T) {
	ctx, cancel := withRequestTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("withRequestTimeout did not set deadline")
	}
	if time.Until(deadline) <= 0 {
		t.Fatalf("deadline = %v, want future deadline", deadline)
	}
}

func TestWithRequestTimeoutDisabled(t *testing.T) {
	parent := context.Background()
	ctx, cancel := withRequestTimeout(parent, 0)
	defer cancel()

	if ctx != parent {
		t.Fatal("withRequestTimeout should return original context when timeout is disabled")
	}
	if _, ok := ctx.Deadline(); ok {
		t.Fatal("withRequestTimeout set deadline when timeout is disabled")
	}
}

func TestBedrockWithRequestTimeoutNilReceiver(t *testing.T) {
	var b *Bedrock
	parent := context.Background()
	ctx, cancel := b.withRequestTimeout(parent)
	defer cancel()

	if ctx != parent {
		t.Fatal("nil Bedrock receiver should return original context")
	}
}

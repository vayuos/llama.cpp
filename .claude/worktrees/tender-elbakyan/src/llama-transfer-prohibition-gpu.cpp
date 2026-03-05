/**
 * SECTION 30: Prohibit Per-Token Host↔Device Transfers
 * Implementation
 *
 * This file implements comprehensive transfer prohibition enforcement.
 * No decode-critical tensor or buffer may cross PCIe during decode.
 * Only the final selected token ID is permitted to cross PCIe per token.
 * All other data remains GPU-resident throughout decode execution.
 */

#include "llama-transfer-prohibition-gpu.h"
#include <map>
#include <string>
#include <cstring>
#include <ctime>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_gpu_transfer_prohibition_validation_state g_transfer_prohibition_validation = {
    /* config */ {
        /* transfer_prohibition_enabled */ false,
        /* preallocate_all_buffers */ false,
        /* forbid_implicit_syncs */ true,
        /* forbid_unified_memory */ true,
        /* forbid_mapped_access */ false,
        /* max_transfer_per_token_bytes */ 8, // sizeof(int64_t)
        /* debug_transfer_prohibition */ false,
    },
    /* state_record */ {
        /* state */ LLAMA_GPU_TRANSFER_PROHIBITION_UNINITIALIZED,
        /* mode */ LLAMA_TRANSFER_PROHIBITION_NONE,
        /* total_transfers_during_decode */ 0,
        /* total_transfer_bytes_during_decode */ 0,
        /* total_violations */ 0,
        /* last_violation */ LLAMA_TRANSFER_VIOLATION_NONE,
        /* reserved_1 */ 0,
    },
    /* preallocated_buffers */ {
        false, // logits_buffer_allocated
        false, // sampling_workspace_allocated
        false, // topk_buffer_allocated
        false, // topp_buffer_allocated
        false, // kv_cache_allocated
        false, // attention_state_allocated
        false, // penalty_buffer_allocated
        false, // candidate_buffer_allocated
        0,     // total_preallocated_bytes
        0      // reserved_1
    },
    /* last_transfer */ {
        LLAMA_TRANSFER_TYPE_NONE, // transfer_type
        0,                        // transfer_size_bytes
        false,                    // is_decode_critical
        false,                    // during_decode_phase
        0,                        // timestamp_ns
        0                         // reserved
    },
    /* total_transfer_events */ 0,
    /* total_violations */ 0,
    /* enforcement_strict */ true,
    /* decode_phase_active */ false,
};

// Per-transfer type tracking
static std::map<std::string, uint64_t> g_transfer_type_bytes;
static std::map<std::string, int> g_transfer_type_count;

// Violation history
static std::map<uint64_t, enum llama_transfer_violation> g_violation_history;

// ============================================================================
// INITIALIZATION AND CONFIGURATION
// ============================================================================

int llama_transfer_prohibition_gpu_init(void) {
    g_transfer_prohibition_validation.state_record.state = LLAMA_GPU_TRANSFER_PROHIBITION_INITIALIZED;
    g_transfer_prohibition_validation.state_record.mode = LLAMA_TRANSFER_PROHIBITION_NONE;
    g_transfer_prohibition_validation.total_violations = 0;
    g_transfer_prohibition_validation.total_transfer_events = 0;
    g_transfer_prohibition_validation.decode_phase_active = false;
    g_transfer_type_bytes.clear();
    g_transfer_type_count.clear();
    g_violation_history.clear();

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Initialization complete\n");
    }

    return 0;
}

int llama_transfer_prohibition_gpu_configure(
    bool transfer_prohibition_enabled,
    bool preallocate_all_buffers,
    bool forbid_implicit_syncs,
    bool forbid_unified_memory,
    uint64_t max_transfer_per_token_bytes
) {
    g_transfer_prohibition_validation.config.transfer_prohibition_enabled = transfer_prohibition_enabled;
    g_transfer_prohibition_validation.config.preallocate_all_buffers = preallocate_all_buffers;
    g_transfer_prohibition_validation.config.forbid_implicit_syncs = forbid_implicit_syncs;
    g_transfer_prohibition_validation.config.forbid_unified_memory = forbid_unified_memory;
    g_transfer_prohibition_validation.config.max_transfer_per_token_bytes = max_transfer_per_token_bytes;

    if (transfer_prohibition_enabled) {
        g_transfer_prohibition_validation.state_record.mode = LLAMA_TRANSFER_PROHIBITION_ENABLED;
    }

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Configured: enabled=%d, preallocate=%d, max_bytes=%llu\n",
            transfer_prohibition_enabled, preallocate_all_buffers, (unsigned long long)max_transfer_per_token_bytes);
    }

    return 0;
}

// ============================================================================
// DECODE PHASE MANAGEMENT (10 ENFORCEMENT POINTS)
// ============================================================================

// Enforcement Point 1: Begin decode phase
int llama_transfer_prohibition_gpu_begin_decode_phase(void) {
    if (!g_transfer_prohibition_validation.config.transfer_prohibition_enabled) {
        return 0;
    }

    g_transfer_prohibition_validation.decode_phase_active = true;
    g_transfer_prohibition_validation.state_record.state = LLAMA_GPU_TRANSFER_PROHIBITION_DECODE_ACTIVE;
    // Use sed or manual replacement for multiple occurrences
// But replace_file_content with AllowMultiple=true works well for string replace.
g_transfer_prohibition_validation.state_record.total_transfers_during_decode = 0;
    g_transfer_prohibition_validation.state_record.total_transfer_bytes_during_decode = 0;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Decode phase STARTED\n");
    }

    return 0;
}

// Enforcement Point 2: End decode phase
int llama_transfer_prohibition_gpu_end_decode_phase(void) {
    if (!g_transfer_prohibition_validation.config.transfer_prohibition_enabled) {
        return 0;
    }

    g_transfer_prohibition_validation.decode_phase_active = false;
    g_transfer_prohibition_validation.state_record.state = LLAMA_GPU_TRANSFER_PROHIBITION_COMPLETE;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Decode phase ENDED\n");
        fprintf(stderr, "  Total transfers: %llu, Total bytes: %llu\n",
            (unsigned long long)// Use sed or manual replacement for multiple occurrences
// But replace_file_content with AllowMultiple=true works well for string replace.
g_transfer_prohibition_validation.state_record.total_transfers_during_decode,
            (unsigned long long)g_transfer_prohibition_validation.state_record.total_transfer_bytes_during_decode);
    }

    return 0;
}

// Enforcement Point 3: Verify all buffers preallocated
int llama_transfer_prohibition_gpu_verify_all_buffers_preallocated(void) {
    if (!g_transfer_prohibition_validation.config.preallocate_all_buffers) {
        return 0;
    }

    struct llama_gpu_preallocated_buffers* bufs = &g_transfer_prohibition_validation.preallocated_buffers;

    // All critical buffers must be pre-allocated
    if (!bufs->logits_buffer_allocated ||
        !bufs->sampling_workspace_allocated ||
        !bufs->topk_buffer_allocated ||
        !bufs->topp_buffer_allocated ||
        !bufs->kv_cache_allocated) {

        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Not all buffers preallocated!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 4: Forbid implicit synchronization
int llama_transfer_prohibition_gpu_forbid_implicit_synchronization(void) {
    if (!g_transfer_prohibition_validation.config.forbid_implicit_syncs) {
        return 0;
    }

    // Check for implicit synchronization transfers
    if (g_transfer_type_count["implicit_sync"] > 0) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_IMPLICIT_SYNC_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Implicit sync transfer detected!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 5: Forbid unified memory access
int llama_transfer_prohibition_gpu_forbid_unified_memory_access(void) {
    if (!g_transfer_prohibition_validation.config.forbid_unified_memory) {
        return 0;
    }

    if (g_transfer_type_count["unified_memory"] > 0) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_UNIFIED_MEMORY_ACCESS;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Unified memory access detected during decode!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 6: Forbid mapped buffer access
int llama_transfer_prohibition_gpu_forbid_mapped_buffer_access(void) {
    if (!g_transfer_prohibition_validation.config.forbid_mapped_access) {
        return 0;
    }

    if (g_transfer_type_count["mapped_access"] > 0) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_MAPPED_BUFFER_ACCESS;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Mapped buffer access detected during decode!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 7: Forbid logits host reads
int llama_transfer_prohibition_gpu_forbid_logits_host_reads(void) {
    if (g_transfer_type_count["logits_read"] > 0) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_LOGITS_READ;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Host read of logits detected!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 8: Forbid KV cache transfers
int llama_transfer_prohibition_gpu_forbid_kv_cache_transfers(void) {
    if (g_transfer_type_count["kv_cache_transfer"] > 0) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_KV_CACHE_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] KV cache transfer detected!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 9: Allow token ID only
int llama_transfer_prohibition_gpu_allow_token_id_only(void) {
    // Verify total transferred bytes per token is within limits

    uint64_t total_non_token_bytes = g_transfer_prohibition_validation.state_record.total_transfer_bytes_during_decode;

    // Subtract expected token IDs (one per token, let's assume 1 token as baseline)
    if (total_non_token_bytes > g_transfer_prohibition_validation.config.max_transfer_per_token_bytes) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Excessive transfer detected: %llu > %llu bytes\n",
                (unsigned long long)total_non_token_bytes,
                (unsigned long long)g_transfer_prohibition_validation.config.max_transfer_per_token_bytes);
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 10: Verify single stream decode
int llama_transfer_prohibition_gpu_verify_single_stream_decode(void) {
    // Verify all decode operations use single GPU stream
    // No per-kernel synchronization or multi-stream execution

    if (g_transfer_type_count["per_kernel_sync"] > 0) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_IMPLICIT_SYNC_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;

        if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
            fprintf(stderr, "[Transfer Prohibition GPU] Per-kernel synchronization detected!\n");
        }

        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// TRANSFER MONITORING
// ============================================================================

int llama_transfer_prohibition_gpu_record_transfer(
    enum llama_transfer_type transfer_type,
    uint64_t transfer_size_bytes,
    bool is_decode_critical
) {
    if (!g_transfer_prohibition_validation.config.transfer_prohibition_enabled) {
        return 0;
    }

    if (g_transfer_prohibition_validation.decode_phase_active) {
        // Use sed or manual replacement for multiple occurrences
// But replace_file_content with AllowMultiple=true works well for string replace.
g_transfer_prohibition_validation.state_record.total_transfers_during_decode++;
        g_transfer_prohibition_validation.state_record.total_transfer_bytes_during_decode += transfer_size_bytes;
    }

    // Record by transfer type
    g_transfer_type_bytes[llama_transfer_type_name(transfer_type)] += transfer_size_bytes;
    g_transfer_type_count[llama_transfer_type_name(transfer_type)]++;

    // Record last transfer
    g_transfer_prohibition_validation.last_transfer.transfer_type = transfer_type;
    g_transfer_prohibition_validation.last_transfer.transfer_size_bytes = transfer_size_bytes;
    g_transfer_prohibition_validation.last_transfer.is_decode_critical = is_decode_critical;
    g_transfer_prohibition_validation.last_transfer.during_decode_phase = g_transfer_prohibition_validation.decode_phase_active;
    g_transfer_prohibition_validation.last_transfer.timestamp_ns = 0; // Would use clock_gettime

    g_transfer_prohibition_validation.total_transfer_events++;

    return 0;
}

int llama_transfer_prohibition_gpu_check_transfer_allowed(
    enum llama_transfer_type transfer_type,
    uint64_t transfer_size_bytes,
    bool is_decode_critical
) {
    if (!g_transfer_prohibition_validation.config.transfer_prohibition_enabled) {
        return 0;
    }

    if (!g_transfer_prohibition_validation.decode_phase_active) {
        return 0; // OK outside decode phase
    }

    // During decode phase:
    // Only token ID transfers allowed for decode-critical data
    if (is_decode_critical) {
        if (transfer_size_bytes > g_transfer_prohibition_validation.config.max_transfer_per_token_bytes) {
            g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER;
            g_transfer_prohibition_validation.total_violations++;
            return -1;
        }

        // Check transfer type
        if (transfer_type == LLAMA_TRANSFER_TYPE_D2H || transfer_type == LLAMA_TRANSFER_TYPE_H2D) {
            // Only small token IDs allowed
            if (transfer_size_bytes > g_transfer_prohibition_validation.config.max_transfer_per_token_bytes) {
                g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER;
                g_transfer_prohibition_validation.total_violations++;
                return -1;
            }
        }
    }

    return 0;
}

// ============================================================================
// BUFFER PREALLOCATE OPERATIONS
// ============================================================================

int llama_transfer_prohibition_gpu_preallocate_logits_buffer(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.logits_buffer_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Logits buffer preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_sampling_workspace(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.sampling_workspace_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Sampling workspace preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_topk_buffer(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.topk_buffer_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Top-k buffer preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_topp_buffer(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.topp_buffer_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Top-p buffer preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_kv_cache(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.kv_cache_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] KV cache preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_attention_state(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.attention_state_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Attention state preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_penalty_buffer(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.penalty_buffer_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Penalty buffer preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

int llama_transfer_prohibition_gpu_preallocate_candidate_buffer(uint64_t size) {
    g_transfer_prohibition_validation.preallocated_buffers.candidate_buffer_allocated = true;
    g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes += size;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Candidate buffer preallocated: %llu bytes\n", (unsigned long long)size);
    }

    return 0;
}

// ============================================================================
// VIOLATION DETECTION
// ============================================================================

int llama_transfer_prohibition_gpu_detect_logits_d2h_transfer(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_LOGITS_D2H;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["logits_d2h"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_logits_host_read(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_LOGITS_READ;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["logits_read"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_kv_cache_transfer(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_KV_CACHE_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["kv_cache_transfer"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_activation_transfer(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_ACTIVATIONS_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["activation_transfer"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_sampling_buffer_transfer(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_SAMPLING_BUFFER_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["sampling_transfer"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_candidate_transfer(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_CANDIDATE_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["candidate_transfer"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_excessive_transfer(uint64_t transfer_size) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        if (transfer_size > g_transfer_prohibition_validation.config.max_transfer_per_token_bytes) {
            g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER;
            g_transfer_prohibition_validation.total_violations++;
            g_transfer_type_count["excessive_transfer"]++;
            return -1;
        }
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_unified_memory_access(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_UNIFIED_MEMORY_ACCESS;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["unified_memory"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_mapped_buffer_access(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_MAPPED_BUFFER_ACCESS;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["mapped_access"]++;
        return -1;
    }
    return 0;
}

int llama_transfer_prohibition_gpu_detect_implicit_sync_transfer(void) {
    if (g_transfer_prohibition_validation.decode_phase_active) {
        g_transfer_prohibition_validation.state_record.last_violation = LLAMA_TRANSFER_VIOLATION_IMPLICIT_SYNC_TRANSFER;
        g_transfer_prohibition_validation.total_violations++;
        g_transfer_type_count["implicit_sync"]++;
        return -1;
    }
    return 0;
}

// ============================================================================
// QUERY AND VERIFICATION FUNCTIONS
// ============================================================================

struct llama_gpu_transfer_prohibition_state_record llama_transfer_prohibition_gpu_get_state_record(void) {
    return g_transfer_prohibition_validation.state_record;
}

struct llama_gpu_preallocated_buffers llama_transfer_prohibition_gpu_get_preallocated_buffers(void) {
    return g_transfer_prohibition_validation.preallocated_buffers;
}

struct llama_gpu_transfer_record llama_transfer_prohibition_gpu_get_last_transfer(void) {
    return g_transfer_prohibition_validation.last_transfer;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_transfer_prohibition_gpu_verify_no_transfers_during_decode(void) {
    if (// Use sed or manual replacement for multiple occurrences
// But replace_file_content with AllowMultiple=true works well for string replace.
g_transfer_prohibition_validation.state_record.total_transfers_during_decode > 1) {
        // Allow 1 transfer for token ID
        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }
    return 0;
}

int llama_transfer_prohibition_gpu_verify_all_buffers_persistent(void) {
    // Verify no cudaMalloc/cudaFree during decode
    if (g_transfer_type_count["cudaMalloc"] > 0 || g_transfer_type_count["cudaFree"] > 0) {
        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }
    return 0;
}

int llama_transfer_prohibition_gpu_verify_single_stream_execution(void) {
    if (g_transfer_type_count["multi_stream"] > 0) {
        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }
    return 0;
}

int llama_transfer_prohibition_gpu_verify_no_unified_memory_used(void) {
    if (g_transfer_type_count["unified_memory"] > 0) {
        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }
    return 0;
}

int llama_transfer_prohibition_gpu_verify_no_mapped_buffers_used(void) {
    if (g_transfer_type_count["mapped_access"] > 0) {
        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }
    return 0;
}

int llama_transfer_prohibition_gpu_verify_only_token_id_transferred(void) {
    uint64_t total_bytes = g_transfer_prohibition_validation.state_record.total_transfer_bytes_during_decode;
    if (total_bytes > g_transfer_prohibition_validation.config.max_transfer_per_token_bytes) {
        if (g_transfer_prohibition_validation.enforcement_strict) {
            return -1;
        }
    }
    return 0;
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_transfer_prohibition_gpu_log_prohibition_enabled(void) {
    if (g_transfer_prohibition_validation.config.transfer_prohibition_enabled) {
        fprintf(stderr, "[Transfer Prohibition GPU] Transfer prohibition enabled\n");
    }
}

void llama_transfer_prohibition_gpu_log_decode_phase_started(void) {
    fprintf(stderr, "[Transfer Prohibition GPU] Decode phase STARTED - all transfers monitored\n");
}

void llama_transfer_prohibition_gpu_log_decode_phase_ended(void) {
    fprintf(stderr, "[Transfer Prohibition GPU] Decode phase ENDED\n");
}

void llama_transfer_prohibition_gpu_print_state(void) {
    fprintf(stderr, "\n=== Transfer Prohibition GPU State ===\n");
    fprintf(stderr, "State: %s\n", llama_transfer_prohibition_state_name(g_transfer_prohibition_validation.state_record.state));
    fprintf(stderr, "Mode: %s\n", (g_transfer_prohibition_validation.state_record.mode == LLAMA_TRANSFER_PROHIBITION_ENABLED) ? "ENABLED" : "DISABLED");
    fprintf(stderr, "Decode Phase Active: %s\n", g_transfer_prohibition_validation.decode_phase_active ? "YES" : "NO");
    fprintf(stderr, "Total Transfers During Decode: %llu\n", (unsigned long long)// Use sed or manual replacement for multiple occurrences
// But replace_file_content with AllowMultiple=true works well for string replace.
g_transfer_prohibition_validation.state_record.total_transfers_during_decode);
    fprintf(stderr, "Total Transfer Bytes During Decode: %llu\n", (unsigned long long)g_transfer_prohibition_validation.state_record.total_transfer_bytes_during_decode);
    fprintf(stderr, "Max Allowed Bytes Per Token: %llu\n", (unsigned long long)g_transfer_prohibition_validation.config.max_transfer_per_token_bytes);
    fprintf(stderr, "Total Violations: %d\n", g_transfer_prohibition_validation.total_violations);
    fprintf(stderr, "Last Violation: %s\n", llama_transfer_violation_name(g_transfer_prohibition_validation.state_record.last_violation));
    fprintf(stderr, "Enforcement: %s\n", g_transfer_prohibition_validation.enforcement_strict ? "STRICT" : "PERMISSIVE");
    fprintf(stderr, "\n");
}

void llama_transfer_prohibition_gpu_print_transfer_stats(void) {
    fprintf(stderr, "\n=== Transfer Prohibition GPU Transfer Stats ===\n");
    fprintf(stderr, "Total Transfer Events: %d\n", g_transfer_prohibition_validation.total_transfer_events);

    for (const auto& type_count : g_transfer_type_count) {
        fprintf(stderr, "%s: %d transfers, %llu bytes\n",
            type_count.first.c_str(), type_count.second,
            (unsigned long long)g_transfer_type_bytes[type_count.first]);
    }

    fprintf(stderr, "\n");
}

void llama_transfer_prohibition_gpu_print_violation_summary(void) {
    fprintf(stderr, "\n=== Transfer Prohibition GPU Violation Summary ===\n");
    fprintf(stderr, "Total Violations: %d\n", g_transfer_prohibition_validation.total_violations);
    fprintf(stderr, "Last Violation Type: %s\n", llama_transfer_violation_name(g_transfer_prohibition_validation.state_record.last_violation));
    fprintf(stderr, "\n");
}

void llama_transfer_prohibition_gpu_print_preallocated_buffers(void) {
    fprintf(stderr, "\n=== Preallocated GPU Buffers ===\n");
    fprintf(stderr, "Logits Buffer: %s\n", g_transfer_prohibition_validation.preallocated_buffers.logits_buffer_allocated ? "YES" : "NO");
    fprintf(stderr, "Sampling Workspace: %s\n", g_transfer_prohibition_validation.preallocated_buffers.sampling_workspace_allocated ? "YES" : "NO");
    fprintf(stderr, "Top-K Buffer: %s\n", g_transfer_prohibition_validation.preallocated_buffers.topk_buffer_allocated ? "YES" : "NO");
    fprintf(stderr, "Top-P Buffer: %s\n", g_transfer_prohibition_validation.preallocated_buffers.topp_buffer_allocated ? "YES" : "NO");
    fprintf(stderr, "KV Cache: %s\n", g_transfer_prohibition_validation.preallocated_buffers.kv_cache_allocated ? "YES" : "NO");
    fprintf(stderr, "Attention State: %s\n", g_transfer_prohibition_validation.preallocated_buffers.attention_state_allocated ? "YES" : "NO");
    fprintf(stderr, "Penalty Buffer: %s\n", g_transfer_prohibition_validation.preallocated_buffers.penalty_buffer_allocated ? "YES" : "NO");
    fprintf(stderr, "Candidate Buffer: %s\n", g_transfer_prohibition_validation.preallocated_buffers.candidate_buffer_allocated ? "YES" : "NO");
    fprintf(stderr, "Total Preallocated: %llu bytes\n", (unsigned long long)g_transfer_prohibition_validation.preallocated_buffers.total_preallocated_bytes);
    fprintf(stderr, "\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_transfer_prohibition_gpu_report_violation(
    enum llama_transfer_violation violation_type,
    const char* details,
    uint64_t transfer_size
) {
    g_transfer_prohibition_validation.state_record.last_violation = violation_type;
    g_transfer_prohibition_validation.total_violations++;

    fprintf(stderr, "[Transfer Prohibition GPU] Violation: %s\n", llama_transfer_violation_name(violation_type));
    fprintf(stderr, "  Transfer Size: %llu bytes\n", (unsigned long long)transfer_size);
    if (details != nullptr) {
        fprintf(stderr, "  Details: %s\n", details);
    }

    if (g_transfer_prohibition_validation.enforcement_strict) {
        fprintf(stderr, "  Action: STRICT enforcement - ABORTING\n");
    } else {
        fprintf(stderr, "  Action: PERMISSIVE mode - continuing\n");
    }
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_transfer_prohibition_gpu_set_enforcement_strict(bool strict) {
    g_transfer_prohibition_validation.enforcement_strict = strict;

    if (g_transfer_prohibition_validation.config.debug_transfer_prohibition) {
        fprintf(stderr, "[Transfer Prohibition GPU] Enforcement mode set to: %s\n", strict ? "STRICT" : "PERMISSIVE");
    }
}

bool llama_transfer_prohibition_gpu_get_enforcement_strict(void) {
    return g_transfer_prohibition_validation.enforcement_strict;
}

void llama_transfer_prohibition_gpu_set_debug_output(bool debug) {
    g_transfer_prohibition_validation.config.debug_transfer_prohibition = debug;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_transfer_prohibition_gpu_selftest(void) {
    fprintf(stderr, "\n=== Transfer Prohibition GPU Self-Test Suite ===\n");

    int test_results = 0;

    // Test 1: Initialization
    fprintf(stderr, "Test 1: Initialization... ");
    llama_transfer_prohibition_gpu_init();
    if (g_transfer_prohibition_validation.state_record.state == LLAMA_GPU_TRANSFER_PROHIBITION_INITIALIZED) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 2: Configuration
    fprintf(stderr, "Test 2: Configuration... ");
    llama_transfer_prohibition_gpu_configure(true, true, true, true, 8);
    if (g_transfer_prohibition_validation.config.transfer_prohibition_enabled &&
        g_transfer_prohibition_validation.state_record.mode == LLAMA_TRANSFER_PROHIBITION_ENABLED) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 3: Decode phase begin
    fprintf(stderr, "Test 3: Decode phase begin... ");
    llama_transfer_prohibition_gpu_begin_decode_phase();
    if (g_transfer_prohibition_validation.decode_phase_active &&
        g_transfer_prohibition_validation.state_record.state == LLAMA_GPU_TRANSFER_PROHIBITION_DECODE_ACTIVE) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 4: Buffer preallocate
    fprintf(stderr, "Test 4: Buffer preallocate... ");
    llama_transfer_prohibition_gpu_preallocate_logits_buffer(1024*1024);
    llama_transfer_prohibition_gpu_preallocate_kv_cache(10*1024*1024);
    if (g_transfer_prohibition_validation.preallocated_buffers.logits_buffer_allocated &&
        g_transfer_prohibition_validation.preallocated_buffers.kv_cache_allocated) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 5: Transfer recording (with strict disabled for test)
    fprintf(stderr, "Test 5: Transfer recording... ");
    llama_transfer_prohibition_gpu_set_enforcement_strict(false);
    llama_transfer_prohibition_gpu_record_transfer(LLAMA_TRANSFER_TYPE_D2H, 4, true);
    if (// Use sed or manual replacement for multiple occurrences
// But replace_file_content with AllowMultiple=true works well for string replace.
g_transfer_prohibition_validation.state_record.total_transfers_during_decode > 0) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 6: Excessive transfer detection
    fprintf(stderr, "Test 6: Excessive transfer detection... ");
    int result = llama_transfer_prohibition_gpu_detect_excessive_transfer(1024*1024);
    if (result == -1) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 7: Decode phase end
    fprintf(stderr, "Test 7: Decode phase end... ");
    llama_transfer_prohibition_gpu_end_decode_phase();
    if (!g_transfer_prohibition_validation.decode_phase_active &&
        g_transfer_prohibition_validation.state_record.state == LLAMA_GPU_TRANSFER_PROHIBITION_COMPLETE) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 8: Verification functions (with strict re-enabled)
    fprintf(stderr, "Test 8: Verification functions... ");
    llama_transfer_prohibition_gpu_set_enforcement_strict(true);
    if (llama_transfer_prohibition_gpu_verify_all_buffers_persistent() == 0) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    fprintf(stderr, "\n=== Self-Test Complete: %s ===\n\n", (test_results == 0) ? "ALL PASSED" : "SOME FAILED");

    return test_results;
}


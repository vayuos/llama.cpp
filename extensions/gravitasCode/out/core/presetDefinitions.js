"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GRAVITAS_PRESETS = void 0;
exports.GRAVITAS_PRESETS = {
    'default': {
        label: 'Default (Mixed)',
        description: 'Standard configuration for Gravitas',
        config: {
            'coder.hardware.contextSize': 8192,
            'reviewer.hardware.contextSize': 8192,
            'coder.sampling.temperature': 0.2,
            'reviewer.sampling.temperature': 0.0
        }
    },
    'high_accuracy': {
        label: 'High Accuracy',
        description: 'Deterministic sampling for Reviewer',
        config: {
            'coder.hardware.contextSize': 8192,
            'coder.sampling.temperature': 0.2,
            'coder.sampling.topP': 0.9,
            'reviewer.hardware.contextSize': 8192,
            'reviewer.sampling.temperature': 0.0,
            'reviewer.sampling.topK': 1
        }
    },
    'fast_draft': {
        label: 'Fast Draft',
        description: 'Optimized context for rapid iteration',
        config: {
            'coder.hardware.contextSize': 8192,
            'coder.sampling.temperature': 0.3,
            'reviewer.hardware.contextSize': 8192,
            'reviewer.sampling.temperature': 0.0
        }
    },
    'safe_mode': {
        label: 'Safe Mode',
        description: 'Standard context with zero temperature',
        config: {
            'coder.hardware.contextSize': 8192,
            'coder.sampling.temperature': 0.0,
            'reviewer.hardware.contextSize': 8192,
            'reviewer.sampling.temperature': 0.0
        }
    }
};
//# sourceMappingURL=presetDefinitions.js.map